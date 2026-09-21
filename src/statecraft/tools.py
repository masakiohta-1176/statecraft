"""
普通のPython関数を、LLMから呼び出せる「ツール」にする。

引数のJSON Schemaは型ヒントから自動生成する（手で書くと二重管理になる）。
説明はsummary / usage / evaluationの3つに分かれており、提示されるタイミングが
それぞれ違う。書き方はREADMEを参照。

失敗は例外ではなくToolResultで返す。Agentが失敗そのものを次の判断材料に
できるようにするため（例外で中断すると、そこまでの作業も失われる）。
"""

import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .interceptor import Interceptor
from .utils import format_message, with_timing

# toolの戻り値に許されるキー。これ以外が入っていたら綴り違いを疑う。
_RESULT_KEYS = frozenset({"value", "blobs", "memory"})


def parse_tool_return(tool_name: str, returned: Any) -> "ToolResult":
    """
    toolの戻り値（辞書）を ToolResult へ変換する。

        {"value": str, "blobs": [...], "memory": {...}}   value以外は省略可

    形は1つに固定する。登録側の設定で形が変わると（「記憶へ書くtoolは差分の
    配列を返す」のような分岐）、valueの意味そのものが入れ替わる。
    契約違反は失敗として返し、valueは推測で埋めない。
    """
    if not isinstance(returned, dict):
        return ToolResult(
            error=(
                f"{tool_name}の戻り値が辞書ではありません（{type(returned).__name__}）。"
                'toolは {"value": ..., "blobs": [...], "memory": {...}} の形で返します。'
                "blobsとmemoryは省略できます。"
            )
        )

    unknown = set(returned) - _RESULT_KEYS
    if unknown:
        return ToolResult(
            error=(
                f"{tool_name}の戻り値に想定外のキーがあります: {sorted(unknown)}。"
                f"使えるのは {sorted(_RESULT_KEYS)} です。"
            )
        )

    if "value" not in returned:
        return ToolResult(error=f'{tool_name}の戻り値に "value" がありません。')

    blobs = returned.get("blobs") or []
    if not isinstance(blobs, list):
        return ToolResult(error=f"{tool_name}のblobsが配列ではありません。")

    memory = returned.get("memory") or {}
    if not isinstance(memory, dict):
        return ToolResult(error=f"{tool_name}のmemoryが辞書ではありません。")
    # 値は本文の配列。文字列を1つ渡された場合も配列として扱う
    # （1件だけ書く時に [] を付け忘れるのはよくある）。
    normalized: dict[str, list[str]] = {}
    for field_name, texts in memory.items():
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            return ToolResult(error=f"{tool_name}のmemory['{field_name}']が配列ではありません。")
        normalized[str(field_name)] = [str(t) for t in texts]

    return ToolResult(value=returned["value"], blobs=list(blobs), memory=normalized)


@dataclass
class ToolResult:
    """
    ツール（またはエージェント）の実行結果。

    成功でも失敗でも必ずこの形で返る。例外は外へ出さない。
    """

    # LLMへ返す結果。関数が返した辞書の "value" がそのまま入る。
    # Agentの委譲結果（Agent.execute）では回答文が入る。
    value: Any = None

    # 失敗時にLLMへ渡すエラーメッセージ。実際の例外メッセージを見せるかは
    # Tool.expose_error_details で切り替える（既定では見せない）。
    error: str | None = None
    execution_time: float = 0.0  # 実行にかかった時間（秒）。with_timingが自動で埋める

    # LLMへ見せる添付。[{"data": bytes, "mime_type": str}, ...]
    # 見えるのは結果を評価するphaseの1回だけで、記憶へ文字起こしされた後は
    # 載らない（文字起こし先を持たないReflexAgentでは渡り続ける）。
    blobs: list[dict] = field(default_factory=list)

    # LLMを介さず記憶へ直接書く内容。{プロパティ名: [本文, ...]}
    #   {"vars": ["予約受付番号: RSV-63082"], "facts": ["予約を登録した"]}
    #
    # 一字一句正確に残したい値のための経路（要約を通さないので値が変わらない）。
    # idは書かない。toolは既存の記憶を知らないため、システムが
    # tool名-プロパティ名-連番 で採番する。書き込むのはAgent側。
    memory: dict[str, list[str]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        errorの有無から導出する。boolを別に持つと
        ToolResult(success=True, error="失敗") のような矛盾した値が作れてしまう。
        """
        return self.error is None


# ---- Tool ----
@dataclass
class Tool:
    # 実行する関数本体。普通のPython関数をそのまま渡す。
    func: Callable

    # ツール選択の判定に使う概要。常時プロンプトに載るため1行に留める。
    summary: str

    # 各引数が何を意味するかの説明。{引数名: 説明}。
    # 型はsignatureから取れるが、意味は人間が書くしかない。
    param_descriptions: dict[str, str] = field(default_factory=dict)

    # 呼び出し候補として選ばれた時だけ渡す、詳細な使い方・注意点。
    # 常時は載らないので長く書いてよい。
    usage: str = ""

    # 結果を受け取った側が、それをどう解釈すべきかの指示。
    # 結果が返ってきた瞬間にだけ付与する。
    evaluation: str = ""

    # Trueの間、このツールは選択肢として提示されない。
    # セッション中にLLM自身が「今回は使わない」と宣言した場合にも立つ。
    disabled: bool = False

    # Trueなら実際の例外メッセージをそのままLLMへ返す。接続先やパスが
    # 混ざりうるため、既定は詳細を伏せるFalse。
    expose_error_details: bool = False

    # 実行中に観測側（利用者の画面など）へ流す文言。引数は {name} で差し込める。
    #   execution_message="『{title}』の所蔵状況を照会しています..."
    # 未指定なら開発者向けの汎用文。ツール自身が持つのは、受け取る側に
    # 名前ごとの対応表を作らせないため。
    execution_message: str = ""
    # 文言が引数だけでは決まらない場合（条件で出し分ける等）に渡す。
    # execution_messageより優先される。
    describe_execution: Callable[[dict], str] | None = None

    @property
    def name(self) -> str:
        """
        LLMへ提示する名前。func.__name__ を直接参照しないのは、
        functools.partial や __call__ を持つインスタンスが持たないため。
        """
        return getattr(self.func, "__name__", None) or type(self.func).__name__

    def _default_describe_execution(self, kwargs: dict) -> str:
        if self.execution_message:
            return format_message(self.execution_message, kwargs)
        args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{self.name}を{args_str}で実行しています..."

    @with_timing
    def execute(
        self,
        *,
        kwargs: dict,
        interceptor: Interceptor | None = None,
    ) -> ToolResult:
        """
        ツールを実行する。

        disabled=True なら interceptorの有無に関わらず即座に拒否する。実行前に
        check.before_execute()を通し、前後に execute_start / execute_end を
        通知する（文言はdescribe_executionが組み立てる）。
        例外は投げず、必ずToolResultで返す。
        """
        tool_name = self.name

        if self.disabled:
            return ToolResult(error="このツールは現在無効化されています。")

        if interceptor:
            verdict = interceptor.check.before_execute(name=tool_name, kwargs=kwargs)
            if not verdict.allowed:
                # 理由をそのままLLMへ返す。「ブロックされました」だけだと、
                # 止められた側が原因を探して追加の実行を始める。
                detail = f" 理由: {verdict.reason}" if verdict.reason else ""
                interceptor.notify.execute_blocked(
                    f"{tool_name}の実行がブロックされました。{detail}"
                )
                return ToolResult(error=verdict.reason or "実行がブロックされました。")

        if interceptor:
            # 文言の組み立ても失敗しうる（引数の__repr__が例外を投げる等）。
            # 通知の失敗で本体の実行を止めない。
            try:
                describe = self.describe_execution or self._default_describe_execution
                interceptor.notify.execute_start(describe(kwargs))
            except Exception as e:  # noqa: BLE001
                interceptor.notify.error(f"{tool_name}の実行中メッセージの生成に失敗しました: {e}")

        try:
            returned = self.func(**kwargs)
            # 戻り値の契約はここで一度だけ解釈する。以降はToolResultしか見ない。
            result = parse_tool_return(tool_name, returned)
            if not result.success and interceptor:
                # 契約違反は実装の誤りなので、観測側へも出す。
                interceptor.notify.error(result.error or "")
        except Exception as e:  # noqa: BLE001
            # 観測者へは実際の例外を伝える（隠す相手はLLMだけ）。
            if interceptor:
                interceptor.notify.error(f"{tool_name}の実行が失敗しました: {e}")
            error_message = (
                str(e) if self.expose_error_details else "ツールの実行中にエラーが発生しました。"
            )
            result = ToolResult(error=error_message)

        if interceptor:
            status = "完了しました" if result.success else "エラーで終了しました"
            interceptor.notify.execute_end(f"{tool_name}の実行が{status}。")

        return result

    def to_catalog_line(self, *, include_usage: bool = False) -> str:
        """
        「こういうtoolが存在する」ことを伝える行。

        include_usage=False（既定）は名前とsummaryだけ。孫として見せる場合に使う。
        Trueは自分の直下向けで、tasksを立てる・評価するphaseで使い方まで見せる。
        """
        line = f"- {self.name}: {self.summary}"
        if include_usage and self.usage:
            line += f"\n  使い方: {self.usage}"
        return line

    def to_declaration(self) -> dict:
        """
        LLM APIへ渡す、プロバイダ非依存の定義情報。
        各LLMクラスの_format_toolsがこれを自社の形式へ変換する。

        descriptionはsummaryとusageを結合する。APIのtools引数へ渡るのは
        呼び出し候補になった時だけなので、詳細を載せても常時は載らない。
        """
        description = "\n".join(p for p in (self.summary, self.usage) if p)
        return {
            "name": self.name,
            "description": description,
            "parameters": self.get_json_schema(),
        }

    def get_json_schema(self) -> dict:
        """関数の型ヒントから標準 JSON Schema パラメータを自動生成"""
        # eval_str=Trueが無いと、利用側が `from __future__ import annotations`
        # を書いているだけで注釈が文字列になり、全引数が"string"へ落ちる。
        try:
            sig = inspect.signature(self.func, eval_str=True)
        except (NameError, TypeError):
            # 解決できない前方参照などがある場合は、文字列のまま扱う
            sig = inspect.signature(self.func)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # *args / **kwargs はJSON Schemaのプロパティとして表現できないので除外する
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            properties[name] = self._schema_for(param.annotation)
            if name in self.param_descriptions:
                properties[name]["description"] = self.param_descriptions[name]
            # Parameter.emptyはシングルトンなので is で比較する。
            # == だと、独自の__eq__を持つ既定値（numpy配列等）で例外になる。
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # Pythonの型 → JSON Schemaの型名。ClassVarが無いとdataclassのフィールドと
    # 見なされ、Tool(...)の引数になってしまう。
    _TYPE_MAP: ClassVar[dict[type, str]] = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        list: "array",
        dict: "object",
    }

    @classmethod
    def _schema_for(cls, annotation) -> dict:
        """
        1つの型注釈をJSON Schemaへ変換する。

        list[str] のようなジェネリクスは list とは別の型オブジェクトなので、
        辞書引きだけでは"string"へ落ちる（LLMが配列を文字列として返す）。
        get_originで元の型へ戻してから判定し、要素の型もitemsへ入れる。
        """
        origin = typing.get_origin(annotation)

        # Optional[X] は None を除いた本来の型で判定する。
        # 判定が2つ必要なのは、書き方で get_origin の戻り値が変わるため。
        #   Optional[list[str]] → typing.Union
        #   list[str] | None    → types.UnionType（3.11〜3.13。3.14では同一）
        if origin is typing.Union or origin is types.UnionType:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if not args:
                return {"type": "string"}
            return cls._schema_for(args[0])

        base = origin or annotation
        json_type = cls._TYPE_MAP.get(base, "string")
        # Anyの注釈が必要なのは、下でitemsへ入れ子の辞書を入れるため
        # （注釈が無いと dict[str, str] と推論される）。
        schema: dict[str, Any] = {"type": json_type}

        # itemsを要求するプロバイダがあるため、要素型が不明でもstringで補う。
        if json_type == "array":
            args = typing.get_args(annotation)
            schema["items"] = cls._schema_for(args[0]) if args else {"type": "string"}

        return schema


