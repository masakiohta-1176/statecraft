"""
普通のPython関数を、LLMから呼び出せる「ツール」にする。


    def check_stock(title: str) -> str:
        '''指定した書名の在庫を調べる'''
        ...


    tool = Tool(
        func=check_stock,
        summary="書名から在庫を調べる",
        param_descriptions={"title": "調べたい書名"},
    )


【引数のスキーマは書かない】
LLMへ渡す引数の定義（JSON Schema）は、関数の型ヒントから自動生成される。
手で書くと関数の実装とスキーマが二重管理になり、片方だけ直した時に
静かに食い違う。型ヒントを唯一の正とする。


    def f(topic: str, limit: int = 10)
      → {"topic": {"type": "string"}, "limit": {"type": "integer"}}
        required は既定値の無い引数だけ（この例では topic のみ）


型からは「意味」が分からないため、param_descriptionsだけは人間が書く。


【説明を3つに分けている】
提示されるタイミングが違うため、別のプロパティにしている。


    summary     常時。「こういうツールが存在する」ことだけを伝える1行
    usage       呼び出し候補として選ばれた時だけ。詳細な使い方や注意点
    evaluation  結果が返ってきた瞬間だけ。その結果をどう解釈すべきか


全部を常時提示すると、ツールが増えるたびにプロンプトが伸び続ける。
必要な瞬間にだけ渡すことで、常時のトークンを使わずに済む。


【失敗しても例外を投げない】
ツールの実行が失敗しても、例外は呼び出し側へ出さない。
必ずToolResultとして「失敗した」という事実を返す。
そうすることで、Agentは失敗そのものを次の判断材料にできる
（「その方法では取得できなかった」と記憶へ書いて別の手を考える）。
例外で中断すると、そこまでの作業も失われる。
"""


import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar


from .interceptor import Interceptor
from .utils import format_message, with_timing




@dataclass
class ToolResult:
    """
    ツール（またはエージェント）の実行結果。


    成功でも失敗でも必ずこの形で返る。例外は外へ出さない。
    Agentはsuccessを見て、失敗なら「その手段では取れなかった」という
    事実として記憶へ書き、別の方法を考えられる。
    """


    # 成功時の戻り値。関数がそのまま返したもの。
    value: Any = None


    # 失敗時にLLMへ渡すエラーメッセージ。
    # 実際の例外メッセージをそのまま見せるかどうかは、
    # Tool.expose_error_details で切り替える（既定では見せない）。
    error: str | None = None
    execution_time: float = 0.0  # 実行にかかった時間（秒）。with_timingが自動で埋める


    @property
    def success(self) -> bool:
        """
        errorの有無から導出する。boolを別に持つと
        ToolResult(success=True, error="失敗") のような矛盾した値が作れてしまうため、
        真実の出所をerrorだけに一本化する。
        @propertyなので、呼び出し側は result.success と属性のまま書ける。
        """
        return self.error is None




# ==========================================
# Tool: AIエージェントが使う「道具」の定義
# ==========================================
@dataclass
class Tool:
    # 実行する関数本体。普通のPython関数をそのまま渡す。
    func: Callable


    # ツール選択の判定に使う概要。
    # 常時プロンプトに載る軽い情報なので、
    # 「このツールが存在する」ことだけ伝わればよい。
    summary: str


    # 各引数が何を意味するかの説明。{引数名: 説明} の形。
    # 型はinspect.signatureから自動生成できるが、
    # 意味までは人間が教えるしかないため明示的に渡す。
    param_descriptions: dict[str, str] = field(default_factory=dict)


    # 実際にこのツールを使うと決まった時だけ渡す、詳細な使い方・注意点。
    # 常時は載せないので、ここに長く書いても普段のトークンには影響しない。
    usage: str = ""


    # このツールの結果を受け取った側が、それをどう解釈すべきかの指示。
    # システムプロンプトへ焼き込まず、結果が返ってきた瞬間にだけ動的に付与する。
    evaluation: str = ""


    # Trueの間、このツールは選択肢として提示されない。
    # セッション中にLLM自身が「今回は使わない」と宣言した場合にも立つ。
    disabled: bool = False


    # このツールの結果を、LLMを介さず直接memoryへ書き込むかどうか。
    # Trueにする場合、funcはLLMの差分と同じ形の配列を返すこと。
    #   [{"field": "facts", "text": "..."}, {"field": "vars", "text": "..."}]
    # idは省略できる（システムが tool名-field-連番 で自動採番する）。
    # 同じ形にしてあるので、書き込み処理はapply_diffがそのまま使える。
    #
    # LLMに要約させずに記録できるため、要約による欠落と1回分の推論が消える。
    # 実行するSQLや取得したIDのように、一字一句正確に残したい値に向く。
    # 実際の書き込みはAgent側が行い、Toolはこの設定を持つだけ。
    write_to_memory: bool = False


    # Trueなら実際の例外メッセージをそのままLLMへ返す。
    # Falseなら詳細を伏せ、汎用的な失敗メッセージだけを返す。
    # 例外メッセージには接続先やパスなどが混ざりうるため、既定は安全側のFalse。
    expose_error_details: bool = False


    # 実行中に観測側（利用者の画面など）へ流す文言。
    # 引数は {name} の形で差し込める。
    #   execution_message="『{title}』の所蔵状況を照会しています..."
    # 未指定なら開発者向けの汎用文になる。
    #
    # 文言はツール自身が持つ。受け取る側に
    #   if name == "check_stock": ... elif name == "search_rules": ...
    # という対応表を作ると、ツールを1つ足すたびに2箇所を直すことになり、
    # 忘れた時に「無言のまま止まる」形で壊れる。
    execution_message: str = ""
    # 文言が引数の値だけでは決まらない場合（条件で出し分けたい等）だけ関数を渡す。
    # 指定するとexecution_messageより優先される。ほとんどの場合は不要。
    describe_execution: Callable[[dict], str] | None = None


    @property
    def name(self) -> str:
        """
        LLMへ提示する名前。


        func.__name__ を直接参照しない。Callableには
        functools.partial や __call__ を持つインスタンスも含まれ、
        それらは __name__ を持たないためAttributeErrorになる。
        型注釈が Callable である以上、名前を持たない関数も正当な入力。
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


        - disabled=True の場合は、interceptorの有無に関わらず即座に拒否する
          （AIがツール名を推測・記憶していて直接呼ぼうとした場合の最後の砦）。
        - 実行前に interceptor.check.before_execute() を通し、
          Falseが返れば実行そのものをブロックする（拒否権あり）。
        - 実行前後には interceptor.notify.execute_start()/execute_end() で
          純粋な通知を行う。メッセージの中身は describe_execution（Toolごとに
          カスタマイズ可能）が組み立て、Interceptor自身は文言を一切持たない。
        - 例外は投げず、必ずToolResultとして成功/失敗を返す。エラーの
          詳細をAgentに見せるかどうかはexpose_error_detailsで制御する
          （情報漏洩防止のため、デフォルトでは詳細を返さない）。
        """
        tool_name = self.name


        if self.disabled:
            return ToolResult(error="このツールは現在無効化されています。")


        if interceptor:
            verdict = interceptor.check.before_execute(name=tool_name, kwargs=kwargs)
            if not verdict.allowed:
                # 理由があれば、それをそのままLLMへ返す。
                # 「ブロックされました」だけを返すと、止められた側は原因を
                # 探すために追加の実行を始める（引数を直せば通る場合でも）。
                detail = f" 理由: {verdict.reason}" if verdict.reason else ""
                interceptor.notify.execute_blocked(
                    f"{tool_name}の実行がブロックされました。{detail}"
                )
                return ToolResult(error=verdict.reason or "実行がブロックされました。")


        if interceptor:
            # 実行中メッセージの組み立ても失敗しうる（引数の__repr__が例外を投げる、
            # describe_execution自体に不具合がある等）。通知の失敗で本体の実行を
            # 止めないよう、ここも保護する。
            try:
                describe = self.describe_execution or self._default_describe_execution
                interceptor.notify.execute_start(describe(kwargs))
            except Exception as e:
                interceptor.notify.error(f"{tool_name}の実行中メッセージの生成に失敗しました: {e}")


        try:
            value = self.func(**kwargs)
            result = ToolResult(value=value)
        except Exception as e:
            # LLMへ返す文言はexpose_error_detailsで制御するが、
            # 観測者へは実際の例外をそのまま伝える。隠す相手はLLMであり、
            # 運用する人間から原因を隠す理由はない。
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


    def to_catalog_line(self) -> str:
        """
        「こういうtoolが存在する」ことだけを伝える1行。
        usageは実際に使うと決まった時にだけ渡す詳細なので、ここには含めない。
        """
        return f"- {self.name}: {self.summary}"


    def to_declaration(self) -> dict:
        """LLM APIに渡すための、プロバイダに依存しない汎用的な定義情報。
        各LLMクラスの_format_toolsは、これを自社の形式に変換するだけでよい。


        descriptionはsummaryとusageを結合する。
        実際に呼び出す段になって初めてAPIのtools引数へ渡るため、
        ここで詳細を載せても常時プロンプトに載るわけではない。
        軽い一覧（to_catalog_line）とはこの点で使い分ける。
        """
        description = "\n".join(p for p in (self.summary, self.usage) if p)
        return {
            "name": self.name,
            "description": description,
            "parameters": self.get_json_schema(),
        }


    def get_json_schema(self) -> dict:
        """関数の型ヒントから標準 JSON Schema パラメータを自動生成"""
        # eval_str=Trueを付けないと、利用側のモジュールが
        # `from __future__ import annotations` を書いているだけで
        # 注釈が文字列（'int'）になり、全引数が"string"へフォールバックする。
        # 型ヒントを唯一の正とする設計が、利用側の1行で静かに崩れるのを防ぐ。
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


    # Pythonの型 → JSON Schemaの型名。
    # ClassVarにしているのは、Toolがdataclassであるため。
    # 注釈を付けないとフィールドと見なされ、Tool(...)の引数として
    # 受け取れる形になってしまう（全インスタンスで共有される辞書が
    # 初期値になる、という危険も伴う）。
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


        list[str] のようなパラメータ化ジェネリクスは、型オブジェクトとしては
        list とは別物になる。素の辞書引きだけで判定すると全て"string"へ
        フォールバックし、LLMが配列を文字列として返して静かに壊れる。
        get_originで元の型へ戻してから判定し、要素の型もitemsへ反映する。
        """
        origin = typing.get_origin(annotation)


        # Optional[X] (= Union[X, None]) は None を除いた本来の型で判定する
        if origin is typing.Union:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if not args:
                return {"type": "string"}
            return cls._schema_for(args[0])


        base = origin or annotation
        json_type = cls._TYPE_MAP.get(base, "string")
        # 値の型をAnyにしているのは、下でitemsへ辞書（入れ子のスキーマ）を
        # 入れるため。注釈が無いと右辺から dict[str, str] と推論され、
        # 文字列しか入らない辞書として扱われてしまう。
        schema: dict[str, Any] = {"type": json_type}


        # 配列はitemsを要求するプロバイダがあるため、要素の型まで埋める。
        # 要素型が不明な素のlistでも、itemsを欠かさないようstringで補う。
        if json_type == "array":
            args = typing.get_args(annotation)
            schema["items"] = cls._schema_for(args[0]) if args else {"type": "string"}


        return schema





