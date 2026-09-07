import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union, get_args, get_origin

from interceptor import Interceptor
from utils import format_message, with_timing

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    value: Any = None  #  成功時の戻り値
    error: str | None = None
    execution_time: float = 0.0

    @property
    def success(self):
        """errorの有無で成功判定"""
        return self.error is None


# ========================================
# Tool: AIエージェントが使う「道具」の定義
# ========================================
@dataclass
class Tool:
    # 実行する関数本体
    func: Callable

    # 何をするtoolなのかという短い説明
    summary: str

    # 実際に使用できる権限を持つエージェントに対する詳細な使い方の説明文
    usage: str
    param_descriptions: dict[str, str] = field(default_factory=dict)
    evaluation: str = (
        ""  # tool実行後、結果に添えるプロンプト。この結果をどう評価すべきかなどを記載。
    )
    disabled: bool = False  # 特定のタイミングでエージェントからこのツールを見えなくする/無効化するためのフラグ。たまたまAIが呼んでしまっても実行しない。
    # toolの戻り値をstate内に直接書き込むかどうか。
    # IDや一言一句変えてほしくない文言など、LLMを通さず直接書き込むことによって変質を防ぐ。
    #
    # 基本はFalse推奨。
    # Trueにする場合は
    # funcの戻り値を[{field:"facts,"text":"..."},{field:"vars,"text":"..."}]の形式にすること。
    write_to_memory: bool = False
    # True時、実際のtoolの例外メッセージを「エージェントも」見せる。
    # Falseの場合は汎用的な失敗メッセージのみを返す。
    expose_error_details: bool = False
    # 実行中にUIなどへ流す文言。
    # デフォルトではf"{self.name}を{key=value}で実行しています。"
    # 引数は{name}の形式で
    # 例:"{title}の状況を確認しています..."

    execution_message: str = ""
    # 文言が引数の値だけでは決まらない（条件分岐など）だけ関数を渡す。
    # 指定した場合はexecution_messageより優先される。

    describe_execution: Callable[[dict], str] | None = None
    _TYPE_MAP: ClassVar[dict[type, str]] = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        list: "array",
        dict: "object",
    }

    @property
    def name(self):
        """LLMに対して提示するtool名称"""
        return getattr(self.func, "__name__", None) or type(self.func).__name__

    def _default_describe_execution(self, kwargs: dict) -> str:
        if self.execution_message:
            return format_message(self.execution_message, kwargs)
        args_str = ",".join(f"{k}={v!r} " for k, v in kwargs.items())
        return f"{self.name}を{args_str}で実行しています。"

    @with_timing
    def execute(
        self, *, kwargs: dict, interceptor: Interceptor | None = None
    ) -> ToolResult:
        """ツールを実行する
        責務↓
        - disabledがtrueの時、interceptorに関わらず即座に拒否、
        - 実行前にinterceptor._check.before_executeを通して、Falseが返ってきたら実行を停止
        - 実行前後にnotifyで通知を行う。
        - 例外で止めることはせず、必ずToolResultとして成功/失敗を返す。
        """
        tool_name = self.name
        if self.disabled:
            return ToolResult(error="このツールは現在無効化されています。")

        if interceptor and not interceptor.check.before_execute(
            name=tool_name, kwargs=kwargs
        ):
            interceptor.notify.execute_blocked(f"{tool_name}の実行がブロックされました")
            return ToolResult(error="実行がブロックされました")
        if interceptor:
            # describe_execution自体に不具合があった場合など、通知で止まらないようにする
            try:
                describe = self.describe_execution or self._default_describe_execution
                interceptor.notify.execute_start(describe(kwargs))
            except Exception as e:
                error_message = f"{tool_name}の実行中メッセージの生成に失敗しました:{e}"
                interceptor.notify.error(error_message)
                logger.exception(f"{error_message} error=%s")
        try:
            value = self.func(**kwargs)
            result = ToolResult(value=value)
        except Exception as e:
            if interceptor:
                interceptor.notify.error(f"{tool_name}の実行に失敗しました:{e}")
                error_message = (
                    str(e)
                    if self.expose_error_details
                    else "ツールの実行中にエラーが発生しました。"
                )
                logger.exception(f"{error_message} error=%s")
        return result

    def to_catalog_line(self) -> str:
        """toolの存在だけを伝える一行"""
        return f"- {self.name}: {self.summary}"

    def to_declaration(self) -> dict:
        """Function calling用の形式で出力する
        各LLMAPIに沿った形に変換するのはLLMクラスが行う
        """
        description = "\n".join(p for p in (self.summary, self.usage) if p)
        return {
            "name": self.name,
            "description": description,
            "parameters": self.get_json_schema(),
        }

    def get_json_schema(self) -> dict:
        """関数の型ヒントからJSONschemaを生成"""
        try:
            sig = inspect.signature(self.func, eval_str=True)
        except (NameError, TypeError):
            sig = inspect.signature(self.func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # *args,**kwargsはschemaプロパティとして表現不可の為無視する
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            properties[name] = self._schema_for(param.annotation)
            if name in self.param_descriptions:
                properties[name]["description"] = self.param_descriptions[name]
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": properties, "required": required}

    @classmethod
    def _schema_for(cls, annotation) -> dict:
        """型解釈をJSON Schemaへ変換"""
        origin = get_origin(annotation)

        if origin is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if not args:
                return {"type": "string"}
            return cls._schema_for(args[0])
        base = origin or annotation
        json_type = cls._TYPE_MAP.get(base, "string")
        schema = {"type": json_type}

        if json_type == "array":
            args = get_args(annotation)
            schema["items"] = cls._schema_for(args[0]) if args else {"type": "string"}
        return schema
