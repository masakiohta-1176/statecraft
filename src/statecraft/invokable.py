"""
「LLMから呼び出せるもの」が満たすべき形の定義。型だけで、実行時の処理は無い。

    tool.execute(kwargs={...}, interceptor=...)   → ToolResult
    agent.execute(kwargs={...}, interceptor=...)  → ToolResult

同じ形で呼べるため、委譲先をtoolへ差し替えても呼び出し側は変わらない。
"""

from typing import Protocol, runtime_checkable

from .interceptor import Interceptor
from .tools import ToolResult


@runtime_checkable
class Invokable(Protocol):
    """
    「LLMから呼び出せるもの」の型。ToolとAgentが独立に満たす。

    Protocolなのは、class Agent(Tool) にすると「AgentはToolの一種」という
    成り立たない関係を宣言することになるため（双方importしない）。
    """

    # 読み取り専用として宣言するため@property。属性で書くと書き換え可能である
    # ことを要求し、nameを@propertyで実装しているTool側が形を満たせない。
    @property
    def name(self) -> str:
        """LLMへ提示する名前。"""
        ...

    # 候補として提示する際に確認する。セッション中に書き換えるので通常の属性。
    disabled: bool

    def to_declaration(self) -> dict:
        """LLM APIへ渡す宣言情報（name / description / parameters）。"""
        ...

    def to_catalog_line(self) -> str:
        """存在だけを伝える1行。"""
        ...

    def execute(self, *, kwargs: dict, interceptor: Interceptor | None = None) -> ToolResult:
        """実行して結果を返す。例外は投げず、失敗もToolResultとして返す。"""
        ...


