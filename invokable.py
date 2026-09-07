from typing import Protocol, runtime_checkable

from interceptor import Interceptor
from tools import ToolResult


@runtime_checkable
class Invokable(Protocol):
    """ざっくり「LLMから呼び出せるもの(function)」の型、ToolとAgentがそれ

    継承ではなくProtocolを使用し、単純な型定義を行うのみ。
    """

    def to_declaration(self) -> dict: ...
    def to_catalog_line(self) -> dict: ...
    def execute(
        self, *, kwargs: dict, interceptor: Interceptor | None
    ) -> ToolResult: ...
