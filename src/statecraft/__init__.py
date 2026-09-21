"""
StateCraft — 記憶を構造として扱うマルチエージェントのフレームワーク。

    from statecraft import Agent, GeminiLLM, Network, SharedMemory, Tool
    from statecraft.agent import Agent          # 直接指定でも同じ

並べているのは利用側が組み立てに使う名前だけ。内部の補助
（プロンプトの断片、差分の適用など）は各モジュールから直接importする。
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .agent import (
    Agent,
    AgentResponse,
    ExecuteEvent,
    GenerationAborted,
    GenerationConfig,
    GenerationEvent,
    MemoryDiffEvent,
    ReflexAgent,
    RunInput,
)
from .interceptor import CheckResult, Interceptor, RetryDecision
from .invokable import Invokable
from .llm import BaseLLM, ClaudeLLM, GeminiLLM, OpenAILLM, ThoughtLevel
from .memory import MemoryEntry, PrivateMemory, SharedMemory, TaskEntry, TaskStatus
from .network import Network
from .prompts import Phase
from .tools import Tool, ToolResult

try:
    # 配布物のメタデータを唯一の出典にする（pyprojectとの二重管理を避ける）。
    __version__ = _version("statecraft")
except PackageNotFoundError:
    # インストールせずにsrc/を直接importした場合。動作はするので落とさない。
    __version__ = "0.0.0+unknown"

__all__ = [
    "Agent",
    "AgentResponse",
    "BaseLLM",
    "CheckResult",
    "ClaudeLLM",
    "ExecuteEvent",
    "GeminiLLM",
    "GenerationAborted",
    "GenerationConfig",
    "GenerationEvent",
    "Interceptor",
    "Invokable",
    "MemoryDiffEvent",
    "MemoryEntry",
    "Network",
    "OpenAILLM",
    "Phase",
    "PrivateMemory",
    "ReflexAgent",
    "RetryDecision",
    "RunInput",
    "SharedMemory",
    "TaskEntry",
    "TaskStatus",
    "ThoughtLevel",
    "Tool",
    "ToolResult",
    "__version__",
]


