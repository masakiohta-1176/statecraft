"""
StateCraft — 記憶を構造として扱うマルチエージェントのフレームワーク。


エージェント同士は会話を渡さない。渡すのは構造化された依頼と、
共有された記憶への追記だけ。


    from statecraft import Agent, GeminiLLM, Network, SharedMemory, Tool


各モジュールを直接指定しても同じものが取れる。


    from statecraft.agent import Agent


ここに並べているのは、利用側が組み立てに使う名前だけ。
内部で使う補助（プロンプトの断片、差分の適用など）は各モジュールから
直接importする。全部をここへ並べると、何が使う側の道具で何が内部の
部品なのかが区別できなくなる。
"""

from .agent import (
    Agent,
    AgentResponse,
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


__all__ = [
    "Agent",
    "AgentResponse",
    "BaseLLM",
    "CheckResult",
    "ClaudeLLM",
    "GeminiLLM",
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
]
