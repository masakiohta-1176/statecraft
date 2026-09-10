"""
「LLMから呼び出せるもの」が満たすべき形の定義。


ToolとAgentは、まったく別のものでありながら、呼び出し側から見ると
同じように扱える。


    tool.execute(kwargs={...}, interceptor=...)   → ToolResult
    agent.execute(kwargs={...}, interceptor=...)  → ToolResult


引数も戻り値も同じなので、Agentは「呼べば結果が返るもの」として、
ツールと並べて扱える。あるエージェントの委譲先をツールへ、
あるいはその逆へ差し替えても、呼び出し側のコードは変わらない。


このモジュールはその「同じ形」を型として書き留めたもので、
実行時に何かをするコードは入っていない。
"""


from typing import Protocol, runtime_checkable


from .interceptor import Interceptor
from .tools import ToolResult




@runtime_checkable
class Invokable(Protocol):
    """
    「LLMから呼び出せるもの」の型。ToolとAgentが独立に満たす。


    継承ではなくProtocol（構造的部分型）を使う。
    class Agent(Tool) のような継承関係を作ると「AgentはToolの一種である」という
    実際には成り立たない関係を宣言してしまうため、
    「この形を満たしていればよい」という条件だけを型として表現する。


    ToolもAgentもこのファイルをimportしない。実装側が何も知らなくても
    型チェッカーが構造だけを見て判定するのがProtocolの利点。
    """


    # 呼び出し側（Agent）が名前で参照するため、名前も形の一部になる。
    #
    # `name: str` と書かず@propertyにしているのは、読み取り専用として
    # 宣言するため。属性として書くと「書き換えもできる」ことを要求してしまい、
    # Tool側がnameを@propertyで実装している（func.__name__から導出する）ため
    # 形が合わなくなる。読むだけなので、読み取り専用で十分。
    @property
    def name(self) -> str:
        """LLMへ提示する名前。"""
        ...


    # 一覧や候補として提示する時に、無効化されていないかを確認するため必要。
    # こちらは読むだけでなく書き換えもする（LLMが「今回は使わない」と宣言した
    # 対象を無効化し、次の起動時に戻す）ので、通常の属性として宣言する。
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





