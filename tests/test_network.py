"""
Network の検証。

ここは配線の場所で、検出すべき誤りのほとんどは「実行時には例外にならず、
なぜか動かない」形で現れるもの。だから構築時に落とすのが仕事になる。
"""

import pytest
from conftest import FakeLLM

from statecraft import (
    Agent,
    GenerationConfig,
    Interceptor,
    MemoryEntry,
    Network,
    Phase,
    ReflexAgent,
    Tool,
)


def make_agent(name: str, **kw) -> Agent:
    return Agent(name=name, summary=f"{name}の役割", llm=FakeLLM(), model="m", **kw)


def make_tool(name: str = "search_rules"):
    def f(topic: str) -> dict:
        return {"value": topic}

    f.__name__ = name
    return Tool(func=f, summary="調べる")


# ---- 構築時の検証 ----
def test_agentsが空なら落ちる():
    with pytest.raises(ValueError, match="1つ以上"):
        Network(agents=[])


def test_名前の重複は落ちる():
    with pytest.raises(ValueError, match="重複"):
        Network(agents=[make_agent("a"), make_agent("a")])


def test_存在しないagentへの委譲は落ちる():
    front = make_agent("front", sub_agent_names=["nobody"])

    with pytest.raises(ValueError, match="存在しないagent"):
        Network(agents=[front])


def test_自分自身への委譲は落ちる():
    front = make_agent("front", sub_agent_names=["front"])

    with pytest.raises(ValueError, match="自分自身"):
        Network(agents=[front])


def test_sub_agent_namesとsub_agentsの併用は落ちる():
    """どちらを採ってももう一方が黙って消える。"""
    librarian = make_agent("librarian")
    front = make_agent("front", sub_agent_names=["librarian"], sub_agents=[librarian])

    with pytest.raises(ValueError, match="同時に指定"):
        Network(agents=[front, librarian])


def test_配線を受けていない実体を委譲先にすると落ちる():
    """shared_memoryが別インスタンスのまま実行される。"""
    outsider = make_agent("outsider")
    front = make_agent("front", sub_agents=[outsider])

    with pytest.raises(ValueError, match="agentsに含まれていません"):
        Network(agents=[front])


# ---- phase_overrides の検証 ----
def test_llmだけ上書きしてmodelを忘れると落ちる():
    """別プロバイダのモデル名を送ることになり、APIは原因を教えてくれない。"""
    agent = make_agent(
        "a", phase_overrides={Phase.ANSWER: GenerationConfig(llm=FakeLLM())}
    )

    with pytest.raises(ValueError, match="modelも指定"):
        Network(agents=[agent])


def test_llmとmodelを揃えて上書きすれば通る():
    agent = make_agent(
        "a", phase_overrides={Phase.ANSWER: GenerationConfig(llm=FakeLLM(), model="pro")}
    )

    Network(agents=[agent])  # 落ちないこと


def test_ReflexAgentが通らないphaseの上書きは落ちる():
    agent = ReflexAgent(
        name="front",
        summary="受付",
        llm=FakeLLM(),
        model="m",
        phase_overrides={Phase.INITIAL_MEMORY: GenerationConfig(model="pro")},
    )

    with pytest.raises(ValueError, match="効きません"):
        Network(agents=[agent])


def test_対象を持つReflexAgentはANSWERの上書きも落ちる():
    """テキストを返した時点で抜けるため、ANSWERは通らない。"""
    agent = ReflexAgent(
        name="front",
        summary="受付",
        llm=FakeLLM(),
        model="m",
        tools=[make_tool()],
        phase_overrides={Phase.ANSWER: GenerationConfig(model="pro")},
    )

    with pytest.raises(ValueError, match="効きません"):
        Network(agents=[agent])


def test_対象を持たないReflexAgentならANSWERを上書きできる():
    """提示できる対象が無ければ、ANSWERが唯一通るphaseになる。"""
    agent = ReflexAgent(
        name="front",
        summary="受付",
        llm=FakeLLM(),
        model="m",
        phase_overrides={Phase.ANSWER: GenerationConfig(model="pro")},
    )

    Network(agents=[agent])  # 落ちないこと


# ---- initial_tools の検証 ----
def test_initial_toolsの重複は落ちる():
    """同じものを2度並べても、2回目は重複として拒否されるだけ。設定ミス以外に理由が無い。"""
    tool = make_tool()
    agent = make_agent("a", tools=[tool], initial_tools=[tool, tool])
    # 引数の検証を先に通すため、input_schemaを噛み合わせておく
    agent.input_schema = {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    }

    with pytest.raises(ValueError, match="重複"):
        Network(agents=[agent])


def test_initial_toolsの必須引数がinput_schemaに無いと落ちる():
    """
    噛み合っていないと引数ゼロで呼ばれ、依頼と無関係な結果が
    事実としてmemoryへ書かれる。
    """
    agent = make_agent("a", initial_tools=[make_tool()])
    # 既定のinput_schemaはmessageしか持たない（toolはtopicを要求する）

    with pytest.raises(ValueError, match="input_schemaに存在しません"):
        Network(agents=[agent])


def test_input_schemaが噛み合っていれば通る():
    agent = make_agent("a", initial_tools=[make_tool()])
    agent.input_schema = {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    }

    Network(agents=[agent])  # 落ちないこと


# ---- 配線の結果 ----
def test_shared_memoryが全員で共有される():
    a, b = make_agent("a"), make_agent("b")
    Network(agents=[a, b])

    a.shared_memory.facts.append(MemoryEntry(id="f-1", text="共有される"))

    assert b.shared_memory is a.shared_memory
    assert [e.text for e in b.shared_memory.facts] == ["共有される"]


def test_sub_agent_namesが実体へ解決される():
    librarian = make_agent("librarian")
    front = make_agent("front", sub_agent_names=["librarian"])

    Network(agents=[front, librarian])

    assert front.sub_agents == [librarian]


def test_toolは複製されてagent間で独立する():
    """あるAgentが立てたdisabledが、他のAgentにも次のリクエストにも残らないため。"""
    tool = make_tool()
    a, b = make_agent("a", tools=[tool]), make_agent("b", tools=[tool])
    Network(agents=[a, b])

    a.tools[0].disabled = True

    assert b.tools[0].disabled is False
    assert tool.disabled is False


def test_共通のinterceptorが配られる():
    interceptor = Interceptor()
    a = make_agent("a")
    Network(agents=[a], interceptor=interceptor)

    assert a.interceptor is interceptor


def test_個別に登録済みのinterceptorは尊重される():
    own = Interceptor()
    own.on.error(lambda m: None)
    a = make_agent("a", interceptor=own)

    Network(agents=[a], interceptor=Interceptor())

    assert a.interceptor is own


def test_二度構築しても配線が壊れない():
    """
    2回目の複製元が「無効化済みのコピー」にならないこと。
    Networkを作り直す運用（リクエストごとに構築）で効いてくる。
    """
    tool = make_tool()
    librarian = make_agent("librarian")
    front = make_agent("front", tools=[tool], sub_agent_names=["librarian"])

    Network(agents=[front, librarian])
    front.tools[0].disabled = True

    Network(agents=[front, librarian])

    assert front.tools[0].disabled is False
    assert front.sub_agents == [librarian]


# ---- 循環の検出 ----
def test_相互に委譲し合う構成は落ちる():
    a = make_agent("a", sub_agent_names=["b"])
    b = make_agent("b", sub_agent_names=["a"])

    with pytest.raises(ValueError):
        Network(agents=[a, b])


def test_三者間の循環も検出する():
    a = make_agent("a", sub_agent_names=["b"])
    b = make_agent("b", sub_agent_names=["c"])
    c = make_agent("c", sub_agent_names=["a"])

    with pytest.raises(ValueError):
        Network(agents=[a, b, c])


def test_木構造なら通る():
    leaf = make_agent("leaf")
    mid_a = make_agent("mid_a", sub_agent_names=["leaf"])
    mid_b = make_agent("mid_b", sub_agent_names=["leaf"])
    root = make_agent("root", sub_agent_names=["mid_a", "mid_b"])

    Network(agents=[root, mid_a, mid_b, leaf])  # 落ちないこと
