"""
Agent の部品の検証。LLMを呼ばずに単体で確かめられるものだけを集める。

ループ本体（respond / execute）は test_agent_loop.py を参照。
"""

import pytest
from conftest import FakeLLM

from statecraft import (
    Agent,
    GenerationConfig,
    MemoryEntry,
    Phase,
    SharedMemory,
    ThoughtLevel,
    Tool,
    ToolResult,
)
from statecraft.agent import ToolHistoryEntry, _extract_rows, _is_blank
from statecraft.llm import FunctionCall


@pytest.fixture
def agent():
    return Agent(name="librarian", summary="調べる", llm=FakeLLM(), model="m")


# ---- _extract_rows（構造化を強制できないプロバイダの応答を吸収する） ----
def test_素の配列をそのまま読む():
    assert _extract_rows('[{"field": "facts"}]') == [{"field": "facts"}]


def test_前後に説明文があっても配列を取り出す():
    text = 'こちらが差分です:\n[{"field": "facts", "id": "f-1", "text": "x"}]\n以上です。'

    assert _extract_rows(text) == [{"field": "facts", "id": "f-1", "text": "x"}]


def test_1つの配列で包まれていれば中身を採用する():
    assert _extract_rows('{"updates": [{"field": "facts"}]}') == [{"field": "facts"}]


def test_配列が複数あるオブジェクトは解釈しない():
    """どちらが差分か決められないので、推測せずNoneを返して再生成させる。"""
    assert _extract_rows('{"a": [1], "b": [2]}') is None


def test_配列が無ければNone():
    assert _extract_rows('{"field": "facts"}') is None
    assert _extract_rows("ただの文章です") is None


def test_空文字はNone():
    assert _extract_rows("") is None
    assert _extract_rows(None) is None


def test_壊れたJSONはNone():
    assert _extract_rows('[{"field": ') is None


# ---- _is_blank（必須引数が実質的に空か） ----
def test_Noneと空白だけの文字列は空扱い():
    assert _is_blank(None)
    assert _is_blank("")
    assert _is_blank("   ")


def test_ゼロや空リストは空扱いにしない():
    """真偽値で判定すると「件数0を指定する」のような正当な呼び出しが弾かれる。"""
    assert not _is_blank(0)
    assert not _is_blank(False)
    assert not _is_blank([])
    assert not _is_blank("0")


# ---- resolve_generation（指定なしのNoneが消える唯一の場所） ----
def test_上書きが無ければ既定値を使う(agent):
    agent.model = "default-model"
    agent.max_tokens = 4096

    cfg = agent.resolve_generation(Phase.ANSWER)

    assert cfg.model == "default-model"
    assert cfg.max_tokens == 4096


def test_指定のある項目だけが差し替わる(agent):
    agent.model = "default-model"
    agent.max_tokens = 4096
    agent.phase_overrides = {Phase.ANSWER: GenerationConfig(model="pro")}

    cfg = agent.resolve_generation(Phase.ANSWER)

    assert cfg.model == "pro"
    assert cfg.max_tokens == 4096  # 指定していない項目は既定のまま


def test_temperature0は既定値へ差し替わらない(agent):
    """`o.temperature or self.temperature` と書くと0が偽と判定されて消える。"""
    agent.temperature = 0.9
    agent.phase_overrides = {Phase.ANSWER: GenerationConfig(temperature=0.0)}

    cfg = agent.resolve_generation(Phase.ANSWER)

    assert cfg.temperature == 0.0


def test_thought_levelのNONEも保たれる(agent):
    """ThoughtLevel.NONEは0なので、同じ理由で消えうる。"""
    agent.thought_level = ThoughtLevel.HIGH
    agent.phase_overrides = {Phase.ANSWER: GenerationConfig(thought_level=ThoughtLevel.NONE)}

    cfg = agent.resolve_generation(Phase.ANSWER)

    assert cfg.thought_level is ThoughtLevel.NONE


def test_上書きは対象のphaseにだけ効く(agent):
    agent.model = "default-model"
    agent.phase_overrides = {Phase.ANSWER: GenerationConfig(model="pro")}

    assert agent.resolve_generation(Phase.FUNCTION_CALL).model == "default-model"


# ---- _matches_output_schema ----
def test_JSONとして読めなければ不一致(agent):
    agent.output_schema = {"type": "object"}

    assert not agent._matches_output_schema("```json\n{}\n```")


def test_オブジェクトを求めたのに配列なら不一致(agent):
    agent.output_schema = {"type": "object"}

    assert not agent._matches_output_schema("[1, 2]")
    assert agent._matches_output_schema('{"a": 1}')


def test_配列を求めた場合も同様(agent):
    agent.output_schema = {"type": "array"}

    assert agent._matches_output_schema("[1, 2]")
    assert not agent._matches_output_schema('{"a": 1}')


def test_型指定が無ければ読めるだけで一致(agent):
    agent.output_schema = {}

    assert agent._matches_output_schema('{"a": 1}')


# ---- _call_signature と重複判定 ----
def test_引数の順序が違っても同じ署名になる(agent):
    a = agent._call_signature(FunctionCall(name="f", args={"x": 1, "y": 2}))
    b = agent._call_signature(FunctionCall(name="f", args={"y": 2, "x": 1}))

    assert a == b


def test_引数が違えば別の署名になる(agent):
    a = agent._call_signature(FunctionCall(name="f", args={"x": 1}))
    b = agent._call_signature(FunctionCall(name="f", args={"x": 2}))

    assert a != b


def test_JSONにできない引数でも署名を作れる(agent):
    """bytes等が来ても落ちないこと（落ちると呼び出し全体が止まる）。"""
    signature = agent._call_signature(FunctionCall(name="f", args={"data": b"\x00"}))

    assert "f(" in signature


# ---- _step_fingerprint（停滞の検出） ----
def test_同じ実行なら同じ指紋になる(agent):
    def make():
        return [
            ToolHistoryEntry(
                name="search", kwargs={"topic": "貸出"}, result=ToolResult(error="失敗")
            )
        ]

    assert agent._step_fingerprint(make()) == agent._step_fingerprint(make())


def test_エラー内容が違えば別の指紋になる(agent):
    first = [ToolHistoryEntry(name="s", kwargs={}, result=ToolResult(error="A"))]
    second = [ToolHistoryEntry(name="s", kwargs={}, result=ToolResult(error="B"))]

    assert agent._step_fingerprint(first) != agent._step_fingerprint(second)


def test_順序が違っても同じ指紋になる(agent):
    """集合なので、同時実行の完了順で停滞判定が揺れないこと。"""
    a = ToolHistoryEntry(name="a", kwargs={}, result=ToolResult(error="x"))
    b = ToolHistoryEntry(name="b", kwargs={}, result=ToolResult(error="y"))

    assert agent._step_fingerprint([a, b]) == agent._step_fingerprint([b, a])


# ---- ToolHistoryEntry.render ----
def test_成功時はsuccessを見せない():
    """「動いた」と「求めた答えが得られた」を混同させないため。"""
    entry = ToolHistoryEntry(name="s", kwargs={"topic": "貸出"}, result=ToolResult(value="5冊"))

    rendered = entry.render()

    assert "5冊" in rendered
    assert "success" not in rendered


def test_失敗時はstatus_errorを出す():
    entry = ToolHistoryEntry(name="s", kwargs={}, result=ToolResult(error="見つからない"))

    rendered = entry.render()

    assert "status: error" in rendered
    assert "見つからない" in rendered


def test_memory記録済みなら中身を繰り返さない():
    entry = ToolHistoryEntry(
        name="s", kwargs={}, result=ToolResult(value="長い本文"), written_to_memory=True
    )

    rendered = entry.render()

    assert "長い本文" not in rendered
    assert "memoryへ記録済み" in rendered


def test_include_result_Falseでも実行した事実は残る():
    entry = ToolHistoryEntry(name="s", kwargs={}, result=ToolResult(value="長い本文"))

    rendered = entry.render(include_result=False)

    assert "s(" in rendered
    assert "長い本文" not in rendered


def test_失敗はinclude_result_Falseでも見せる():
    """隠すと、失敗したことが次の判断材料から消える。"""
    entry = ToolHistoryEntry(name="s", kwargs={}, result=ToolResult(error="見つからない"))

    assert "見つからない" in entry.render(include_result=False)


def test_output_schemaの説明が結果に添えられる():
    entry = ToolHistoryEntry(
        name="s",
        kwargs={},
        result=ToolResult(value='{"status": "PARTIAL"}'),
        output_schema={"properties": {"status": {"description": "処理の結果の区分"}}},
    )

    assert "処理の結果の区分" in entry.render()


def test_説明を持たないスキーマは何も添えない():
    entry = ToolHistoryEntry(
        name="s",
        kwargs={},
        result=ToolResult(value="x"),
        output_schema={"properties": {"status": {"type": "string"}}},
    )

    assert "返ってきた値の意味" not in entry.render()


# ---- 名前引き ----
def test_toolとsub_agentを名前で引ける(agent):
    def search_rules(topic: str) -> dict:
        return {"value": topic}

    tool = Tool(func=search_rules, summary="x")
    sub = Agent(name="sub", summary="s", llm=FakeLLM(), model="m")
    agent.tools = [tool]
    agent.sub_agents = [sub]

    assert agent.get_tool("search_rules") is tool
    assert agent.get_tool("nope") is None
    assert agent.get_sub_agent("sub") is sub
    assert agent.get_sub_agent("nope") is None


# ---- to_declaration ----
def test_declarationは配下の能力一覧を含む(agent):
    def search_rules(topic: str) -> dict:
        return {"value": topic}

    agent.tools = [Tool(func=search_rules, summary="規程を調べる")]
    d = agent.to_declaration()

    assert d["name"] == "librarian"
    assert "search_rules" in d["description"]


def test_declarationはoutput_schemaの項目名だけを見せる(agent):
    """値の意味まで載せると、委譲先の全員分が毎ステップ載って膨らむ。"""
    agent.output_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "とても長い説明文"}},
    }

    d = agent.to_declaration()

    assert "answer" in d["description"]
    assert "とても長い説明文" not in d["description"]


# ---- reset ----
def test_resetで記憶と累計が消える(agent):
    agent.private_memory.goals.append(MemoryEntry(id="g-1", text="x"))
    agent.total_input_tokens = 100
    agent.tool_history = ["なにか"]
    agent.current_run_id = "abc"

    agent.reset()

    assert agent.private_memory.goals == []
    assert agent.total_input_tokens == 0
    assert agent.tool_history == []
    assert agent.current_run_id == ""


def test_resetはshared_memoryを消さない(agent):
    """共有記憶は全員のものなので、1体のresetで消えては困る。"""
    shared = SharedMemory()
    shared.facts.append(MemoryEntry(id="f-1", text="残るべき"))
    agent.shared_memory = shared

    agent.reset()

    assert len(agent.shared_memory.facts) == 1
