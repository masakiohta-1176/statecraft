"""
Agentの本体。ReActループ、プロンプトの組み立て、他Agentへの委譲を担う。


【AgentとReflexAgentの違い】
    Agent        記憶を持つ。調べる前に計画を立て、結果を評価して記憶へ書く。
                 複数の手順を踏む役割（統括、専門家）に使う。
    ReflexAgent  記憶を持たない（共有記憶は読む）。実行して答えるだけ。
                 会話の窓口や単発の判断に使う。生成回数が少なく速い。


差異はフラグではなく継承で表現している。ReflexAgentはAgentを継承し、
記憶に関わるhook（_before_loop / _after_tools など）を空にするだけ。
ループ本体（_run_react）は両者で共通の1つしかない。


【1回の応答で何が起きるか】
    respond() または execute()
      → _begin_run()      前回の実行の痕跡を捨てる
      → _before_loop()    初期ツールを実行し、最初の記憶を作る（Agentのみ）
      → ループ開始
          _select_invokables()  次に呼ぶ対象を決める
          ・対象がある → 生成してツールを呼ぶ → 結果を記憶へ（_after_tools）
          ・対象がない → ループを抜けて回答へ
      → _final_answer()   回答を生成する


【外から呼ばれる入口は2つ】
    respond()  人間からの入口。会話履歴や添付を受け取る
    execute()  他のAgentから委譲された時の入口。Toolと同じ形で呼べる


execute()がToolと同じ形（同じ引数・同じ戻り値）をしているため、
呼び出し側はツールとエージェントを区別せずに扱える（invokable.py参照）。
"""

import copy
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Optional


from .interceptor import Interceptor
from .invokable import Invokable
from .llm import BaseLLM, FunctionCall, LLMResponse, ThoughtLevel
from .memory import (
    DIFF_EXAMPLE,
    DISABLE_FIELD,
    ROUTING_GUIDE,
    DiffError,
    MemoryEntry,
    PrivateMemory,
    SharedMemory,
    TaskStatus,
    apply_diff,
    build_diff_schema,
)
from .prompts import (
    PHASE_SPECS,
    POURING,
    STALL_NOTICE,
    Phase,
    PhaseSpec,
    join_sections,
    section,
)
from .tools import Tool, ToolResult
from .utils import format_message, with_timing


def _extract_rows(text: str) -> list | None:
    """
    LLMの応答からmemory差分の配列を取り出す。


    Geminiはresponse_schemaで配列を構造的に強制できるが、
    OpenAIとClaudeには同等の仕組みが無く、前後に文章が付いたり
    オブジェクトで包まれたりする。パース側で吸収する。


    取れなければNoneを返す（例外にしない）。呼び出し側はリトライで直させる。
    """
    if not text:
        return None

    def as_rows(parsed):
        if isinstance(parsed, list):
            return parsed
        # {"updates": [...]} のように1つの配列で包まれていた場合は中身を採用する
        if isinstance(parsed, dict):
            arrays = [v for v in parsed.values() if isinstance(v, list)]
            if len(arrays) == 1:
                return arrays[0]
        return None

    try:
        rows = as_rows(json.loads(text))
        if rows is not None:
            return rows
    except json.JSONDecodeError:
        pass

    # 前後に説明文が付いている場合に備えて、最初の[から最後の]までを試す
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            return as_rows(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class RunInput:
    """
    1回の起動で受け取った入力をまとめたもの。


    ReActループの各メソッド（プロンプト構築、初期ツール実行、記憶更新）が
    それぞれこの内容を必要とする。引数として延々と引き回すのを避けるため、
    実行中のAgentインスタンスが `current_input` として保持し、各メソッドは
    そこから読む。


    1回の起動＝respond()またはexecute()の1回の呼び出し。
    次の起動時には_begin_run()で新しいRunInputへ差し替わる。
    """

    # プロンプトの【依頼内容】へ載せる本文。
    message: str

    # 呼び出し元から受け取った構造化引数（このAgentのinput_schemaに従う形）。
    # messageが「人間が読む文章」なのに対し、こちらは項目ごとに分かれた辞書。
    # 初期ツール（initial_tool_name）へ渡す引数は、ここから名前が一致する
    # ものだけを抜き出して使う。
    kwargs: dict = field(default_factory=dict)
    # 過去のチャット履歴。[{"role": "user"|"assistant", "content": str}, ...]
    # 利用者と直接会話するAgent（respond経由）だけが受け取る。
    # 委譲で呼ばれたAgent（execute経由）には渡らない——委譲では
    # 「何をしてほしいか」だけが渡り、会話の経緯は渡らないため。
    history: list[dict] = field(default_factory=list)
    # 添付ファイル。[{"data": bytes, "mime_type": str}, ...]
    # 画像やPDFをLLMへ渡す場合に使う。
    blobs: list[dict] = field(default_factory=list)


@dataclass
class AgentResponse:
    """respond()の戻り値。"""

    text: str
    steps: int = 0  # 実際に回ったループ回数

    # このAgent自身が消費したトークン。委譲先が消費した分は含まない。
    # どのAgentが重いのかを個別に測るためであり、意図的に分離している。
    # セッション全体の合計はNetwork.total_tokens()、
    # Agentごとの内訳はNetwork.token_usage()で取得する。
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MemoryDiffEvent:
    """
    記憶の差分を1回の生成につき1つ、構造のまま観測側へ渡す。


    memory_updatedの文字列は人間が読むためのもので、
    「facts:f1 / goals:g1」までしか分からず、textも失敗も落ちている。
    「この入力に対して、この記憶の作り方で合っているか」を検証するには、
    材料・出力・結果の3つが同時に要る。それを1つのイベントに揃える。


    成功した時だけでなく、失敗した時も必ず発火する。
    検証で見たいのはむしろ失敗した回であり、
    そこだけerrorイベントの文字列に落ちるのでは追えない。
    """

    agent: str  # どのAgentが出したか
    phase: Phase | None  # どのphaseの生成か。Noneはtoolの直接書き込み
    attempt: int  # 何回目の生成か（1始まり。再生成の経過が見える）
    rows: list  # 差分そのもの。textを含む
    errors: list  # 適用に失敗した行。空なら全行が反映された
    tool_results: str = ""  # この差分を作らせた材料
    raw_text: str = ""  # 配列として解釈できなかった場合の生出力

    # write_to_memory=True のtoolが直接書き込んだ場合、そのtool名。
    # LLMが作った差分と、toolが返した値をそのまま入れた差分は、
    # 検証の意味が違う（前者は生成の妥当性、後者はtool実装の妥当性）。
    # 同じイベントで流すが、どちらなのかは区別できる必要がある。
    source_tool: str = ""

    @property
    def applied(self) -> bool:
        """全行が反映されたか。errorsの有無から導出する（ToolResult.successと同じ考え方）。"""
        return not self.errors


@dataclass
class GenerationEvent:
    """
    生成1回分の実測値。時間とトークンを構造のまま観測側へ渡す。


    セッション全体の合計はNetworkが持っているが、それだけでは
    「どのAgentのどのphaseが重いのか」が分からない。
    生成回数が多いのか、1回が重いのかも切り分けられない。
    """

    agent: str  # どのAgentの生成か
    phase: Phase  # どのphaseの生成か
    model: str  # 実際に使われたモデル名（phase_overridesの結果が見える）
    elapsed: float  # 生成にかかった秒数
    input_tokens: int
    output_tokens: int


@dataclass
class GenerationConfig:
    """
    phaseごとの生成設定の「上書き指示」。phase_overridesに書く側の型。


    全項目をOptionalにしてあり、Noneは「指定なし＝Agentの既定値を使う」を意味する。
    書かなかった項目がNoneとして残ることで「触らない」が表現できる。
    初期memory構築だけ安価なモデルを使う、といった使い分けをここで書く。


    modelとllmが別項目なのは、接続とモデル指定が別物だから。
    同じ接続（GeminiLLM）のまま、phaseごとにモデルだけ差し替えられる。
        phase_overrides={Phase.ANSWER: GenerationConfig(model="gemini-3.5-pro")}


    実際に生成へ渡す確定値はResolvedGenerationが持つ。
    """

    model: str | None = None
    llm: BaseLLM | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thought_level: ThoughtLevel | None = None


@dataclass
class ResolvedGeneration:
    """
    そのphaseで実際に使う生成設定。resolve_generation()の戻り値。


    GenerationConfigと項目は同じだが、意味が違うので別の型にしてある。
    こちらは上書きと既定値を突き合わせた「後」の姿なので、
    指定なしを表すNoneは残っていない。全項目に既定値を持たせず必須にすることで、
    「解決済みである」ことを型として宣言している。


    同じ型を使い回すと、解決後もmodelがstr | Noneのままになり、
    生成へ渡す時点でNoneの可能性を考えなければならなくなる。
    実際には埋まっているのだが、それはresolve_generation()の中身を
    読まないと分からない。型を分ければ読まなくても分かる。


    thought_levelだけは解決後もNoneを許す。ここでのNoneは「指定なし」ではなく
    「思考の指定を送らず、モデルの動的思考に任せる」という有効な値だから。
    """

    model: str
    llm: BaseLLM
    temperature: float
    max_tokens: int
    thought_level: ThoughtLevel | None


@dataclass
class ToolHistoryEntry:
    """
    ReActループ内で1回実行したtool/sub_agentの記録。


    結果本文は破壊的に消さず、常に完全な形で保持する。
    「プロンプトに載せる時だけ結果を隠す」のはrender()側の責務であり、
    データ自体を書き換えて実現しない（memory.pyのrender(include_how_to=...)と同じ考え方）。


    sub_agent呼び出しの結果も、回答文をvalueに入れたToolResultとして
    同じ形で記録する（履歴の読み手にとってtoolとagentを区別する必要がないため）。
    """

    name: str  # 実行したtool名またはagent名
    kwargs: dict  # 渡した引数
    result: ToolResult  # 実行結果（成功/失敗と戻り値）

    # 呼ばれた側が持つ「この結果をどう解釈すべきか」の指示。
    # 呼び出し時に実体から取得して保持する。結果を隠す時は一緒に隠す
    # （memoryへ要約済みなら、解釈指示はもう役目を終えている）。
    evaluation: str = ""

    # 呼ばれた側のoutput_schema。値そのものではなく、値の意味を添えるために持つ。
    #
    # output_schemaはto_declaration()経由で呼び出し側へ渡るが、
    # それが載るのはtoolをAPIへ渡すphase（FUNCTION_CALL）だけである。
    # 結果を評価するphaseと回答するphaseはallows_tools=Falseなので、
    # 軽い一覧（to_catalog_line）しか載らない。
    # そのため「status=PARTIAL が返ってきた」ことは見えても、
    # PARTIALが何を意味するのかが見えない状態になっていた。
    #
    # 意味を人格定義へ書き写すと二重管理になるため、
    # 結果と一緒にスキーマの説明を添える形にしている。
    output_schema: dict | None = None

    # write_to_memoryによって、結果が既にmemoryへ記録済みかどうか。
    # 記録済みなら履歴側では中身を繰り返さない（同じ内容を二重に載せない）。
    written_to_memory: bool = False

    def render(self, *, include_result: bool = True) -> str:
        """
        1件分をプロンプト用のテキストにする。


        成功時は戻り値だけを載せ、成功したという事実自体は書かない。
        success:true のような表現を見せると、LLMが「ツールが動いた」ことと
        「求めていた答えが得られた」ことを混同するため、構造的に防ぐ。
        異常時だけ status: error を明示する。


        include_result: Falseなら戻り値を載せず、「実行した」という事実だけを残す。
            Statefulなagentは結果をmemoryへ要約済みなので、生の結果を再度
            載せるとトークンを二重に消費し、要約前の情報に引きずられる。
            ただしエラーは要約の有無に関わらず常に見せる。次の行動判断に
            直接必要で、隠すと同じ失敗を繰り返すため。
        """
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        head = f"  - {self.name}({args})"

        if not self.result.success:
            return f"{head}\n    status: error\n    {self.result.error}"
        # 既にmemoryへ記録済みなら、記憶を持つかどうかに関わらず中身を繰り返さない。
        if self.written_to_memory:
            return f"{head}\n    ※結果はmemoryへ記録済み"
        if not include_result:
            return f"{head}\n    ※結果はmemoryへ要約済み"

        body = f"{head}\n    {self.result.value}"
        # 値の意味を先に添える。何を意味する値なのかが分からないまま
        # 「どう解釈するか」を読んでも噛み合わない。
        if schema_note := self._render_schema_note():
            body += f"\n    【返ってきた値の意味】\n{schema_note}"
        if self.evaluation:
            body += f"\n    【この結果の解釈】{self.evaluation}"
        return body

    def _render_schema_note(self) -> str:
        """
        output_schemaの各項目の説明を、結果に添える形にする。


        説明を持たないスキーマ（型だけ指定したもの）は何も返さない。
        載せる意味があるのは、値の意味がdescriptionに書かれている場合だけ。
        """
        props = (self.output_schema or {}).get("properties", {})
        lines = [
            f"      {name}: {spec['description']}"
            for name, spec in props.items()
            if isinstance(spec, dict) and spec.get("description")
        ]
        return "\n".join(lines)


# ほとんどのAgentは、委譲される依頼を「自然言語のメッセージ1つ」として受け取る
# のが共通の妥当なパターンなので、これをinput_schemaのデフォルトにする。
# 複数の構造化された引数が必要なAgentだけ、明示的に上書きすればよい。
DEFAULT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
}


@dataclass
class Agent:
    # tool実行結果をプロンプトに載せるかどうか。
    # このクラスは結果をmemoryへ要約するため、生の結果は履歴から隠す。
    # 記憶を持たないReflexAgentはTrueで上書きする。
    # ClassVarにしているのでdataclassのフィールドにはならず、インスタンス生成時に
    # 指定する対象にもならない（クラスの性質そのものなので、設定値ではない）。
    _show_tool_results: ClassVar[bool] = False

    # ==========================================
    # 必須項目（デフォルト無し）
    # ==========================================
    name: str

    # 委譲先候補としてLLMに提示する時の概要。
    # 「何に詳しいか」ではなく「何を返すか」を書く
    # （詳しい領域を書くと、質問を投げる先として選ばれてしまう）。
    summary: str

    # Agentごとに別のプロバイダ（Gemini / Claude / OpenAI）を使うのが前提なので、
    # Network側で既定値を配らず、必ずAgent自身が持つ。
    #
    # llmは「どこへ繋ぐか」、modelは「どのモデルを使うか」。
    # 同じ接続を複数のAgentで使い回し、モデルだけAgentごとに変えられる。
    #   llm = GeminiLLM(api_key=..., thinking_style="level")
    #   front     = ReflexAgent(llm=llm, model="gemini-3.5-flash-lite", ...)
    #   librarian = Agent(      llm=llm, model="gemini-3.5-pro",        ...)
    llm: BaseLLM
    model: str

    # ==========================================
    # Networkが配線する項目
    # ==========================================
    # Networkが生成した1つの実体を全Agentへ注入する。
    # 既定値を持たせているのは単体でも動かせるようにするためで、
    # 複数Agentで使う場合はNetworkが必ず上書きする。
    shared_memory: SharedMemory = field(default_factory=SharedMemory)
    # 委譲先の宣言。相互参照があるためインスタンスでは書けず、名前で宣言して
    # Networkがsub_agentsへ解決する。
    sub_agent_names: list[str] = field(default_factory=list)

    # ==========================================
    # 他のエージェントに見せるもの
    # （このAgentを部下に持つ側のプロンプトに載る情報）
    # ==========================================
    # 実際に委譲すると決まった時だけ渡す詳細。
    # 引数の意味はinput_schemaのdescriptionが説明するので、
    # ここに残すのはスキーマでは表現できない使い方の制約。
    usage: str = ""

    # 委譲時に受け取れる引数の形（JSON Schema）。
    #   {"type": "object", "properties": {...}, "required": [...]}
    #
    # dictは可変オブジェクトなので、直接代入せずdefault_factoryで
    # インスタンスごとに新しいコピーを作る（全インスタンス共有を防ぐ）。
    # dict(...)は浅いコピーで、ネストした"properties"辞書までは複製されず
    # 共有されたままになってしまうため、copy.deepcopy()で完全に複製する。
    input_schema: dict = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_INPUT_SCHEMA)
    )

    # 最終回答に強制する形式。Noneなら自然言語で回答する。
    output_schema: dict | None = None

    # Trueの間は委譲候補として提示しない。
    disabled: bool = False

    # このAgentの回答を受け取った側が、それをどう解釈すべきかの指示。
    evaluation: str = ""

    # ==========================================
    # このAgent自身の振る舞い
    # ==========================================
    # 人格定義。llm.generate(system_instruction=...)へそのまま渡す。
    system_instruction: str = ""

    # 知識として常時プロンプトに載せる本文。
    knowledge: str = ""

    # このAgentが使えるツール。名前引き用の辞書は持たせず、リストで受け取る。
    tools: list[Tool] = field(default_factory=list)

    # sub_agent_namesからNetworkが解決して入れる。直接指定してもよい（単体実行時）。
    sub_agents: list["Agent"] = field(default_factory=list)

    private_memory: PrivateMemory = field(default_factory=PrivateMemory)

    # ReActループの上限回数。Noneは禁止（無限ループを防ぐため必ず数値を持つ）。
    max_steps: int = 10

    # 記憶の差分が不正だった時、再生成を試みる回数。
    max_memory_retries: int = 3

    # 同じ実行が何ステップ続いたら停滞とみなして打ち切るか。
    # 2なら「1度立て直しを促し、それでも変わらなければ止める」になる。
    max_stalled_steps: int = 2
    temperature: float = 1.0
    max_tokens: int = 8192
    # どれだけ考えさせるか。Noneならパラメータ自体を送らず、モデルの動的思考に任せる
    # （APIも必須ではないため、フレームワーク側でも必須にしない）。
    # 明示的に思考をOFFにしたい場合だけThoughtLevel.NONEを指定する。
    thought_level: ThoughtLevel | None = None
    # phaseごとの生成設定の上書き。指定しなかったphaseは上の既定値を使う。
    # フィールドをinitial_model/answer_model…と並べるとphaseが増える度に増え、
    # Phase Enumとの二重管理になるため、Phaseをキーにした辞書で持つ。
    #   phase_overrides={Phase.INITIAL_MEMORY: GenerationConfig(llm=flash_lite)}
    phase_overrides: dict[Phase, GenerationConfig] = field(default_factory=dict)
    # ループ開始前に、システムが機械的に1回だけ実行するツールの名前。
    # 「呼ぶかどうか」も「何を呼ぶか」も判断させないため、生成が1回分減る。
    # 引数はinput_schemaの同名プロパティから自動で渡される。
    initial_tool_name: str | None = None

    # 委譲された時に観測側（利用者の画面など）へ流す文言。
    # 引数は {name} の形で差し込めるが、委譲の引数は依頼文そのもので
    # 長くなりうるため、載せないのが普通。
    #   execution_message="司書が調べています..."
    # Tool.execution_messageと同じ扱い。Invokableとして同一に扱う以上、
    # 文言の指定方法を分ける理由がない。
    execution_message: str = ""
    # 条件で出し分けたい場合だけ関数を渡す。execution_messageより優先される。
    describe_execution: Callable[[dict], str] | None = None
    # コールバック未登録なら何もしないだけなので、Optionalにせず常に実体を持たせる。
    # こうしておくと、呼び出し側で毎回 if self.interceptor: を書く必要がなくなる。
    interceptor: Interceptor = field(default_factory=Interceptor)

    # ==========================================
    # 実行中に変化する状態
    # ==========================================
    tool_history: list[ToolHistoryEntry] = field(default_factory=list)
    # respond() / execute() が受け取った入力。ループ中の各メソッドがここから読む。
    current_input: RunInput | None = None
    # Networkが最初に配線した時点のTool一覧。2回目以降の配線で
    # ミューテート済みのコピーを複製元にしないために保持する。
    _source_tools: list | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # 直前のステップで成功した呼び出しの署名。同一呼び出しの再実行を弾くために使う。
    last_successful_calls: set = field(default_factory=set)
    # 直近の_select_invokablesで解決できなかった (名前, 理由) の一覧。
    _unresolved_targets: list = field(default_factory=list)
    # このAgentがセッション中に無効化した対象の名前。起動ごとに戻す。
    _runtime_disabled: set = field(default_factory=set)
    # 直前のステップの指紋と、同じ指紋が続いた回数。停滞の検出に使う。
    _last_fingerprint: frozenset | None = None
    _stalled_count: int = 0

    def get_tool(self, name: str) -> Tool | None:
        """名前からToolを引く。辞書はここで都度、実体（func.__name__）から導出する。"""
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def get_sub_agent(self, name: str) -> Optional["Agent"]:
        """名前からsub_agentを引く。"""
        for a in self.sub_agents:
            if a.name == name:
                return a
        return None

    def render_tool_history(self) -> str:
        """
        実行履歴をプロンプト用のテキストにする。
        結果本文を載せるかどうかはクラスの性質（_show_tool_results）で決まるため、
        呼び出し側が毎回判断する必要はない。
        """
        if not self.tool_history:
            return ""
        return "\n".join(
            e.render(include_result=self._show_tool_results) for e in self.tool_history
        )

    def to_catalog_line(self) -> str:
        """Tool.to_catalog_line()と同じ形。委譲先候補として存在を伝える1行。"""
        return f"- {self.name}: {self.summary}"

    def resolve_generation(self, phase: Phase) -> ResolvedGeneration:
        """
        そのphaseで実際に使う生成設定を確定させる。
        phase_overridesに指定があればその項目だけを差し替え、無い項目は既定値を使う。


        受け取るのは上書き指示（GenerationConfig）、返すのは確定値
        （ResolvedGeneration）で、型が変わる。ここが「指定なしのNone」が
        消える唯一の場所であり、この先はNoneを気にせず値として扱える。


        `o.temperature or self.temperature` と書くと temperature=0 が
        偽と判定されて既定値に差し替わってしまうため、必ず is None で判定する。
        （0や空文字を有効な値として扱いたい場合に共通する注意点）
        """
        o = self.phase_overrides.get(phase) or GenerationConfig()
        return ResolvedGeneration(
            model=o.model if o.model is not None else self.model,
            llm=o.llm if o.llm is not None else self.llm,
            temperature=o.temperature
            if o.temperature is not None
            else self.temperature,
            max_tokens=o.max_tokens if o.max_tokens is not None else self.max_tokens,
            thought_level=o.thought_level
            if o.thought_level is not None
            else self.thought_level,
        )

    def to_declaration(self) -> dict:
        """
        他のエージェント（このAgentを部下に持つ側）にLLM経由で見せるための、
        Tool.to_declaration()と同じ形の宣言情報。


        descriptionは summary + usage + 配下の能力一覧 で構成する。
        委譲先を選ぶ側は「そのAgentが何を返せるか」を知る必要があり、
        それは配下にどんなtool/agentがあるかで決まる。
        1段だけ見せることで、成果物の型が合う相手かを判断できるようにする。
        深く辿らせない（孫以降は見せない）のは、階層が深くなるほど
        トークンが増える一方で、選定の判断材料としての価値が下がるため。
        """
        parts = [self.summary]
        if self.usage:
            parts.append(self.usage)

        catalog = self._render_capability_catalog()
        if catalog:
            parts.append("このエージェントが使用できるtool / agent:\n" + catalog)

        # 返ってくる項目の名前だけを見せる。各項目の意味は載せない。
        #
        # ここは「どこへ委譲するか」を選ぶ段階で読まれる。
        # その判断に必要なのは「何が返るか」の形であって、
        # それぞれの値が何を意味するかではない。
        # 値の意味が必要になるのは結果を受け取った後で、
        # その時はToolHistoryEntryが結果と一緒に説明を添える。
        #
        # スキーマ全体を文字列化して載せると、descriptionまで丸ごと入る。
        # output_schemaの説明は長くなりやすく、委譲先が複数あれば全員分が
        # 毎ステップ載ることになる。
        if props := (self.output_schema or {}).get("properties"):
            parts.append("このエージェントは次の項目で回答を返す: " + ", ".join(props))

        return {
            "name": self.name,
            "description": "\n".join(parts),
            "parameters": self.input_schema,
        }

    # ==========================================
    # 起動の入り口
    # ==========================================
    def _begin_run(self, run_input: RunInput) -> None:
        """
        起動ごとに、前回の実行の痕跡を捨てる。


        Networkは1リクエスト1インスタンスだが、同一セッション内で同じAgentが
        複数回呼ばれることはある（親が2回委譲する等）。その際に前回の状態が
        残っていると、今回の依頼とは無関係な処理が走る。


        tasksを捨てるのは必須。status=nextのまま残ったタスクを
        _select_invokablesが拾い、前回の依頼のためのtool実行が発生する。
        goalsは「これまでに何を目指してきたか」として意味を持つため残す。
        shared_memoryも当然残す（セッション全体の記憶であり、起動単位ではない）。
        """
        self.current_input = run_input
        self.tool_history = []
        # 前回の依頼にもとづく無効化を持ち越さない。
        # 無効化は依頼内容から導かれる判断であって、恒久的な設定ではない。
        self._restore_disabled()
        # 依頼が変われば、同じ呼び出しでも正当な実行になりうる。
        self.last_successful_calls = set()
        self.private_memory.tasks = []
        self._last_fingerprint = None
        self._stalled_count = 0

    def respond(
        self,
        *,
        message: str,
        history: list[dict] | None = None,
        blobs: list[dict] | None = None,
    ) -> AgentResponse:
        """
        人間からの入り口。原文・チャット履歴・添付を受け取る。


        ここでshared_memoryへ書き込みは行わない。
        ユーザー原文を共有記憶へ持ち込まないことが、プロンプトインジェクションに
        対する境界になっている。原文はこのAgentのプロンプト内だけで消費され、
        構造化された依頼だけがrequestsを通じて後続へ渡る。
        """
        self._begin_run(
            RunInput(
                message=message,
                kwargs={"message": message},
                history=list(history or []),
                blobs=list(blobs or []),
            )
        )
        # トークンは累計を保持したまま、今回分だけを差分で返す。
        # 累計はNetwork.total_tokens()がセッション全体の集計に使う。
        before_in, before_out = self.total_input_tokens, self.total_output_tokens
        self.interceptor.notify.respond_start(f"{self.name}が応答を開始しました。")

        text, steps = self._run_react()

        self.interceptor.notify.respond_end(f"{self.name}が応答を完了しました。")
        return AgentResponse(
            text=text,
            steps=steps,
            input_tokens=self.total_input_tokens - before_in,
            output_tokens=self.total_output_tokens - before_out,
        )

    def _default_describe_execution(self, kwargs: dict) -> str:
        """
        execution_messageも describe_execution も無い場合の既定文言。


        Toolと違い引数の中身は載せない。委譲の引数は依頼文そのもの
        （長文になりうる）で、toolの引数のように短い識別子とは限らないため。
        """
        if self.execution_message:
            return format_message(self.execution_message, kwargs)
        return f"{self.name}へ委譲しました。"

    @with_timing
    def execute(
        self, *, kwargs: dict, interceptor: Interceptor | None = None
    ) -> ToolResult:
        """
        他のエージェントからの入り口。Invokableとしての実体であり、
        Tool.execute()と同じ形（同じ引数・同じ戻り値）で呼び出せる。


        チャット履歴は受け取らない。委譲では「何をしてほしいか」だけが渡り、
        会話の経緯は渡らない。
        """
        if self.disabled:
            # Tool.executeと同じく、呼ばれなかった理由を観測側へ伝える。
            if interceptor:
                interceptor.notify.execute_blocked(f"{self.name}は無効化されています。")
            return ToolResult(error="このエージェントは現在無効化されています。")

        if interceptor:
            verdict = interceptor.check.before_execute(name=self.name, kwargs=kwargs)
            if not verdict.allowed:
                # Tool.executeと同じ扱い。理由があれば依頼元へそのまま返す。
                detail = f" 理由: {verdict.reason}" if verdict.reason else ""
                interceptor.notify.execute_blocked(
                    f"{self.name}への委譲がブロックされました。{detail}"
                )
                return ToolResult(error=verdict.reason or "委譲がブロックされました。")

        # 依頼本文。input_schemaが既定（messageのみ）ならその中身を、
        # 構造化スキーマならJSONのまま見せる。
        if set(kwargs) == {"message"}:
            message = str(kwargs["message"])
        else:
            # 直列化できない値が混ざっても落とさない。default=strで文字列化する
            # （_call_signatureと同じ扱い）。ここで例外を出すとInvokableの
            # 「例外を外へ出さない」契約が破れる。
            message = json.dumps(kwargs, ensure_ascii=False, indent=2, default=str)

        self._begin_run(RunInput(message=message, kwargs=dict(kwargs)))

        # 最初に呼ばれたエージェント（オーケストレーター）が受け取った依頼だけが
        # セッション全体の方向を定義する。以降の委譲では上書きしない。
        # Frontはユーザー原文をここへ書かず、input_schemaに沿って構造化した
        # 依頼を委譲時に渡す。その引数がここに入ることで、原文が共有記憶へ
        # 持ち込まれないまま方向だけが共有される。
        if self.shared_memory.requests is None:
            self.shared_memory.requests = MemoryEntry(id="request_1", text=message)

        # toolと同じイベントで通知する。Invokableとして同一に扱う以上、
        # 観測側が「今何が動いているか」を知るのに区別は要らない。
        #
        # 文言の組み立ては失敗しうる（describe_executionの不具合、引数の__repr__が
        # 例外を投げる等）。通知の失敗で委譲そのものを止めないよう保護する。
        # Tool.executeと同じ扱い。
        try:
            describe = self.describe_execution or self._default_describe_execution
            self.interceptor.notify.execute_start(describe(kwargs))
        except Exception as e:
            self.interceptor.notify.error(
                f"{self.name}の実行中メッセージの生成に失敗しました: {e}"
            )

        try:
            text, _ = self._run_react()
        except Exception as e:
            # Invokableの契約として例外を外へ出さない。
            # ただし握りつぶすと原因が追えなくなるため、観測者へは通知する。
            self.interceptor.notify.error(f"{self.name}の実行が失敗しました: {e}")
            self.interceptor.notify.execute_end(
                f"{self.name}の委譲がエラーで終了しました。"
            )
            return ToolResult(error=f"{self.name}の実行中にエラーが発生しました: {e}")

        # agent_answersはシステム所有。LLMの差分では書けないため直接記録する。
        self.shared_memory.agent_answers.append(
            MemoryEntry(
                id=f"{self.name}-answer-{len(self.shared_memory.agent_answers) + 1}",
                text=text,
            )
        )
        self.interceptor.notify.execute_end(f"{self.name}の委譲が完了しました。")

        return ToolResult(value=text)

    # ==========================================
    # ReActループ
    # ==========================================
    def _run_react(self) -> tuple[str, int]:
        """
        ReActループの本体。「実行 → 評価 → 次を決める」を繰り返す。


        ループの骨格はこのメソッドだけに書かれている。記憶を持つAgentと
        持たないReflexAgentの違いは、途中で呼ばれるhookの中身だけで表現される
        （ReflexAgentは記憶に関わるhookを空にしている）。
        そのため、このメソッド自体には両者を区別する分岐が1つも無い。


        1周が1ステップで、最大max_steps回まで回る。
        ループを抜ける条件は4つ。


            ・実行すべき対象が無くなった      → ANSWER（正常な終わり方）
            ・同じ実行を繰り返して進まない    → STALLED_ANSWER
            ・max_stepsに達した               → CUTOFF_ANSWER
            ・ReflexAgentがテキストで答えた   → その場で返す


        どの終わり方でも必ず回答を返す。途中で例外を投げて終わることはない。


        戻り値: (最終回答のテキスト, 実際に回ったステップ数)
        """
        # 記憶を持つAgentなら、ここで初期ツールを実行して最初の記憶を作る。
        # ReflexAgentでは何も起きない。
        self._before_loop()

        for step in range(1, self.max_steps + 1):
            # このステップで呼ぶ対象を決める。
            #   記憶を持つAgent … tasksに書かれた対象だけを提示し、呼び出しを強制する
            #   ReflexAgent     … 有効な全ツールを提示し、呼ぶかどうかも委ねる
            targets, tool_choice = self._select_invokables()

            # 実行対象として書かれた名前が1つも解決できなかった場合。
            # 「実行すべきものが無い」のではなくタスクの記述が誤っているので、
            # 失敗として記録し、memory更新でLLM自身に直させる。
            # ここを素通りさせると、未実行のタスクを残したまま
            # 「実行すべきタスクが無くなった」という前提で回答させることになる。
            if not targets and self._unresolved_targets:
                entries = [
                    ToolHistoryEntry(
                        name=name, kwargs={}, result=ToolResult(error=reason)
                    )
                    for name, reason in self._unresolved_targets
                ]
                self.tool_history.extend(entries)
                if self._process_step(entries):
                    return self._final_answer(
                        Phase.STALLED_ANSWER, notice=self._pending_task_notice()
                    ), step
                continue

            # 実行すべきものが無い ＝ 追加実行なしで答えられる状態。
            # toolsを渡さないのではなく、そもそも呼び出しフェーズへ入らない。
            if not targets:
                return self._final_answer(
                    Phase.ANSWER, notice=self._pending_task_notice()
                ), step

            # 呼び出しフェーズ。ここでLLMに決めさせるのは「引数」だけで、
            # 何を呼ぶかは既にtargetsで絞られている。
            response = self._generate(
                Phase.FUNCTION_CALL, tools=targets, tool_choice=tool_choice
            )

            # --- 呼び出し要求が返ってこなかった場合の3つの分岐 ---
            if not response.has_calls:
                if not response.text:
                    # テキストも呼び出し要求も無い異常な応答。
                    # ステップを1つ消費して、次の周で再試行する。
                    continue

                if tool_choice != "any" and not self.output_schema:
                    # 呼ぶかどうかをモデルに委ねていた場合（ReflexAgent）は、
                    # テキストで答えてきた時点で終了して構わない。
                    # それが「調べる必要はなかった」という判断の表明になる。
                    return response.text, step

                # ここへ来るのは次の2つの場合。
                #   ・呼び出しを強制した（"any"）のに、答えを返してきた
                #   ・output_schemaがあるのにテキストで答えてきた
                # どちらもこのテキストをそのまま最終回答にはできない。
                # このテキストは「呼び出しフェーズ用のプロンプト」で生成されて
                # いるため、回答フェーズの指示も出力形式の強制も適用されていない。
                # 改めて回答フェーズで答えさせる。
                return self._final_answer(
                    Phase.ANSWER, notice=self._pending_task_notice()
                ), step

            # --- 実際に呼ぶ ---
            # 無効化の確認、重複の拒否、実行前判定はこの中で行われる。
            entries = self._execute_calls(response.function_calls)

            # 結果を評価して記憶へ書く。同時に「進んでいるか」も判定する。
            # 進んでいないと判断されたらTrueが返り、打ち切る。
            if self._process_step(entries):
                return self._final_answer(
                    Phase.STALLED_ANSWER, notice=self._pending_task_notice()
                ), step

        # max_stepsを使い切った。ここまでに分かったことで回答する。
        return self._final_answer(
            Phase.CUTOFF_ANSWER, notice=self._pending_task_notice()
        ), self.max_steps

    @staticmethod
    def _step_fingerprint(entries: list[ToolHistoryEntry]) -> frozenset:
        """
        そのステップで何が起きたかの指紋。


        名前・引数・エラー内容の集合で表す。前ステップと一致するなら、
        新しい情報が1つも入っていないということ。


        成功した呼び出しは重複拒否で弾かれるため、指紋が一致する状況は
        「失敗か拒否の繰り返し」に限られる。逆に、1件でも成功したり
        エラーの内容が変われば指紋は変わり、進捗とみなされる。
        """
        return frozenset(
            (
                e.name,
                json.dumps(e.kwargs, sort_keys=True, ensure_ascii=False, default=str),
                e.result.error,
            )
            for e in entries
        )

    def _process_step(self, entries: list[ToolHistoryEntry]) -> bool:
        """
        1ステップ分の結果を処理し、停滞で打ち切るべきならTrueを返す。


        同じ指紋が続いた場合、まずmemory更新へ「進んでいない」ことを明示して
        立て直しを促す。それでも変わらなければ打ち切る。
        max_stepsまで回しても進捗ゼロのまま推論を払い続けるのを避ける。
        """
        fingerprint = self._step_fingerprint(entries)
        if fingerprint and fingerprint == self._last_fingerprint:
            self._stalled_count += 1
        else:
            self._stalled_count = 0
        self._last_fingerprint = fingerprint

        if self._stalled_count >= self.max_stalled_steps:
            self.interceptor.notify.error(
                f"{self.name}が同じ実行を{self._stalled_count + 1}回繰り返し、"
                f"進捗しなくなったため打ち切ります。"
            )
            return True

        # 1度目の繰り返しでは、状況を伝えたうえでmemory更新をやり直させる。
        self._after_tools(entries, notice=STALL_NOTICE if self._stalled_count else "")
        return False

    def _pending_task_notice(self) -> str:
        """
        実行されないまま残ったタスクを、最終回答の生成へ知らせる文言。


        actionable_tasks()が拾うのはnextとnext_parallelだけなので、
        conditionalのまま残ったタスクは実行対象にならない。
        前提が満たされたのにnextへ上げ忘れた場合も、静かに実行対象が空になる。
        打ち切りや上限到達で終わる場合は、nextのまま残ることもある。


        そのまま回答させると「実行していないことを、実行したかのように」書く。
        実際に、予約タスクがconditionalのまま残った状態で
        「予約を受け付け、受付番号の発行手続きを進めました」と回答した例がある。
        実行の記録が無いにもかかわらず、失敗としては現れない。


        ここで止めずに知らせるだけにしているのは、残っているのが
        正常な場合もあるため（前提が満たされていないので実行しない）。
        誤りかどうかは記憶を読まないと決まらないので、判断は生成側へ渡す。
        """
        unfinished = (TaskStatus.NEXT, TaskStatus.NEXT_PARALLEL, TaskStatus.CONDITIONAL)
        pending = [t for t in self.private_memory.tasks if t.status in unfinished]
        if not pending:
            return ""

        lines = [f"- [{t.id}] ({t.status.value}) {t.text}" for t in pending]
        return section(
            "実行されていないタスク",
            "次のタスクは完了しておらず、実行もされていない。\n"
            + "\n".join(lines)
            + "\n\nこれらが実行済みであるかのように回答してはならない。"
            "実行すべきだったものが実行できていない場合は、"
            "その事実と、何が足りなかったのかを回答に明示する。",
        )

    def _final_answer(self, phase: Phase, notice: str = "") -> str:
        """
        toolを渡さずに最終回答を生成する。


        output_schemaが指定されている場合はresponse_schemaとして渡し、
        形式をプロンプトでの依頼ではなく構造で強制する
        （memory差分で使っているのと同じ仕組みを、回答側でも使う）。
        Geminiはスキーマで直接強制でき、OpenAI/Claudeは指示文へ変換される。
        後者は強制しきれないため、生成後に解釈できるかを確かめて1度だけ作り直す。


        notice: 回答の前に伝えておく必要がある事情（未実行のタスクなど）。
        """
        spec = PHASE_SPECS[phase]
        schema = self.output_schema if spec.respects_output_schema else None

        response = self._generate(phase, response_schema=schema, extra_prompt=notice)

        if schema and response.text and not self._matches_output_schema(response.text):
            self.interceptor.notify.error(
                f"{self.name}の回答がoutput_schemaに従っていないため、作り直します。"
            )
            response = self._generate(
                phase,
                response_schema=schema,
                # noticeも一緒に渡す。作り直しで落とすと、
                # 2回目だけ未実行タスクを知らない状態で回答が作られる。
                extra_prompt=join_sections(
                    notice,
                    section(
                        "前回の応答の問題",
                        "指定された形式のJSONとして解釈できなかった。"
                        "説明文やコードブロックの記号を付けず、JSONのみを出力すること。",
                    ),
                ),
            )

        if response.text:
            return response.text
        return "【システム通知】回答の生成に失敗しました。memoryに残っている調査結果を確認してください。"

    def _matches_output_schema(self, text: str) -> bool:
        """
        回答がoutput_schemaの形として解釈できるか。


        全項目の検証はしない。ここで防ぎたいのは「JSONですらない」
        「オブジェクトを求めたのに配列が返った」という、呼び出し側が
        パースした時点で落ちる類の不一致であり、それはトップレベルの型で判る。
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False

        expected = (self.output_schema or {}).get("type")
        if expected == "object":
            return isinstance(parsed, dict)
        if expected == "array":
            return isinstance(parsed, list)
        return True

    # ---- ここから下がhook。ReflexAgentが上書きする ----
    def _before_loop(self) -> None:
        """
        ループ開始前。初期toolを実行し、その結果も材料にして初期memoryを作る。


        初期tool結果は、履歴経由ではなく明示的に見せる。
        _show_tool_results=Falseのため履歴からは隠れてしまうが、
        このステップではまさにその中身を材料にするため。
        """
        entry = self._execute_initial_tool()
        tool_results = entry.render(include_result=True) if entry else ""
        self._update_memory(phase=Phase.INITIAL_MEMORY, tool_results=tool_results)

    def _after_tools(
        self, entries: list[ToolHistoryEntry], *, notice: str = ""
    ) -> None:
        """
        実行直後。今回の結果を評価してmemoryへ反映する。


        notice: 停滞を検出した時に差し込む注意文。
            memory更新はtasksのstatusを変える唯一の場所なので、
            立て直しを促すならここへ渡すのが最短経路になる。
        """
        tool_results = "\n".join(e.render(include_result=True) for e in entries)
        self._update_memory(
            phase=Phase.MEMORY_UPDATE,
            tool_results=tool_results,
            notice=notice,
            executed=entries,
        )

    def _select_invokables(self) -> tuple[list[Invokable], str | None]:
        """
        このステップで提示するtool/agentと、tool_choiceを決める。


        tasksでnext / next_parallelになっているものだけを提示し、"any"で強制する。
        「全部見せてtasksに従うようお願いする」のではなく、選ばせない。
        提示するものが無ければ空リストを返し、ループは最終回答へ向かう。


        解決できなかった名前は_unresolved_targetsへ記録する。黙って除外すると、
        タスクがnextのまま残っているのに実行対象が空になり、そのまま最終回答へ
        飛んでしまう（memory更新が一度も走らないので、LLMは誤りに気付けない）。
        """
        targets = []
        seen = set()
        self._unresolved_targets = []
        for task in self.private_memory.actionable_tasks():
            for name in task.target_names:
                if name in seen:
                    continue
                seen.add(name)
                found = self._resolve_invokable(name)
                if found is None:
                    self._unresolved_targets.append(
                        (name, "存在しないtool/agentです。")
                    )
                elif found.disabled:
                    self._unresolved_targets.append((name, "現在無効化されています。"))
                else:
                    targets.append(found)
        return targets, "any"

    # ==========================================
    # 生成呼び出し
    # ==========================================
    def _generate(
        self,
        phase: Phase,
        *,
        tools: list | None = None,
        tool_choice: str | None = None,
        tool_results: str = "",
        extra_prompt: str = "",
        response_schema: Any | None = None,
    ) -> LLMResponse:
        """
        1回の生成。phaseに応じた設定・プロンプト・添付をまとめて組み立てる。


        ここを通した呼び出しだけがトークンを積算するので、
        使用量の集計漏れが起きない。
        """
        spec = PHASE_SPECS[phase]
        cfg = self.resolve_generation(phase)

        # 会話履歴はプロンプト文字列へ埋め込まず、messagesとして渡す。
        # 埋め込むと、モデルが本来持っているマルチターンの扱い
        # （どこまでが過去の発話か）を捨てて、ただの長文として読ませることになる。
        # プロンプトの組み立ては1回だけ行う。再試行のたびに作り直しても
        # 同じものができるが、記憶を読んで文字列を組む処理を無駄に繰り返す。
        prompt = join_sections(
            self._build_prompt(phase=phase, tool_results=tool_results),
            extra_prompt,
        )
        system_instruction = self._build_system_instruction(phase)

        # 実際に使うモデル。再試行で差し替わりうるので、cfgとは別に持つ。
        model = cfg.model

        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                response = cfg.llm.generate(
                    model=model,
                    prompt=prompt,
                    messages=self.current_input.history if self.current_input else None,
                    system_instruction=system_instruction,
                    blobs=self._blobs_for(spec),
                    tools=tools,
                    tool_choice=tool_choice,
                    response_schema=response_schema,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    thought_level=cfg.thought_level,
                )
            except Exception as e:
                # 再試行するかどうかはフレームワークが決めない。
                # 503のような一時的な過負荷は待てば通ることが多いが、
                # どれだけ待つか・何回試すかは運用の方針であり、
                # 対話UIとバッチで正解が違う。
                # 登録が無ければ再試行せず、例外をそのまま外へ出す。
                decision = self.interceptor.check.generation_failed(
                    agent=self.name,
                    phase=phase,
                    model=model,
                    attempt=attempt,
                    error=e,
                )
                if not decision.retry:
                    raise
                # モデル名が返った場合はそれに差し替える。枠を使い切った
                # モデルは待っても同じエラーが返るため、同じ条件での
                # やり直しでは越えられない失敗がある。
                if decision.model:
                    model = decision.model
                continue
            break

        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens

        self.interceptor.notify.generated(
            GenerationEvent(
                agent=self.name,
                phase=phase,
                model=model,
                elapsed=time.perf_counter() - started,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        )
        return response

    # ==========================================
    # memoryの更新
    # ==========================================
    def _writable_memories(self) -> tuple:
        """
        差分の書き込み先。sharedとprivateの両方。


        ReflexAgentも_write_tool_result_to_memory経由でここを通るため、
        あちらはsharedのみへ絞るようオーバーライドしている。
        """
        return (self.shared_memory, self.private_memory)

    def _stale_task_errors(self, executed: list[ToolHistoryEntry]) -> list[DiffError]:
        """
        実行済みの対象がstatusに反映されているかを検証する。


        tasksは「次に何を実行するか」をシステムへ指示する場所であり、
        記録用のメモではない。実行し終えたものがnextのまま残ると、
        次のステップで同じ対象がまた選ばれる。重複した呼び出しは拒否されるが、
        そのステップは無駄になり、max_stepsを使い切るまで同じことが続く。


        逆に、実行したのにstatusを進めないまま「実行対象が無くなった」と
        判断される経路もある。その場合は実行していないタスクを残したまま回答へ進む。


        そのためこれは記憶の書き方の誤りとして差し戻す。
        通常のDiffErrorと同じ経路に乗せることで、失敗した行を添えた
        再生成が1回分そのまま使える。


        失敗した実行は対象にしない。再試行のためにnextのまま残るのが正しい。
        """
        succeeded = {e.name for e in executed if e.result.success}
        if not succeeded:
            return []

        pending = (TaskStatus.NEXT, TaskStatus.NEXT_PARALLEL)
        return [
            DiffError(
                "tasks",
                {"field": "tasks", "id": task.id, "status": task.status.value},
                f"このタスクの対象（{'、'.join(task.target_names)}）は今回実行済みだが、"
                f"statusが{task.status.value}のまま残っている。"
                "done（完了）/ unnecessary（不要になった）/ conditional（前段の結果次第）"
                "のいずれかへ必ず更新すること。",
            )
            for task in self.private_memory.tasks
            if task.status in pending and succeeded & set(task.target_names)
        ]

    def _update_memory(
        self,
        *,
        phase: Phase,
        tool_results: str = "",
        notice: str = "",
        executed: list[ToolHistoryEntry] | None = None,
    ) -> list[DiffError]:
        """
        LLMに記憶の差分（JSON配列）を作らせて、記憶へ反映する。
        初期構築（INITIAL_MEMORY）と更新（MEMORY_UPDATE）の両方でここを通る。


        処理の流れ:
          1. そのphaseで書き込めるプロパティから、差分用のJSONスキーマを組む
          2. LLMに差分を生成させる
          3. 適用する。失敗した行があれば、その行だけを差し戻して再生成させる
          4. 全部通るか、リトライ上限に達するまで2〜3を繰り返す


        失敗した行だけを差し戻すのは、通った分の作業を捨てないためと、
        どこが悪かったかを明示した方が直しやすいため。全体をやり直させると
        既に正しく書けた項目まで作り直させることになる。


        リトライを使い切っても処理は中断しない。通った分だけを残して進む。
        記憶が多少欠けていても、ループを続けて回答へ向かった方が、
        利用者には価値が残るという判断（欠けた事実はerrorイベントで通知される）。
        """
        memories = self._writable_memories()
        spec = PHASE_SPECS[phase]
        schema = build_diff_schema(*memories, allow_disable=spec.allows_disable)

        # スキーマが形式を保証していても、粒度は例で示した方が安定する。
        # 「fieldに書き込み先を指定する」という構造も例で伝わる。
        base_extra = join_sections(
            section("停滞の検出", notice),
            section("出力例", DIFF_EXAMPLE),
        )
        extra = base_extra

        errors: list[DiffError] = []
        # ループ内で必ず代入されるが（max(1, ...)により最低1回は回る）、
        # 型チェッカーはそこまで推論しないため明示的に初期化する。
        rows: list | None = None
        # 0以下だとループが一度も回らず、memoryを作らないまま「更新しました」と
        # 通知してしまう。最低1回は必ず生成する。
        for attempt in range(1, max(1, self.max_memory_retries) + 1):
            response = self._generate(
                phase,
                tool_results=tool_results,
                extra_prompt=extra,
                response_schema=schema,
            )

            rows = _extract_rows(response.text)
            if rows is None:
                errors = [
                    DiffError(
                        "(全体)", response.text[:200], "JSONの配列として解釈できません"
                    )
                ]
            else:
                # 無効化の指示はmemoryへの書き込みではないので、先に取り出して処理する。
                if spec.allows_disable:
                    self._apply_disable_rows(rows)
                errors = apply_diff(rows, *memories)
                # 書き込みが通ったうえで、実行済みのタスクが放置されていないかを見る。
                # apply_diffは1行ずつの妥当性しか見ないため、
                # 「書かれなかったこと」はここでしか検出できない。
                if executed:
                    errors.extend(self._stale_task_errors(executed))

            # 適用の直後に、材料・出力・結果を揃えて流す。
            # ループの内側に置くのは、再生成が起きた時に
            # 「1回目に何を間違え、2回目にどう直したか」まで観測させるため。
            # 外に出すと最後の1回しか残らない。
            self.interceptor.notify.memory_diff(
                MemoryDiffEvent(
                    agent=self.name,
                    phase=phase,
                    attempt=attempt,
                    # 観測側が書き換えても本体へ影響しないよう浅く複製する。
                    # 通知は観測のための仕組みであり、観測が対象を変えてよい理由がない。
                    rows=list(rows or []),
                    errors=list(errors),
                    tool_results=tool_results,
                    raw_text="" if rows is not None else response.text,
                )
            )

            if not errors:
                break

            # 失敗内容を添えて再生成させる。通った分は既に適用済みなので、
            # やり直させるのは失敗した行だけでよい。
            # 毎回ベースから作り直す。前回のextraへ積み増すと、既に修正済みの
            # 古い失敗リストまで「失敗した項目」として提示され続ける。
            extra = join_sections(
                base_extra,
                section(
                    "前回の更新で失敗した項目",
                    "以下は反映されなかった。原因を確認し、修正した分のみを出力すること。\n"
                    + "\n".join(e.render() for e in errors),
                ),
            )

        if errors:
            # リトライを使い切っても中断しないため、通知しなければ
            # 記憶が欠けたまま進んだことに誰も気付けない。
            detail = "\n".join(e.render() for e in errors)
            self.interceptor.notify.error(
                f"{self.name}のmemory更新で{len(errors)}件が反映されませんでした。\n{detail}"
            )
        else:
            # 何を更新したかを添える。「更新しました」だけでは、
            # 観測側が実際に記憶が育っているのか判断できない。
            # ここへ到達するのは全行が適用された時なので、rowsがそのまま適用内容になる。
            applied = [
                f"{r.get('field')}:{r.get('id')}"
                for r in (rows or [])
                if isinstance(r, dict) and r.get("field") != DISABLE_FIELD
            ]
            detail = f"（{' / '.join(applied)}）" if applied else "（変更なし）"
            self.interceptor.notify.memory_updated(
                f"{self.name}のmemoryを更新しました。{detail}"
            )
        return errors

    def _apply_disable_rows(self, rows: list) -> None:
        """
        差分に含まれる無効化指示を適用する。


        無効化されたtool/agentは_select_invokablesと_render_capability_catalogの
        両方から外れるため、以降のステップでは存在自体が見えなくなる。
        「使わないでください」と依頼するのではなく、選択肢から消す。


        無効化した名前は記録しておき、次の起動時に戻す。戻さないと、
        このAgentが再度呼ばれた時に前回の判断が残ったまま実行される
        （無効化はセッションの依頼内容にもとづく判断であり、恒久設定ではない）。
        既に設定として無効化されているものは、記録に含めないため戻されない。
        """
        for row in rows:
            if not isinstance(row, dict) or row.get("field") != DISABLE_FIELD:
                continue

            name = str(row.get("id") or "")
            target = self._resolve_invokable(name)
            if target is None:
                # 提示していない名前を無効化しようとした場合。処理は続ける。
                self.interceptor.notify.error(
                    f"{self.name}が存在しない対象の無効化を指示しました: '{name}'"
                )
                continue
            if target.disabled:
                continue  # 元から無効。戻す対象にもしない

            target.disabled = True
            self._runtime_disabled.add(name)
            self.interceptor.notify.memory_updated(
                f"{self.name}が{name}を無効化しました。理由: {row.get('text') or '(記載なし)'}"
            )

    def _restore_disabled(self) -> None:
        """このAgentがセッション中に無効化したものを元に戻す。"""
        for name in self._runtime_disabled:
            target = self._resolve_invokable(name)
            if target is not None:
                target.disabled = False
        self._runtime_disabled = set()

    # ==========================================
    # tool / agent の呼び出し
    # ==========================================
    def _resolve_invokable(self, name: str) -> Invokable | None:
        """
        名前からtoolまたはsub_agentを引く。


        Invokableを満たす限り、呼び出し側はどちらなのかを知る必要がない。
        toolを先に探すのは、同名が存在した場合に単機能処理を優先するため
        （そもそも同名にすべきではないので、実質は探索順の明示）。
        """
        tool = self.get_tool(name)
        if tool is not None:
            return tool
        return self.get_sub_agent(name)

    @staticmethod
    def _call_signature(call: FunctionCall) -> str:
        """
        呼び出しの同一性を判定するための文字列。
        引数の辞書は順序が変わりうるのでsort_keysで正規化する。
        JSONにできない値（bytes等）が来てもdefault=strで落ちないようにしておく。
        """
        args = json.dumps(call.args, sort_keys=True, ensure_ascii=False, default=str)
        return f"{call.name}({args})"

    def _execute_calls(self, calls: list[FunctionCall]) -> list[ToolHistoryEntry]:
        """
        LLMが要求した呼び出しをすべて実行し、履歴へ積んで返す。


        戻り値は今回実行した分だけ。memory更新フェーズはこれを評価対象にする
        （履歴全体ではなく今回の差分を渡したいため）。


        例外は投げない。存在しない名前を指定された場合も、失敗した結果として
        返すことで、次のmemory更新でLLM自身が気付いて計画を直せるようにする。


        処理を3段に分けている。並列にしてよいのは真ん中だけ。


          ① 判定   … 重複拒否と対象の解決。順序に依存するので逐次
          ② 実行   … 2件以上なら並列。外部の待ち時間が重なる
          ③ 反映   … 記憶への書き込みと履歴への追加。呼び出し順のまま逐次


        ①を並列にできないのは、重複の判定が「他の呼び出しをすでに通したか」を
        見る判定であるため。並列にすると同じ引数の呼び出しが両方走りうる。
        ③を並列にできないのは、記憶への書き込みが同じidを探して差し替える
        操作であり、複数スレッドから同時に行うと片方が消えるため。
        """
        # --- ① 判定（逐次） ---------------------------------------------
        # 実行するもの、拒否するもの、解決できなかったものを先に振り分ける。
        # ここで確定させておけば、②の順序が結果に影響しない。
        succeeded_now = set()
        planned = set()
        plan: list[tuple[FunctionCall, Invokable | None, ToolResult | None]] = []

        for call in calls:
            signature = self._call_signature(call)

            # 直前のステップで成功済みの呼び出し、または今回のバッチで
            # すでに実行することにした呼び出しは、同じ結果しか返さないため
            # 二重に実行しない。
            # プロンプトで「同じ呼び出しを繰り返すな」と依頼するのではなく、
            # 実際に止める。外部APIの無駄打ちと、同じ場所での停滞を防ぐ。
            #
            # ステップをまたぐ判定を「成功したもの」に限っているのは、失敗した
            # 呼び出しのやり直しは正当なリトライであり、止めると復帰できなくなるため。
            # 一方、同じバッチの中では成否がまだ分からないので、実行することに
            # した時点で止める（同じ相手へ同じ引数で同時に2回投げる理由はない）。
            if signature in self.last_successful_calls or signature in planned:
                plan.append(
                    (
                        call,
                        None,
                        ToolResult(
                            error="直前と完全に同じ呼び出しのため実行しませんでした。"
                            "同じ結果しか得られません。すでに得られている結果をもとに、"
                            "次に必要な行動を判断してください。"
                        ),
                    )
                )
                # 拒否したものも「結果を既に持っている」側へ引き継ぐ。
                # ここで落とすと、次のステップでは「直前に成功したもの」が空になり、
                # 拒否と実行が交互に繰り返されてしまう。
                succeeded_now.add(signature)
                continue

            planned.add(signature)

            target = self._resolve_invokable(call.name)
            if target is None:
                # 提示していないものを呼んできた場合。ここへ来る時点でLLMの逸脱か
                # tasksの記述ミスなので、事実として記録して次の判断材料にする。
                plan.append(
                    (
                        call,
                        None,
                        ToolResult(
                            error=f"'{call.name}' というtool/agentは存在しません。"
                        ),
                    )
                )
                continue

            plan.append((call, target, None))

        # --- ② 実行（単位ごとに並列） -------------------------------------
        # 並列にする単位を作る。単位の中は順番に実行される。
        #
        #   tool  … 1件ごとに別の単位。関数を呼ぶだけで状態を持たないため
        #   agent … 同じ実体への呼び出しをまとめて1つの単位にする
        #
        # Agentは記憶・履歴・受け取った依頼を自分のフィールドに持つ。同じ実体を
        # 2つのスレッドから同時に走らせると、片方の記憶にもう片方の結果が混ざる。
        # 引数が違えば重複拒否も働かないので、ここで分けておく必要がある。
        units: list[list[tuple[int, FunctionCall, Invokable]]] = []
        agent_units: dict[int, list[tuple[int, FunctionCall, Invokable]]] = {}

        for index, (call, target, done) in enumerate(plan):
            if done is not None or target is None:
                continue
            item = (index, call, target)
            if isinstance(target, Agent):
                unit = agent_units.get(id(target))
                if unit is None:
                    unit = agent_units[id(target)] = []
                    units.append(unit)
                unit.append(item)
            else:
                units.append([item])

        results: dict[int, ToolResult] = {}

        def run_unit(
            unit: list[tuple[int, FunctionCall, Invokable]],
        ) -> dict[int, ToolResult]:
            return {
                index: target.execute(kwargs=call.args, interceptor=self.interceptor)
                for index, call, target in unit
            }

        if len(units) > 1:
            # スレッド数は単位の数と同じにする。tool_choice="any"で提示するのは
            # tasksが指定した対象だけなので、この数はtasksの件数で抑えられている。
            #
            # interceptorのコールバックはInterceptor側で直列化してあるため、
            # 登録した処理が同時に走ることはない（順序は終了順になる）。
            # tool本体の関数は同時に呼ばれる。外部の状態を書き換えるtoolを
            # 並列で呼ばせる場合は、利用側で保護する必要がある。
            with ThreadPoolExecutor(max_workers=len(units)) as pool:
                futures = {pool.submit(run_unit, unit): unit for unit in units}
                for future in as_completed(futures):
                    try:
                        results.update(future.result())
                    except Exception as e:
                        # Invokableは例外を投げない契約だが、スレッドの側で
                        # 何かが起きた場合にここで落とすと1件の失敗が全体を止める。
                        self.interceptor.notify.error(
                            f"並列実行が例外で終了しました: {e}"
                        )
                        for index, _, _ in futures[future]:
                            results.setdefault(
                                index,
                                ToolResult(error=f"実行中にエラーが発生しました: {e}"),
                            )
        else:
            for unit in units:
                results.update(run_unit(unit))

        # --- ③ 反映（呼び出し順のまま逐次） -------------------------------
        entries = []
        for index, (call, target, done) in enumerate(plan):
            result = done if done is not None else results[index]

            entry = ToolHistoryEntry(
                name=call.name,
                kwargs=call.args,
                result=result,
                # 結果の解釈指示と値の意味は、呼ばれた側が持つ。
                # 呼んだ側の人格定義へ書き写さない。
                evaluation=getattr(target, "evaluation", "") if target else "",
                output_schema=getattr(target, "output_schema", None)
                if target
                else None,
            )

            # write_to_memory=Trueのtoolは、結果をLLMに要約させず直接記録する。
            # _after_toolsではなくここで行うのは、評価フェーズを持たない
            # ReflexAgentでも同じように機能させるため。
            if getattr(target, "write_to_memory", False) and result.success:
                entry.written_to_memory = self._write_tool_result_to_memory(
                    tool_name=call.name, value=result.value
                )

            if target is not None and result.success:
                succeeded_now.add(self._call_signature(call))

            self.tool_history.append(entry)
            entries.append(entry)

        # 次のステップでの「直前」はこのバッチになる。
        self.last_successful_calls = succeeded_now
        return entries

    def _write_tool_result_to_memory(self, *, tool_name: str, value: Any) -> bool:
        """
        toolの戻り値をmemoryへ直接書き込む。書き込めたらTrueを返す。


        戻り値はLLMの差分と同じ形（fieldとtextを持つ行の配列）を期待する。
        形が違う場合は書き込みを行わず、通知だけ出して処理を続ける。
        toolの実装ミスであってセッションを止める理由ではないため。
        """
        if not isinstance(value, list):
            self.interceptor.notify.error(
                f"{tool_name}はwrite_to_memory=Trueだが、戻り値が配列ではありません。"
            )
            return False

        errors = apply_diff(value, *self._writable_memories(), id_prefix=tool_name)

        # LLMを介さない書き込みも記憶を変える以上、同じ経路で観測できる必要がある。
        # 「記憶に何が入ったか」を追う側が、書いた主体ごとに別の窓を見に行かずに済む。
        self.interceptor.notify.memory_diff(
            MemoryDiffEvent(
                agent=self.name,
                phase=None,
                attempt=1,
                rows=list(value),
                errors=list(errors),
                source_tool=tool_name,
            )
        )

        if errors:
            self.interceptor.notify.error(
                f"{tool_name}の結果のうち{len(errors)}件をmemoryへ書き込めませんでした。\n"
                + "\n".join(e.render() for e in errors)
            )
        # 1件でも通っていれば記録済みとして扱う
        return len(errors) < len(value)

    # ==========================================
    # 初期tool
    # ==========================================
    def _execute_initial_tool(self) -> ToolHistoryEntry | None:
        """
        ループ開始前に、システムが機械的にtoolを1つ実行する。


        LLMに「まずこれを呼んで」と依頼するのではなく、確定している手順を
        システムが実行してしまう。呼ぶかどうかをLLMに判断させない分、
        1ステップ分の推論とその失敗可能性がまるごと消える。


        引数は受け取った構造化引数（input_schema準拠）から、そのtoolが
        受け取れるものだけを抜き出して渡す。受け取れる引数の一覧は
        関数シグネチャから導出されたJSON Schemaを正とするため、
        別途手で列挙する必要がない。
        """
        if not self.initial_tool_name or not self.current_input:
            return None

        tool = self.get_tool(self.initial_tool_name)
        if tool is None:
            # データ由来のエラーではなく配線ミスなので、黙って続行せず落とす。
            raise ValueError(
                f"{self.name}のinitial_tool_name '{self.initial_tool_name}' が"
                f"toolsの中に見つかりません。"
            )

        accepted = tool.get_json_schema()["properties"].keys()
        args = {k: v for k, v in self.current_input.kwargs.items() if k in accepted}

        # execute()は例外を投げずToolResultを返すため、失敗しても
        # そのままループへ進める（失敗した事実は履歴に残る）。
        result = tool.execute(kwargs=args, interceptor=self.interceptor)
        # 初期toolの結果はINITIAL_MEMORYのプロンプトに載るため、LLMが同じ呼び出しを
        # 要求してくることがある。署名を登録しておかないと同一引数で2回実行される。
        if result.success:
            self.last_successful_calls.add(
                self._call_signature(
                    FunctionCall(name=self.initial_tool_name, args=args)
                )
            )
        # 通常の呼び出しと同じく、呼ばれた側が持つ解釈指示を添える。
        # ここで渡し忘れると、初期toolに指定したtoolのevaluationだけが
        # 使われないことになる。初期toolは「最初に必ず実行するもの」なので、
        # その結果をどう読むかの指示が最も必要な場面で欠ける。
        entry = ToolHistoryEntry(
            name=self.initial_tool_name,
            kwargs=args,
            result=result,
            evaluation=tool.evaluation,
        )

        # 通常の呼び出し（_execute_calls）と同じくwrite_to_memoryを尊重する。
        # ここを忘れると、初期toolに指定した場合だけLLMの要約を経由することになり、
        # 「値を一字一句正確に残す」というwrite_to_memoryの目的が失われる。
        if tool.write_to_memory and result.success:
            entry.written_to_memory = self._write_tool_result_to_memory(
                tool_name=self.initial_tool_name, value=result.value
            )

        self.tool_history.append(entry)
        return entry

    # ==========================================
    # プロンプト組み立て
    # ==========================================
    def _render_capability_catalog(self) -> str:
        """自分が使えるtool / agentの一覧。disabledなものは存在自体を見せない。"""
        lines = [t.to_catalog_line() for t in self.tools if not t.disabled]
        lines += [a.to_catalog_line() for a in self.sub_agents if not a.disabled]
        return "\n".join(lines)

    def _blobs_for(self, spec: PhaseSpec) -> list[dict]:
        """
        そのphaseでLLMへ渡す添付。


        渡すかどうかの判断はPhaseSpecが持つ（初期memory構築時と最終回答時のみ）。
        ReflexAgentはmemoryを持たず文字起こしができないため、
        このメソッドをオーバーライドして常に渡す。
        """
        if not self.current_input or not spec.include_blobs:
            return []
        return self.current_input.blobs

    def _render_memory(self, spec: PhaseSpec) -> str:
        """
        プロンプトに載せるmemory。sharedとprivateの両方。


        ガイド文と中身を分離せず交互に並べるのは意図的な設計。
        ガイドから離れた位置に中身があると、LLMが書き方の指示を無視しやすくなる。
        （その代わりガイド文がキャッシュに乗らないが、遵守率を取る判断）
        """
        how_to = spec.include_memory_how_to
        return join_sections(
            self.shared_memory.intro(include_how_to=how_to),
            self.shared_memory.render(include_how_to=how_to),
            self.private_memory.intro(include_how_to=how_to),
            self.private_memory.render(include_how_to=how_to),
            # 書き分けの判断はsharedとprivateをまたぐ（factsはshared、tasksはprivate）ため、
            # どちらか片方のガイドに置くと不完全になる。両方を見せた後に1回だけ置く。
            # 書き込みが発生しないphaseでは不要。
            section("書き込み先の選び方", ROUTING_GUIDE) if how_to else "",
        )

    def _render_input_schema(self) -> str:
        """
        自分が受け取る引数の意味。


        input_schemaは、これまで呼び出す側へ渡す宣言（to_declaration）でしか
        使われていなかった。委譲された側には引数の値だけがJSONで届き、
        各項目が何を意味するかは届かない。
        結果として、受け取った側はキー名から意味を推測して読むことになる。


        「back_groundに完了条件を書く」と決めても、受け取る側がそれを知らなければ、
        書かれているものを読み取れない。スキーマは呼ぶ側と呼ばれる側の
        取り決めなので、両方が同じ説明を見る必要がある。


        既定のスキーマ（messageだけ）の場合は何も返さない。
        「messageに依頼内容が入っている」ことは説明しなくても分かる。
        """
        props = self.input_schema.get("properties", {})
        if not props or set(props) == {"message"}:
            return ""

        required = set(self.input_schema.get("required", []))
        lines = [
            "委譲された場合、依頼は次の項目で届く。それぞれの意味は以下のとおり。",
            "",
        ]
        for name, spec in props.items():
            mark = "必須" if name in required else "任意"
            desc = spec.get("description", "（説明なし）")
            lines.append(f"■ {name}（{mark}）")
            lines.append(f"{desc}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _build_system_instruction(self, phase: Phase) -> str:
        """
        phase内で変化しない部分。llm.generate(system_instruction=...)へ渡す。


        毎ステップ変わるもの（memory / 実行履歴 / 依頼内容）は一切含めない。
        FUNCTION_CALLフェーズはmax_steps回繰り返されるため、ここが不変であることで
        プロンプトキャッシュが効く。
        """
        spec = PHASE_SPECS[phase]

        # tool/agentがAPIのtools引数で渡されるphaseでは、同じ内容を文章でも
        # 説明すると二重になる。逆にtoolを渡さないphase（memory構築・更新）では、
        # 何が使えるのかを知らないとタスクを立てられないため文章で見せる。
        catalog = "" if spec.allows_tools else self._render_capability_catalog()

        return join_sections(
            section("あなたに適用された人格定義", self.system_instruction),
            section("全ステップ共通の姿勢", POURING.common),
            section("あなたが受け取る依頼の形", self._render_input_schema()),
            section("会話履歴の扱い", self._history_stance()),
            section("現在のステップでやるべきこと", spec.instruction),
            section(
                "思考の方向づけ", POURING.thinking if spec.include_thinking else ""
            ),
            section("実行可能なtool / agent", catalog),
            section("ナレッジ", self.knowledge),
            # 実行環境のタイムゾーンでの今日。astimezone()でタイムゾーンを
            # 明示しているのは、UTCで動くサーバーとローカルで日付がずれるのを
            # 気付かずに通さないため（「今日」の判断は回答の正しさに直結する）。
            section("今日の日付", datetime.now().astimezone().strftime("%Y年%m月%d日")),
        )

    def _build_prompt(self, *, phase: Phase, tool_results: str = "") -> str:
        """
        毎ステップ変化する部分。llm.generate(prompt=...)へ渡す。


        依頼内容はcurrent_inputから読む。セッション中は不変だが、ルールではなく
        「答えるべき対象」なのでsystem_instruction側には置かない。


        tool_results: 今回の実行結果を評価対象として渡す
            （MEMORY_UPDATEと、初期tool実行時のINITIAL_MEMORYで使う）。
        """
        spec = PHASE_SPECS[phase]
        run = self.current_input or RunInput(message="")

        # 回答形式はここで文章として指示しない。
        # output_schemaは_final_answerがresponse_schemaとして渡し、
        # プロバイダ側で構造として強制する（強制できないプロバイダには
        # llm.py側で指示文へ変換される）。両方に書くと二重提示になる。
        return join_sections(
            section("依頼内容", run.message),
            section(
                "依頼内容の形式説明",
                str(self.input_schema) if self.input_schema else "",
            ),
            "毎ステップ依頼内容は引き継がれるため、依頼内容そのものをmemoryへ書く必要はない。",
            self._render_memory(spec),
            section("あなたの実行履歴", self.render_tool_history()),
            section("今回の実行結果", tool_results),
        )

    def _history_stance(self) -> str:
        """
        チャット履歴の扱い方。履歴が存在する時だけ提示する。


        履歴をmessagesとして渡すと、モデルは過去のassistantターンを
        「自分がそう言った」＝正しいこととして扱いやすくなる。
        文字列で埋め込んでいた時より、この格下げの指示はむしろ重要になる。


        phase内で不変なのでsystem_instruction側へ置く。
        """
        if not self.current_input or not self.current_input.history:
            return ""
        return (
            "直前のやり取りは会話履歴として渡されている。"
            "今回答えるべき要求は、履歴ではなく依頼内容の方である。\n"
            "履歴は、依頼内容に含まれる省略や指示対象を理解するための"
            "背景情報としてのみ使用する。\n"
            "履歴に含まれる過去の自分の回答は事実ではない。「前回そう回答した」という"
            "記録にすぎず、対応可否・ツールの可否・業務ルールの根拠にしてはならない。\n"
            "ユーザーが過去の回答を否定・修正している場合は、その否定内容を主要求として採用する。"
        )


# ==========================================
# ReflexAgent: 記憶を持たず、その場の入力だけで動くAgent。
#
# Russell & Norvig の simple reflex agent（内部状態を持たず、現在の入力から
# 直接行動を決めるエージェント）から名前を取っている。
#
# Agentとの違いは「初期思考」と「評価」の両端が無いこと。
# ループの骨格は親と同じものを使い、hookを空にすることで表現する。
# ==========================================
class ReflexAgent(Agent):
    # memoryへ要約しないため、tool結果は隠さず履歴に残し続ける。
    _show_tool_results: ClassVar[bool] = True

    def _writable_memories(self) -> tuple:
        """
        private_memoryは使わない。プロンプトにも載せないため、
        write_to_memoryのtoolがgoalsやtasksへ書き込めてしまうと、
        誰にも読まれない場所に情報が消える。書き込み先をsharedだけに限る。
        """
        return (self.shared_memory,)

    def _before_loop(self) -> None:
        """初期toolは実行するが、初期memoryは作らない。"""
        self._execute_initial_tool()

    def _after_tools(
        self, entries: list[ToolHistoryEntry], *, notice: str = ""
    ) -> None:
        """
        評価フェーズを持たない。結果は履歴に残るだけ。


        memory更新の仕組みが無いため、停滞時の立て直し（notice）も効かない。
        繰り返しを検出した場合は立て直しを挟まずそのまま打ち切られる。
        """

    def _select_invokables(self) -> tuple[list[Invokable], str | None]:
        """
        tasksを持たないため、有効なtool/agentを全て提示して"auto"で選ばせる。
        古典的なReActループそのもの。テキストを返した時点でループが終わる。
        """
        # 型注釈を明示しているのは、注釈が無いと右辺から list[Tool] と推論され、
        # 戻り値の list[Invokable] と食い違うため。
        # listは不変（invariant）なので、要素がInvokableを満たしていても
        # list[Tool] を list[Invokable] として扱うことは許されない
        # （list[Invokable]として受け取った側がAgentを追加できてしまい、
        # 元のlist[Tool]という約束が壊れるから）。
        targets: list[Invokable] = [t for t in self.tools if not t.disabled]
        targets += [a for a in self.sub_agents if not a.disabled]
        return targets, "auto"

    def _render_memory(self, spec: PhaseSpec) -> str:
        """shared_memoryは読む。private_memoryは持たない（更新する仕組みが無い）。"""
        return join_sections(
            self.shared_memory.intro(include_how_to=False),
            self.shared_memory.render(include_how_to=False),
        )

    def _blobs_for(self, spec: PhaseSpec) -> list[dict]:
        """
        添付は常に渡す。
        memoryへ文字起こしする仕組みが無いため、渡さないと二度と参照できない。
        """
        if not self.current_input:
            return []
        return self.current_input.blobs
