"""
Agentの本体。ReActループ、プロンプトの組み立て、他Agentへの委譲を担う。

ReflexAgentはAgentを継承し、記憶に関わるhookを空にするだけ。
使い分けと各フィールドの意味はREADMEを参照。

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
               （呼び出し側がtoolとagentを区別せずに扱える。invokable.py参照）
"""

import copy
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import uuid4

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

    配列を構造的に強制できないプロバイダでは、前後に文章が付いたり
    オブジェクトで包まれたりするため、ここで吸収する。
    取れなければNone（呼び出し側がリトライで直させる）。
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


class GenerationAborted(Exception):
    """
    before_generateが生成を中止した。

    例外にしているのは、空の応答を返すとループが同じ場所を回り続け、
    理由も見えないままmax_stepsを消費するため。
    委譲先ではexecuteがToolResult(error=...)へ変え、respondでは外へ出る。
    """


@dataclass
class RunInput:
    """
    1回の起動（respond() / execute() 1回）で受け取った入力。

    ループの各メソッドが必要とするため、引数で引き回さずAgentが
    current_inputとして保持する。次の起動時に_begin_run()が差し替える。
    """

    # プロンプトの【依頼内容】へ載せる本文。
    message: str

    # 呼び出し元から受け取った構造化引数（input_schemaに従う形）。
    # 初期ツールへ渡す引数は、ここから名前が一致するものだけを使う。
    kwargs: dict = field(default_factory=dict)
    # 過去のチャット履歴。[{"role": "user"|"assistant", "content": str}, ...]
    # respond経由だけが受け取る（委譲では依頼だけが渡り、経緯は渡らない）。
    history: list[dict] = field(default_factory=list)
    # 添付ファイル。[{"data": bytes, "mime_type": str}, ...]
    # 画像やPDFをLLMへ渡す場合に使う。
    blobs: list[dict] = field(default_factory=list)


@dataclass
class AgentResponse:
    """respond()の戻り値。"""

    text: str
    steps: int = 0  # 実際に回ったループ回数

    # このAgent自身が消費したトークン。委譲先が消費した分は含まない
    # （どのAgentが重いかを個別に測るため）。合計はNetwork.total_tokens()。
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MemoryDiffEvent:
    """
    記憶の差分を1回の生成につき1つ、構造のまま観測側へ渡す。

    材料・出力・結果を1つのイベントに揃えてあり、失敗した時も発火する
    （検証で見たいのはむしろ失敗した回）。各項目の意味はREADMEを参照。
    """

    agent: str  # どのAgentが出したか
    phase: Phase | None  # どのphaseの生成か。Noneはtoolの直接書き込み
    attempt: int  # 何回目の生成か（1始まり。再生成の経過が見える）
    rows: list  # 差分そのもの。textを含む
    errors: list  # 適用に失敗した行。差し戻して再生成させる

    # 反映しなかったが差し戻さない行（システム所有プロパティへの書き込み等）。
    # errorsと違い再生成では直らないため、次のステップで1回だけ伝える。
    ignored: list = field(default_factory=list)
    tool_results: str = ""  # この差分を作らせた材料
    raw_text: str = ""  # 配列として解釈できなかった場合の生出力

    # memoryを返したtoolが直接書き込んだ場合、そのtool名。
    # LLMが作った差分とtoolが返した値は検証の意味が違うため区別する。
    source_tool: str = ""

    # ReActループの何周目で起きたか（1始まり）。1周は
    # 「FUNCTION_CALL → 実行 → MEMORY_UPDATE」で、INITIAL_MEMORYはstep 1に属する。
    step_no: int = 0

    # 記憶へ何番目に適用されたか（apply_diffがロックの中で採番した番号）。
    # 通知はロックの外なので、届く順序は適用順ではない。後から再適用するなら
    # 受け取った順ではなくこの昇順で並べ直す。
    apply_seq: int = 0

    # この差分が、どの起動（respond / execute 1回）の中で作られたか。
    # GenerationEvent.run_id と同じ定義（適用の順序はapply_seqが持つ）。
    run_id: str = ""

    @property
    def applied(self) -> bool:
        """全行が反映されたか。errorsの有無から導出する（ToolResult.successと同じ考え方）。"""
        return not self.errors


@dataclass
class GenerationEvent:
    """
    生成1回分の実測値。時間とトークンを構造のまま観測側へ渡す。
    各項目の意味はREADMEを参照。
    """

    agent: str  # どのAgentの生成か
    phase: Phase  # どのphaseの生成か
    model: str  # 実際に使われたモデル名（phase_overridesの結果が見える）
    elapsed: float  # この試行にかかった秒数
    input_tokens: int  # キャッシュから読まれた分を含む合計
    output_tokens: int

    # input_tokensのうちキャッシュから読まれた分。プロンプトキャッシュが
    # 実際に効いているかは、ここを見ないと分からない（合計だけでは判断できない）。
    cached_tokens: int = 0
    # キャッシュへ書き込んだ分。Claudeのみ報告される。
    # 毎回これが立つなら、キャッシュが当たっておらず割高になっている。
    cache_write_tokens: int = 0

    # 何回目の試行か（1始まり）。再試行が起きた場合、同じphaseから
    # このイベントが複数回流れる。
    attempt: int = 1
    # 失敗した試行では例外の内容が入る。成功した試行では空文字。
    # 失敗した回も流す（落とすと再試行に費やした時間が計測から消える）。
    # 再試行の待ち時間は含まない（待つのはgeneration_failedのコールバック側）。
    error: str = ""

    # ReActループの何周目の生成か（1始まり）。MemoryDiffEventと同じ定義。
    step_no: int = 0

    # この生成が、どの起動（respond / execute 1回）の中で走ったか。
    # agentは名前なので、同じエージェントが並列に動くと区別が付かない。
    # ExecuteEvent.callee_run_idと同じ値で結ばれ、呼び出しの木が辿れる。
    run_id: str = ""

    # なぜ生成が終わったか。"end" / "tool_use" / "max_tokens" / "refusal" 等。
    # 例外で失敗した試行では空文字（応答自体が無い）。
    #
    # max_tokens と refusal はHTTP 200で返るためerrorには現れない。ここを見ないと
    # 「同じphaseの生成が何度も並んでいる」理由が内訳から読めない。
    stop_reason: str = ""


@dataclass
class ExecuteEvent:
    """
    tool / agentの実行1回分。各項目の意味はREADMEを参照。

    発火するのは呼び出した側（_run_plan）。
    Tool.execute / Agent.executeは呼び出し元も周回数も知らない。
    """

    caller: str  # 呼び出した側のAgent名
    callee: str  # 呼ばれたtool / agent名
    call_type: str  # "tool" | "agent"。解決できなかった名前は空文字
    kwargs: dict  # 渡した引数
    step_no: int  # ReActループの何周目か（1始まり）

    # 戻り値そのもの。整形は観測側が決める（MemoryDiffEvent.rowsと同じ扱い）。
    # 失敗・拒否された場合はNone。
    value: Any = None
    # 失敗した理由、または拒否の理由。成功した場合は空文字。
    # before_executeによる拒否とtool自身の失敗はどちらもここへ入るので、
    # 拒否だけを見たいならexecute_blockedを使う。
    error: str = ""
    # 呼び出しにかかった秒数。before_executeの判定時間も含む
    # （人の承認を待つ構成では、その待ち時間がここへ乗る）。
    elapsed: float = 0.0
    # memoryを返したtoolで、実際に記憶へ書き込めたか。
    written_to_memory: bool = False

    # 呼び出した側の起動ID（caller の current_run_id）。
    caller_run_id: str = ""
    # 呼ばれた側の起動ID。calleeがagentの場合だけ入り、toolでは空文字。
    # callee（名前）だけでは同じエージェントが並列に呼ばれた時に区別できない。
    # callee側のGenerationEvent.run_idと一致する。
    callee_run_id: str = ""

    @property
    def success(self) -> bool:
        """成功したか。errorの有無から導出する（ToolResult.successと同じ考え方）。"""
        return not self.error


@dataclass
class GenerationConfig:
    """
    phaseごとの生成設定の「上書き指示」。Noneは「指定なし＝既定値を使う」。

        phase_overrides={Phase.ANSWER: GenerationConfig(model="gemini-3.5-pro")}

    確定値はResolvedGenerationが持つ。
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

    GenerationConfigと項目は同じだが、必須にすることで「解決済み」を型で
    宣言している（生成へ渡す時点でNoneを考えなくて済む）。

    temperatureとthought_levelのNoneだけは「そのパラメータを送らない」
    という有効な値。
    """

    model: str
    llm: BaseLLM
    temperature: float | None
    max_tokens: int
    thought_level: ThoughtLevel | None


@dataclass
class ToolHistoryEntry:
    """
    ReActループ内で1回実行したtool/sub_agentの記録。

    結果本文は常に完全な形で保持し、隠すのはrender()側の責務。
    sub_agentの結果も、回答文をvalueに入れたToolResultとして記録する。
    """

    name: str  # 実行したtool名またはagent名
    kwargs: dict  # 渡した引数
    result: ToolResult  # 実行結果（成功/失敗と戻り値）

    # 呼ばれた側が持つ「この結果をどう解釈すべきか」の指示。
    # 結果を隠す時は一緒に隠す（要約済みなら解釈指示も役目を終えている）。
    evaluation: str = ""

    # 呼ばれた側のoutput_schema。値の意味を結果に添えるために持つ。
    # 結果を評価するphaseには軽い一覧しか載らないため、これが無いと
    # 「status=PARTIALが返った」ことは見えても意味が見えない。
    output_schema: dict | None = None

    # toolが返したmemoryによって、結果が既に記憶へ記録済みかどうか。
    # 記録済みなら履歴側では中身を繰り返さない（同じ内容を二重に載せない）。
    written_to_memory: bool = False

    def render(self, *, include_result: bool = True) -> str:
        """
        1件分をプロンプト用のテキストにする。

        成功時は戻り値だけを載せる（success:true を見せると「動いた」と
        「求めた答えが得られた」を混同する）。異常時だけ status: error を出す。

        include_result: Falseなら戻り値を載せず、実行した事実だけを残す
            （要約済みの結果を再度載せない）。エラーは常に見せる。
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


@dataclass
class _PlannedCall:
    """
    実行の1件分。「実行するか」を決めた後、「実行して反映する」側へ渡す器。

    decidedが入っているものは実行しない。結果を先に決めてあり、②を飛ばして
    ③でそのまま履歴とイベントへ乗る（実行しなかった事実も観測できる）。
    """

    call: FunctionCall
    # 呼び出す相手。解決できなかった場合と、実行しないと決めた場合はNone。
    target: Invokable | None = None
    # 実行せずに確定させた結果。重複拒否・名前の解決失敗・引数不足。
    decided: ToolResult | None = None
    # 重複として拒否したもの。結果を既に持っている扱いなので、次のステップでも
    # 「直前に成功した」側へ引き継ぐ（落とすと拒否と実行が交互に繰り返される）。
    duplicate: bool = False


def _is_blank(value: Any) -> bool:
    """
    必須の引数が実質的に空か。

    input_schemaのrequiredを満たしていても中身は空でありうる。LLMが値を
    埋められず""を入れる場合と、respond()経由でそのキー自体が存在しない場合
    （あちらのkwargsはmessageだけ）が該当する。

    真偽値では判定しない。0 / False / [] は正当な値であり、
    「件数0を指定する」のような呼び出しが空扱いになる。
    """
    return value is None or (isinstance(value, str) and not value.strip())


# 委譲される依頼を「自然言語のメッセージ1つ」で受け取るのが大半なので既定にする。
# 構造化された引数が必要なAgentだけ明示的に上書きする。
DEFAULT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
}


@dataclass(repr=False)
class Agent:
    # tool実行結果をプロンプトに載せるか。このクラスは結果をmemoryへ要約する
    # ため隠す（ReflexAgentはTrueで上書き）。設定値ではないのでClassVar。
    _show_tool_results: ClassVar[bool] = False

    # 最終回答をループの中で出すか。Falseなら_final_answerでoutput_schemaを
    # response_schemaとして強制できる（ReflexAgentは強制できない）。
    _answers_in_loop: ClassVar[bool] = False

    # ---- 必須項目（デフォルト無し） ----
    name: str

    # 委譲先候補としてLLMに提示する概要。「何に詳しいか」ではなく
    # 「何を返すか」を書く（前者だと質問を投げる先として選ばれる）。
    summary: str

    # llmは接続、modelはモデル名。同じ接続を複数のAgentで使い回し、
    # モデルだけAgentごとに変えられる。
    llm: BaseLLM
    model: str

    # ---- Networkが配線する項目 ----
    # Networkが1つの実体を全Agentへ注入する（既定値は単体実行のため）。
    shared_memory: SharedMemory = field(default_factory=SharedMemory)
    # 委譲先の宣言。相互参照があるため実体では書けず、Networkが解決する。
    sub_agent_names: list[str] = field(default_factory=list)

    # ---- 他のエージェントに見せるもの（部下に持つ側のプロンプトに載る） ----
    # 委譲すると決まった時だけ渡す詳細。スキーマで表現できない制約を書く。
    usage: str = ""

    # 委譲時に受け取れる引数の形（JSON Schema）。deepcopyなのは、浅いコピーでは
    # ネストした"properties"が全インスタンスで共有されるため。
    input_schema: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_INPUT_SCHEMA))

    # 最終回答に強制する形式。Noneなら自然言語で回答する。
    output_schema: dict | None = None

    # Trueの間は委譲候補として提示しない。
    disabled: bool = False

    # このAgentの回答を受け取った側が、それをどう解釈すべきかの指示。
    evaluation: str = ""

    # ---- このAgent自身の振る舞い ----
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
    # Noneならtemperature自体を送らず、モデルの既定に任せる
    # （thought_levelと同じ扱い）。既定がNoneなのは、受け付けないモデルが
    # あるため（OpenAIの推論モデル等は値に関わらず400）。
    temperature: float | None = None
    max_tokens: int = 8192
    # どれだけ考えさせるか。Noneならパラメータ自体を送らず、モデルの動的思考に
    # 任せる。明示的にOFFにしたい場合だけThoughtLevel.NONEを指定する。
    thought_level: ThoughtLevel | None = None
    # phaseごとの生成設定の上書き。指定しなかったphaseは上の既定値を使う。
    #   phase_overrides={Phase.INITIAL_MEMORY: GenerationConfig(llm=flash_lite)}
    phase_overrides: dict[Phase, GenerationConfig] = field(default_factory=dict)
    # ループ開始前に機械的に1回だけ実行するtool / agent（判断させないので生成が
    # 1回分減る）。引数はinput_schemaの同名プロパティから渡される。
    #
    # 互いに依存できない——引数はinput_schemaからしか来ないので、ある要素の
    # 結果が別の要素の引数になる経路が無い。そのため並列で実行してよく、
    # 順序に意味は無い（プロンプトへ載る順序だけはこの並び順になる）。
    #
    # tools / sub_agentsへ登録するかは任意。登録しなければカタログにも候補にも
    # 出ないので、LLMからは呼べない準備ステップになる。登録すれば後から呼べる。
    #
    # agentを置く場合、この時点では記憶がまだ作られていない（INITIAL_MEMORYは
    # この後）。届いた依頼だけを見て仕事が完結するものに限る。
    initial_tools: list[Invokable] = field(default_factory=list)

    # 初期memory構築の時だけ、材料としてpromptへ載るナレッジ（任意）。
    # 主な用途はタスクの組み立て方のfew-shot。何を書くかはREADME参照。
    # knowledgeと分けているのは載る回数が違うため（ReflexAgentでは載らない）。
    initial_knowledge: str = ""

    # 委譲された時に観測側へ流す文言。Tool.execution_messageと同じ扱い。
    #   execution_message="司書が調べています..."
    execution_message: str = ""
    # 条件で出し分けたい場合だけ関数を渡す。execution_messageより優先される。
    describe_execution: Callable[[dict], str] | None = None
    # 未登録でも何もしないだけなので、常に実体を持たせる
    # （呼び出し側で if self.interceptor: を書かなくて済む）。
    interceptor: Interceptor = field(default_factory=Interceptor)

    # ---- 実行中に変化する状態 ----
    tool_history: list[ToolHistoryEntry] = field(default_factory=list)
    # respond() / execute() が受け取った入力。ループ中の各メソッドがここから読む。
    current_input: RunInput | None = None
    # Networkが最初に配線した時点のTool一覧。2回目以降の配線で
    # ミューテート済みのコピーを複製元にしないために保持する。
    _source_tools: list | None = None
    # 最初に配線した時点のsub_agents（実体で直接指定された分）。Networkは
    # 解決した実体をsub_agentsへ書き込むため、こちらが無いと2回目の構築で
    # 名前と実体の併用を誤判定する。
    _source_sub_agents: list | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # total_input_tokensのうち、キャッシュから読まれた分の累計。
    total_cached_tokens: int = 0
    # 直前のステップで成功した呼び出しの署名。同一呼び出しの再実行を弾くために使う。
    last_successful_calls: set = field(default_factory=set)
    # 直近の_select_invokablesで解決できなかった (名前, 理由) の一覧。
    _unresolved_targets: list = field(default_factory=list)
    # このAgentがセッション中に無効化した対象の名前。起動ごとに戻す。
    _runtime_disabled: set = field(default_factory=set)
    # 直前のステップの指紋と、同じ指紋が続いた回数。停滞の検出に使う。
    _last_fingerprint: frozenset | None = None
    _stalled_count: int = 0
    # 前回のmemory更新で反映されなかった行（差し戻しても直らないもの、
    # リトライを使い切ったもの）。次のmemory更新で1回だけ提示して空になる。
    # 記憶へ書かないのは、直せば消えるべき情報が記憶には残り続けるため。
    _unapplied_rows: list = field(default_factory=list)
    # ReActループの何周目か。イベントへ載せるためだけに保持する。
    # ループの前に走るINITIAL_MEMORYもstep 1として扱うため1で始める。
    _step_no: int = 1

    # この起動（respond / execute 1回）を指すID。_begin_run()で発行する。
    # publicなのは、ExecuteEvent.callee_run_idを呼び出した側がここから読むため。
    current_run_id: str = ""

    # 同じ実体へ同時に入るのを防ぐ錠。execute()だけが取る（respond()の同時
    # 呼び出しは利用側が作るもので、守るとshared_memoryが混ざったまま動く）。
    # 1つの親からの同時呼び出しは_execute_callsがid(target)で防ぐので、
    # こちらは親をまたいだ場合に効く。RLockは万一の再入でハングしないため。
    _lock: Any = field(default_factory=threading.RLock, compare=False)
    # toolが返した添付の蓄積。文字起こしする先を持たないReflexAgentが
    # 以降も見せ続けるために使う（記憶を持つAgentは1回渡して終わり）。
    _carried_blobs: list = field(default_factory=list)

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

    def to_catalog_line(self, *, include_usage: bool = False) -> str:
        """Tool.to_catalog_line()と同じ形。委譲先候補として存在を伝える行。"""
        line = f"- {self.name}: {self.summary}"
        if include_usage and self.usage:
            line += f"\n  使い方: {self.usage}"
        return line

    def resolve_generation(self, phase: Phase) -> ResolvedGeneration:
        """
        そのphaseで使う生成設定を確定させる。指定のある項目だけを差し替える。
        ここが「指定なしのNone」が消える唯一の場所。

        `o.temperature or self.temperature` と書くと temperature=0 が偽と
        判定されて既定値に差し替わるため、必ず is None で判定する。
        """
        o = self.phase_overrides.get(phase) or GenerationConfig()
        return ResolvedGeneration(
            model=o.model if o.model is not None else self.model,
            llm=o.llm if o.llm is not None else self.llm,
            temperature=o.temperature if o.temperature is not None else self.temperature,
            max_tokens=o.max_tokens if o.max_tokens is not None else self.max_tokens,
            thought_level=o.thought_level if o.thought_level is not None else self.thought_level,
        )

    def to_declaration(self) -> dict:
        """
        このAgentを部下に持つ側へ見せる宣言情報。Tool.to_declaration()と同じ形。

        descriptionは summary + usage + 配下の能力一覧（1段だけ）。
        孫以降は、トークンが増える割に選定の判断材料にならないため見せない。
        """
        parts = [self.summary]
        if self.usage:
            parts.append(self.usage)

        catalog = self._render_capability_catalog()
        if catalog:
            parts.append("このエージェントが使用できるtool / agent:\n" + catalog)

        # 項目の名前だけを見せる。委譲先を選ぶ段階で必要なのは「何が返るか」の
        # 形だけで、値の意味は結果を受け取った後にToolHistoryEntryが添える。
        # スキーマ全体だとdescriptionまで入り、委譲先の全員分が毎ステップ載る。
        if props := (self.output_schema or {}).get("properties"):
            parts.append("このエージェントは次の項目で回答を返す: " + ", ".join(props))

        return {
            "name": self.name,
            "description": "\n".join(parts),
            "parameters": self.input_schema,
        }

    def reset(self) -> None:
        """
        次のリクエストへ持ち込む痕跡が無い状態へ戻す。

        _begin_run()は同じリクエスト内の再呼び出し用でgoalsと累計を残すが、
        こちらはリクエストの区切りなので残すものが無い。
        自動では走らない（区切りを知っているのは利用側だけ）。
        全員へ配るならNetwork.reset()。
        """
        # 空にするのではなく作り直す（項目が増えた時の消し忘れを防ぐ）。
        self.private_memory = type(self.private_memory)()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0

        # このAgentが無効化した対象を戻す。_runtime_disabledを空にするだけでは
        # toolの実体に立てたdisabledが残る。
        self._restore_disabled()

        # 起動ごとの状態も消す。_begin_run()と重複するが、reset()の後に
        # 前のリクエストの値が読める状態を残さないため。
        self.current_input = None
        self.tool_history = []
        self.last_successful_calls = set()
        self._unresolved_targets = []
        self._last_fingerprint = None
        self._stalled_count = 0
        self._unapplied_rows = []
        self._step_no = 1
        self._carried_blobs = []
        self.current_run_id = ""

    # ---- 起動の入り口 ----
    def _begin_run(self, run_input: RunInput) -> None:
        """
        起動ごとに、前回の実行の痕跡を捨てる（同一セッション内で同じAgentが
        複数回呼ばれることがある）。

        tasksを捨てるのは必須。status=nextのまま残ると_select_invokablesが
        拾い、前回の依頼のためのtool実行が起きる。goalsとshared_memoryは残す。
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
        # 前回の依頼で反映できなかった行を、次の依頼のプロンプトへ持ち込まない。
        self._unapplied_rows = []
        # 前回の周回数を持ち越さない。INITIAL_MEMORYはループへ入る前に
        # 走るため、ここで1に戻しておく必要がある。
        self._step_no = 1
        # 起動の単位は_begin_run()の呼び出しと一致するのでここで発行する。
        # 用途は1セッション内のイベントの突き合わせだけなので、人が読める
        # 長さまで切っている（外部へ出す識別子ではない）。
        self.current_run_id = uuid4().hex[:12]
        # 前回の依頼で取得した添付を持ち越さない。
        self._carried_blobs = []

    def respond(
        self,
        *,
        message: str,
        history: list[dict] | None = None,
        blobs: list[dict] | None = None,
    ) -> AgentResponse:
        """
        人間からの入り口。原文・チャット履歴・添付を受け取る。

        ここではshared_memoryへ書き込まない。原文を共有記憶へ持ち込まないことが
        プロンプトインジェクションへの境界になっており、後続へ渡るのは
        構造化された依頼だけになる。
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
    def execute(self, *, kwargs: dict, interceptor: Interceptor | None = None) -> ToolResult:
        """
        他のエージェントからの入り口。Tool.execute()と同じ形で呼べる。

        チャット履歴は受け取らない（委譲では依頼だけが渡り、経緯は渡らない）。
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
            # 直列化できない値はdefault=strで文字列化する。ここで例外を出すと
            # Invokableの「例外を外へ出さない」契約が破れる。
            message = json.dumps(kwargs, ensure_ascii=False, indent=2, default=str)

        # ここから先は自分の状態を書き換えながら進むため1件ずつに直列化する。
        # 門より後で取るのは、before_executeが人の承認を待つ構成でも
        # 他の親を待たせないため。
        with self._lock:
            self._begin_run(RunInput(message=message, kwargs=dict(kwargs)))

            # 最初に呼ばれたエージェントの依頼だけがセッション全体の方向を定義する
            # （以降の委譲では上書きしない）。入るのは構造化された引数だけ。
            if self.shared_memory.requests is None:
                self.shared_memory.requests = MemoryEntry(id="request_1", text=message)

            # toolと同じイベントで通知する。文言の組み立ては失敗しうるため、
            # 通知の失敗で委譲を止めないよう保護する（Tool.executeと同じ）。
            try:
                describe = self.describe_execution or self._default_describe_execution
                self.interceptor.notify.execute_start(describe(kwargs))
            except Exception as e:  # noqa: BLE001
                self.interceptor.notify.error(f"{self.name}の実行中メッセージの生成に失敗しました: {e}")

            try:
                text, _ = self._run_react()
            except Exception as e:  # noqa: BLE001
                # Invokableの契約として例外を外へ出さない。
                # ただし握りつぶすと原因が追えなくなるため、観測者へは通知する。
                self.interceptor.notify.error(f"{self.name}の実行が失敗しました: {e}")
                self.interceptor.notify.execute_end(f"{self.name}の委譲がエラーで終了しました。")
                return ToolResult(error=f"{self.name}の実行中にエラーが発生しました: {e}")

            # agent_answersはシステム所有。LLMの差分では書けないため直接記録する。
            self.shared_memory.agent_answers.append(
                MemoryEntry(
                    id=f"{self.name}-answer-{len(self.shared_memory.agent_answers) + 1}", text=text
                )
            )
            self.interceptor.notify.execute_end(f"{self.name}の委譲が完了しました。")

            return ToolResult(value=text)

    # ---- ReActループ ----
    def _run_react(self) -> tuple[str, int]:
        """
        ReActループの本体。「実行 → 評価 → 次を決める」を繰り返す。
        AgentとReflexAgentの違いはhookの中身だけなので、ここに分岐は無い。

        抜ける条件は5つ。どの終わり方でも必ず回答を返す。

            ・実行すべき対象が無くなった      → ANSWER
            ・同じ実行を繰り返して進まない    → STALLED_ANSWER
            ・max_stepsに達した               → CUTOFF_ANSWER
            ・ReflexAgentがテキストで答えた   → その場で返す
            ・呼び出しを強制したのにテキストが返った → ANSWER（やり直し）

        戻り値: (最終回答のテキスト, 実際に回ったステップ数)
        """
        # 記憶を持つAgentなら、ここで初期ツールを実行して最初の記憶を作る。
        # ReflexAgentでは何も起きない。
        self._before_loop()

        for step in range(1, self.max_steps + 1):
            # この周のイベントに載る番号。_generate / _update_memory /
            # _execute_calls が self._step_no から読む。
            self._step_no = step

            # 記憶を持つAgentはtasksの対象だけを提示して呼び出しを強制し、
            # ReflexAgentは全ツールを提示して呼ぶかどうかも委ねる。
            targets, tool_choice = self._select_invokables()

            # 名前が1つも解決できなかった場合。「実行すべきものが無い」のではなく
            # タスクの記述が誤っているので、失敗として記録しmemory更新で直させる
            # （素通りさせると未実行のタスクを残したまま回答へ飛ぶ）。
            if not targets and self._unresolved_targets:
                entries = [
                    ToolHistoryEntry(name=name, kwargs={}, result=ToolResult(error=reason))
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
                return self._final_answer(Phase.ANSWER, notice=self._pending_task_notice()), step

            # 呼び出しフェーズ。ここでLLMに決めさせるのは「引数」だけで、
            # 何を呼ぶかは既にtargetsで絞られている。
            response = self._generate(Phase.FUNCTION_CALL, tools=targets, tool_choice=tool_choice)

            # --- 呼び出し要求が返ってこなかった場合の3つの分岐 ---
            if not response.has_calls:
                if not response.text:
                    # テキストも呼び出し要求も無い異常な応答。
                    # ステップを1つ消費して、次の周で再試行する。
                    continue

                if self._answers_in_loop:
                    # ReflexAgentはテキストで答えてきた時点で終了する
                    # （「調べる必要はなかった」という判断の表明）。output_schemaが
                    # あっても形式はsystem_instructionで伝えてあるので作り直さない。
                    return response.text, step

                # 呼び出しを強制した（"any"）のに答えが返ってきた場合。この
                # テキストには回答フェーズの指示も出力形式の強制も効いていないため、
                # 改めて回答フェーズで答えさせる。
                return self._final_answer(Phase.ANSWER, notice=self._pending_task_notice()), step

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
        そのステップで何が起きたかの指紋。名前・引数・エラー内容の集合。

        前ステップと一致するなら新しい情報が入っていない（成功した呼び出しは
        重複拒否で弾かれるため、一致するのは失敗か拒否の繰り返しの時だけ）。
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

        同じ指紋が続いた場合、まずmemory更新へ立て直しを促し、それでも
        変わらなければ打ち切る（進捗ゼロのまま推論を払い続けない）。
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

        そのまま回答させると、実行していないことを実行したかのように書く。
        止めずに知らせるだけなのは、残っているのが正常な場合もあるため
        （誤りかどうかは記憶を読まないと決まらない）。
        """
        unfinished = (TaskStatus.NEXT, TaskStatus.NEXT_PARALLEL, TaskStatus.CONDITIONAL)
        pending = [t for t in self.private_memory.tasks if t.status in unfinished]
        if not pending:
            return ""

        listing = "\n".join(f"- [{t.id}] ({t.status.value}) {t.text}" for t in pending)
        return section(
            "実行されていないタスク",
            f"""次のタスクは完了しておらず、実行もされていない。
{listing}

これらが実行済みであるかのように回答してはならない。実行すべきだったものが実行できていない場合は、その事実と、何が足りなかったのかを回答に明示する。""",
        )

    def _final_answer(self, phase: Phase, notice: str = "") -> str:
        """
        toolを渡さずに最終回答を生成する。

        output_schemaはresponse_schemaとして渡すが、強制しきれないプロバイダ
        があるため、生成後に解釈できるかを確かめて1度だけ作り直す。

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
                        """指定された形式のJSONとして解釈できなかった。説明文やコードブロックの記号を付けず、JSONのみを出力すること。""",
                    ),
                ),
            )

        if response.text:
            return response.text
        return "【システム通知】回答の生成に失敗しました。memoryに残っている調査結果を確認してください。"

    def _matches_output_schema(self, text: str) -> bool:
        """
        回答がoutput_schemaの形として解釈できるか。

        全項目は検証しない。防ぎたいのは「JSONですらない」「オブジェクトを
        求めたのに配列」のような、パースした時点で落ちる不一致だけ。
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
    @staticmethod
    def _collect_blobs(entries: list[ToolHistoryEntry]) -> list[dict]:
        """
        今回実行したtoolが返した添付を、呼び出し順のまま集める。

        評価するphaseへ1回だけ渡すもので、起動の間ずっと有効な
        current_input.blobs（利用者の添付）とは混ぜない。
        """
        return [b for e in entries for b in e.result.blobs]

    def _before_loop(self) -> None:
        """
        ループ開始前。初期toolを実行し、その結果も材料にして初期memoryを作る。

        結果と添付は履歴経由ではなく明示的に渡す（_show_tool_results=Falseで
        履歴からは隠れるが、このステップはその中身を材料にする）。
        """
        entries = self._execute_initial_tools()
        tool_results = "\n".join(e.render(include_result=True) for e in entries)
        self._update_memory(
            phase=Phase.INITIAL_MEMORY,
            tool_results=tool_results,
            tool_blobs=self._collect_blobs(entries) or None,
            knowledge=self.initial_knowledge,
        )

    def _after_tools(self, entries: list[ToolHistoryEntry], *, notice: str = "") -> None:
        """
        実行直後。今回の結果を評価してmemoryへ反映する。

        notice: 停滞を検出した時に差し込む注意文。memory更新はtasksのstatusを
            変える唯一の場所なので、立て直しはここから促す。
        """
        tool_results = "\n".join(e.render(include_result=True) for e in entries)
        self._update_memory(
            phase=Phase.MEMORY_UPDATE,
            tool_results=tool_results,
            # toolが返した添付は、それを評価するこの生成にだけ渡す
            # （記憶へ文字起こしされた後は以降のプロンプトへ載らない）。
            tool_blobs=self._collect_blobs(entries),
            notice=notice,
            executed=entries,
        )

    def _select_invokables(self) -> tuple[list[Invokable], str | None]:
        """
        このステップで提示するtool/agentと、tool_choiceを決める。

        next / next_parallel のものだけを提示し、"any"で強制する（選ばせない）。
        提示するものが無ければ空リストを返し、ループは最終回答へ向かう。
        解決できなかった名前は_unresolved_targetsへ記録する（黙って除外すると、
        nextのまま残っているのに実行対象が空になる）。
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
                    self._unresolved_targets.append((name, "存在しないtool/agentです。"))
                elif self._is_disabled(found):
                    self._unresolved_targets.append((name, "現在無効化されています。"))
                else:
                    targets.append(found)
        return targets, "any"

    # ---- 生成呼び出し ----
    def _generate(
        self,
        phase: Phase,
        *,
        tools: list | None = None,
        tool_choice: str | None = None,
        tool_results: str = "",
        tool_blobs: list[dict] | None = None,
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

        # 会話履歴はプロンプト文字列へ埋め込まずmessagesとして渡す（埋め込むと
        # モデルのマルチターンの扱いを捨てて、ただの長文として読ませることになる）。
        # プロンプトの組み立ては再試行のたびに同じ結果になるので1回だけ行う。
        prompt = self._build_prompt(tool_results=tool_results, extra=extra_prompt)
        system_instruction = self._build_system_instruction(phase)

        # 実際に使うモデル。再試行で差し替わりうるので、cfgとは別に持つ。
        model = cfg.model

        attempt = 0
        while True:
            attempt += 1

            # 試行ごとに尋ねる（1回目だけだと再試行で予算を超えても止まらない）。
            # 計測より前に置くのは、判定が待つ場合にその時間を秒数へ混ぜないため。
            gate = self.interceptor.check.before_generate(
                agent=self.name,
                phase=phase,
                model=model,
                attempt=attempt,
                step_no=self._step_no,
            )
            if not gate.proceed:
                raise GenerationAborted(f"{self.name}の{phase.value}の生成が中止されました。")
            # モデル名が返った場合はそれに差し替える。走行中の状態を見て
            # 軽いモデルへ落とす用途（予算が残り少ない等）のため。
            if gate.model:
                model = gate.model

            started = time.perf_counter()
            try:
                response = cfg.llm.generate(
                    model=model,
                    prompt=prompt,
                    messages=self.current_input.history if self.current_input else None,
                    system_instruction=system_instruction,
                    blobs=self._blobs_for(spec) + list(tool_blobs or []),
                    tools=tools,
                    tool_choice=tool_choice,
                    response_schema=response_schema,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    thought_level=cfg.thought_level,
                )
            except Exception as e:
                # 失敗した試行も計測として流す（落とすと再試行に費やした時間が
                # 消える）。判定より先に流すのは、待ち時間を秒数へ混ぜないため。
                self.interceptor.notify.generated(
                    GenerationEvent(
                        agent=self.name,
                        phase=phase,
                        model=model,
                        elapsed=time.perf_counter() - started,
                        input_tokens=0,
                        output_tokens=0,
                        attempt=attempt,
                        step_no=self._step_no,
                        error=str(e),
                        run_id=self.current_run_id,
                    )
                )

                # 再試行するかはフレームワークが決めない（どれだけ待つか・何回
                # 試すかは運用の方針で、対話UIとバッチで正解が違う）。
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
                # モデル名が返った場合はそれに差し替える（枠を使い切ったモデルは
                # 待っても同じエラーが返る）。
                if decision.model:
                    model = decision.model
                continue
            break

        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cached_tokens += response.cached_tokens

        self.interceptor.notify.generated(
            GenerationEvent(
                agent=self.name,
                phase=phase,
                model=model,
                elapsed=time.perf_counter() - started,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                cache_write_tokens=response.cache_write_tokens,
                attempt=attempt,
                step_no=self._step_no,
                run_id=self.current_run_id,
                stop_reason=response.stop_reason,
            )
        )
        return response

    # ---- memoryの更新 ----
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

        nextのまま残ると次のステップで同じ対象が選ばれ、重複拒否で弾かれる
        ぶんmax_stepsを使い切るまで無駄が続く。記憶の書き方の誤りとして
        DiffErrorで差し戻す。失敗した実行は対象にしない（再試行が正しい）。
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
        tool_blobs: list[dict] | None = None,
        notice: str = "",
        knowledge: str = "",
        executed: list[ToolHistoryEntry] | None = None,
    ) -> list[DiffError]:
        """
        LLMに記憶の差分（JSON配列）を作らせて反映する。初期構築と更新の両方が通る。

        失敗した行があればその行だけを差し戻して再生成させる（全体をやり直すと
        既に正しく書けた項目まで作り直させる）。リトライを使い切っても中断せず、
        通った分だけを残して進む（欠けた事実はerrorイベントで通知される）。
        """
        memories = self._writable_memories()
        spec = PHASE_SPECS[phase]
        schema = build_diff_schema(*memories, allow_disable=spec.allows_disable)

        # スキーマが形式を保証していても、粒度は例で示した方が安定する。
        # 「fieldに書き込み先を指定する」という構造も例で伝わる。
        base_extra = join_sections(
            section("停滞の検出", notice),
            # 前回の更新で消えた行。ここで出し切って空にする（同じ書き方を
            # 繰り返さなければ次は出ない）。
            section("前回、反映されなかった項目", self._take_unapplied_notice()),
            # この構築だけで使うナレッジ（判断の例など）。出力例より前に置くのは、
            # 「何をどう判断するか」を読んでから「どの形で書くか」を読む方が
            # 噛み合うため。
            section("この構築のためのナレッジ", knowledge),
            section("出力例", DIFF_EXAMPLE),
        )
        extra = base_extra

        errors: list[DiffError] = []
        # 試行をまたいで積む。1回目で消えた行は2回目の差し戻しには乗らない
        # （直せないため）ので、ここで持っておかないと最後の試行の分しか残らない。
        ignored_all: list[DiffError] = []
        # ループ内で必ず代入されるが（max(1, ...)により最低1回は回る）、
        # 型チェッカーはそこまで推論しないため明示的に初期化する。
        rows: list | None = None
        # 0以下だとループが一度も回らず、memoryを作らないまま「更新しました」と
        # 通知してしまう。最低1回は必ず生成する。
        for attempt in range(1, max(1, self.max_memory_retries) + 1):
            response = self._generate(
                phase,
                tool_results=tool_results,
                tool_blobs=tool_blobs,
                extra_prompt=extra,
                response_schema=schema,
            )

            rows = _extract_rows(response.text)
            if rows is None:
                errors = [
                    DiffError("(全体)", response.text[:200], "JSONの配列として解釈できません")
                ]
                ignored = []
                # 配列として読めなかった場合は適用そのものが起きていない。
                applied_seq = 0
            else:
                # 無効化の指示はmemoryへの書き込みではないので、先に取り出して処理する。
                if spec.allows_disable:
                    self._apply_disable_rows(rows)
                # LLM由来の差分はidが必須なので、idの採番は起きない。
                # apply_seq（適用順の番号）は通知へ載せるので受け取る。
                diff = apply_diff(rows, *memories, handled_disable=spec.allows_disable)
                errors = diff.errors
                ignored = diff.ignored
                applied_seq = diff.apply_seq
                # 書き込みが通ったうえで、実行済みのタスクが放置されていないかを見る。
                # apply_diffは1行ずつしか見ないので「書かれなかったこと」は拾えない。
                if executed:
                    errors.extend(self._stale_task_errors(executed))

            ignored_all.extend(ignored)

            # 適用の直後に、材料・出力・結果を揃えて流す。ループの内側なのは、
            # 再生成の「1回目に何を間違え、2回目にどう直したか」を残すため。
            self.interceptor.notify.memory_diff(
                MemoryDiffEvent(
                    agent=self.name,
                    phase=phase,
                    attempt=attempt,
                    # 観測側が書き換えても本体へ影響しないよう浅く複製する。
                    # 通知は観測のための仕組みであり、観測が対象を変えてよい理由がない。
                    rows=list(rows or []),
                    errors=list(errors),
                    # この試行で消えた分だけ。積み上げた全体はループの外で扱う。
                    ignored=list(ignored),
                    tool_results=tool_results,
                    raw_text="" if rows is not None else response.text,
                    step_no=self._step_no,
                    apply_seq=applied_seq,
                    run_id=self.current_run_id,
                )
            )

            if not errors:
                break

            # 失敗内容を添えて再生成させる。通った分は適用済みなので、やり直すのは
            # 失敗した行だけ。毎回ベースから作り直すのは、前回のextraへ積み増すと
            # 修正済みの古い失敗まで提示され続けるため。
            extra = join_sections(
                base_extra,
                section(
                    "前回の更新で失敗した項目",
                    "以下は反映されなかった。原因を確認し、修正した分のみを出力すること。\n"
                    + "\n".join(e.render() for e in errors),
                ),
            )

        # リトライを使い切って残った分と、差し戻さずに消した分。どちらも
        # このステップでは記憶に入っていないので、次の更新で1回だけ提示する。
        self._unapplied_rows = [*errors, *ignored_all]

        if ignored_all:
            detail = "\n".join(e.render() for e in ignored_all)
            self.interceptor.notify.error(
                f"{self.name}のmemory更新で{len(ignored_all)}件を反映しませんでした"
                f"（差し戻しても直らないため再生成はしていません）。\n{detail}"
            )

        if errors:
            # リトライを使い切っても中断しないため、通知しなければ
            # 記憶が欠けたまま進んだことに誰も気付けない。
            detail = "\n".join(e.render() for e in errors)
            self.interceptor.notify.error(
                f"{self.name}のmemory更新で{len(errors)}件が反映されませんでした。\n{detail}"
            )
        else:
            # 何を更新したかを添える（「更新しました」だけでは記憶が育っているか
            # 判断できない）。ここは全行が適用された時なのでrowsがそのまま適用内容。
            applied = [
                f"{r.get('field')}:{r.get('id')}"
                for r in (rows or [])
                if isinstance(r, dict) and r.get("field") != DISABLE_FIELD
            ]
            detail = f"（{' / '.join(applied)}）" if applied else "（変更なし）"
            self.interceptor.notify.memory_updated(f"{self.name}のmemoryを更新しました。{detail}")
        return errors

    def _take_unapplied_notice(self) -> str:
        """
        前回の更新で記憶へ入らなかった行を伝える文言。読み出したら空にする。

        差し戻し（"前回の更新で失敗した項目"）とは別の経路になる。あちらは
        同じ生成の中でやり直させるもので、ここは「やり直しても直らなかった／
        直せない」ものを次のステップへ1回だけ運ぶ。
        記憶へ書かないのは、直れば消えるべき情報が記憶には残り続けるため。
        """
        rows, self._unapplied_rows = self._unapplied_rows, []
        if not rows:
            return ""
        return (
            "以下は記憶に入っていない。同じ書き方を繰り返さないこと。\n"
            "内容が必要なら、書き込めるプロパティを選び直して書く。\n"
            + "\n".join(r.render() for r in rows)
        )

    def _apply_disable_rows(self, rows: list) -> None:
        """
        差分に含まれる無効化指示を適用する。

        無効化された対象は候補の選択とカタログの両方から外れ、以降のステップ
        では存在自体が見えなくなる（依頼ではなく選択肢から消す）。
        名前は記録して次の起動時に戻す（依頼にもとづく判断で、恒久設定ではない）。
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
            if self._is_disabled(target):
                continue  # 元から無効。戻す対象にもしない

            # toolは_wireがAgentごとに複製するため実体へ書き込んでよい。agentは
            # 複製されない（集計の宛先を兼ねており、複製するとトークン累計と
            # private_memoryが分かれる）ため、_runtime_disabled側で持つ。
            if not isinstance(target, Agent):
                target.disabled = True
            self._runtime_disabled.add(name)
            self.interceptor.notify.memory_updated(
                f"{self.name}が{name}を無効化しました。理由: {row.get('text') or '(記載なし)'}"
            )

    def _is_disabled(self, target: Invokable) -> bool:
        """
        その対象が、このAgentから見て無効化されているか。

            target.disabled   … 利用側の設定、およびtoolの実行時無効
            _runtime_disabled … このAgentが今回のセッション中に無効化した名前

        agentの無効化は実体へ書き込まないため、実体だけを見ると取りこぼす。
        """
        return target.disabled or target.name in self._runtime_disabled

    def _restore_disabled(self) -> None:
        """このAgentがセッション中に無効化したものを元に戻す。"""
        for name in self._runtime_disabled:
            target = self._resolve_invokable(name)
            # agentへは書き込んでいないので戻す対象もtoolだけ（Falseにすると
            # 利用側が設定した無効まで解除してしまう）。
            if target is not None and not isinstance(target, Agent):
                target.disabled = False
        self._runtime_disabled = set()

    # ---- tool / agent の呼び出し ----
    def _resolve_invokable(self, name: str) -> Invokable | None:
        """
        名前からtoolまたはsub_agentを引く（呼び出し側はどちらか知らなくてよい）。
        toolを先に探すのは同名時に単機能処理を優先するため（実質は順序の明示）。
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
        LLMが要求した呼び出しを、実行する／しないに振り分けて_run_planへ渡す。

        重複拒否と名前の解決はここでしか起きない。逐次で回すのは、並列にすると
        同じ引数の呼び出しが両方走ってしまうため。
        """
        planned = set()
        plan: list[_PlannedCall] = []

        for call in calls:
            signature = self._call_signature(call)

            # 同じ結果しか返さない呼び出しは二重に実行しない（依頼ではなく止める）。
            # ステップをまたぐ判定を成功したものに限るのは、失敗のやり直しは正当な
            # リトライで、止めると復帰できなくなるため。同じバッチの中は成否が
            # 分からないので、実行すると決めた時点で止める。
            if signature in self.last_successful_calls or signature in planned:
                plan.append(
                    _PlannedCall(
                        call=call,
                        decided=ToolResult(
                            error="""直前と完全に同じ呼び出しのため実行しませんでした。同じ結果しか得られません。すでに得られている結果をもとに、次に必要な行動を判断してください。"""
                        ),
                        duplicate=True,
                    )
                )
                continue

            planned.add(signature)

            target = self._resolve_invokable(call.name)
            if target is None:
                # 提示していないものを呼んできた場合。ここへ来る時点でLLMの逸脱か
                # tasksの記述ミスなので、事実として記録して次の判断材料にする。
                plan.append(
                    _PlannedCall(
                        call=call,
                        decided=ToolResult(
                            error=f"'{call.name}' というtool/agentは存在しません。"
                        ),
                    )
                )
                continue

            plan.append(_PlannedCall(call=call, target=target))

        return self._run_plan(plan)

    def _run_plan(self, plan: list["_PlannedCall"]) -> list[ToolHistoryEntry]:
        """
        振り分け済みの呼び出しを実行し、記憶と履歴へ反映して返す。

        LLMが要求した分（_execute_calls）と、システムが機械的に走らせる分
        （_execute_initial_tools）の両方がここを通る。結果の扱い——記憶への
        書き込み、履歴、イベント——を1箇所に持つため、どちらの経路でも
        観測される形が同じになる。

          ② 実行 … 2件以上なら並列。外部の待ち時間が重なる
          ③ 反映 … 記憶への書き込みと履歴への追加。同時に行うと片方が消える
        """
        succeeded_now = {
            self._call_signature(item.call) for item in plan if item.duplicate
        }

        # --- ② 実行（単位ごとに並列） -------------------------------------
        # 単位の中は順番に実行される。toolは1件ごとに別の単位、agentは同じ実体
        # への呼び出しを1単位へまとめる（同時に走らせると記憶が混ざる）。
        units: list[list[tuple[int, _PlannedCall]]] = []
        agent_units: dict[int, list[tuple[int, _PlannedCall]]] = {}

        for index, item in enumerate(plan):
            if item.decided is not None or item.target is None:
                continue
            if isinstance(item.target, Agent):
                unit = agent_units.get(id(item.target))
                if unit is None:
                    unit = agent_units[id(item.target)] = []
                    units.append(unit)
                unit.append((index, item))
            else:
                units.append([(index, item)])

        results: dict[int, ToolResult] = {}

        def run_unit(unit: list[tuple[int, _PlannedCall]]) -> dict[int, ToolResult]:
            return {
                index: item.target.execute(kwargs=item.call.args, interceptor=self.interceptor)
                for index, item in unit
                if item.target is not None
            }

        if len(units) > 1:
            # スレッド数は単位の数と同じ（提示するのはtasksが指定した対象だけなので
            # 件数はそちらで抑えられている）。interceptorのコールバックはInterceptor
            # 側で直列化済みだが、tool本体は同時に呼ばれる。外部の状態を書き換える
            # toolを並列で呼ばせる場合は利用側で保護する。
            with ThreadPoolExecutor(max_workers=len(units)) as pool:
                futures = {pool.submit(run_unit, unit): unit for unit in units}
                for future in as_completed(futures):
                    try:
                        results.update(future.result())
                    except Exception as e:  # noqa: BLE001
                        # Invokableは例外を投げない契約だが、スレッドの側で
                        # 何かが起きた場合にここで落とすと1件の失敗が全体を止める。
                        self.interceptor.notify.error(f"並列実行が例外で終了しました: {e}")
                        for index, _ in futures[future]:
                            results.setdefault(
                                index, ToolResult(error=f"実行中にエラーが発生しました: {e}")
                            )
        else:
            for unit in units:
                results.update(run_unit(unit))

        # --- ③ 反映（呼び出し順のまま逐次） -------------------------------
        # 並列でも②の完了順ではなくplanの順で回す。プロンプトへ載る順序が
        # 実行のタイミングで変わると、同じ依頼でも読み方が変わってしまう。
        entries = []
        for index, item in enumerate(plan):
            call, target = item.call, item.target
            result = item.decided if item.decided is not None else results[index]

            entry = ToolHistoryEntry(
                name=call.name,
                kwargs=call.args,
                result=result,
                # 結果の解釈指示と値の意味は、呼ばれた側が持つ。
                # 呼んだ側の人格定義へ書き写さない。
                evaluation=getattr(target, "evaluation", "") if target else "",
                output_schema=getattr(target, "output_schema", None) if target else None,
            )

            # memoryを返したtoolは、その内容をLLMに要約させず直接記録する。
            # ここで行うのは、評価フェーズを持たないReflexAgentでも効かせるため。
            # 登録側の設定ではなく戻り値で決まるので、同じtoolが状況に応じて
            # 記憶へ書くか選べる。
            if result.memory and result.success:
                entry.written_to_memory = self._write_tool_result_to_memory(
                    tool_name=call.name, memory=result.memory
                )

            if target is not None and result.success:
                succeeded_now.add(self._call_signature(call))

            # 呼び出し1件ごとに、引数と戻り値を構造のまま流す。execute()の中から
            # 出せないのは、callerと周回数を知っているのが呼び出した側だけだから。
            # 実行されなかったもの（存在しない名前・重複で弾いたもの）も流す。
            self.interceptor.notify.executed(
                ExecuteEvent(
                    caller=self.name,
                    callee=call.name,
                    call_type=("agent" if isinstance(target, Agent) else "tool")
                    if target is not None
                    else "",
                    kwargs=call.args,
                    step_no=self._step_no,
                    value=result.value,
                    error=result.error or "",
                    elapsed=result.execution_time,
                    written_to_memory=entry.written_to_memory,
                    caller_run_id=self.current_run_id,
                    # 実行が終わった後に読む（Agentならこの呼び出しで発行されたIDが
                    # 入る）。toolは属性を持たないので既定値の空文字で吸収する。
                    callee_run_id=getattr(target, "current_run_id", ""),
                )
            )

            self.tool_history.append(entry)
            entries.append(entry)

        # 次のステップでの「直前」はこのバッチになる。
        self.last_successful_calls = succeeded_now
        return entries

    def _write_tool_result_to_memory(self, *, tool_name: str, memory: dict) -> bool:
        """
        toolが返したmemoryを記憶へ直接書き込む。書き込めたらTrueを返す。

        受け取るのは {プロパティ名: [本文, ...]}。idはapply_diffのid_prefixで
        採番させる（toolは既存の記憶を知らない）。存在しないプロパティ名は
        通知だけ出して処理を続ける（toolの実装ミスでセッションは止めない）。
        """
        rows = [
            {"field": field_name, "text": text}
            for field_name, texts in memory.items()
            for text in texts
        ]
        if not rows:
            return False

        result = apply_diff(rows, *self._writable_memories(), id_prefix=tool_name)
        errors = result.errors

        # LLMを介さない書き込みも同じ経路で観測できるようにする（追う側が主体ごとに
        # 別の窓を見に行かずに済む）。渡すのは採番済みidを持つresult.rows。
        # toolの戻り値（value）はidを持たないため、どこへ入ったかが見えない。
        self.interceptor.notify.memory_diff(
            MemoryDiffEvent(
                agent=self.name,
                phase=None,
                attempt=1,
                rows=list(result.rows),
                errors=list(errors),
                ignored=list(result.ignored),
                source_tool=tool_name,
                step_no=self._step_no,
                apply_seq=result.apply_seq,
                run_id=self.current_run_id,
            )
        )

        # 書けなかった分はLLMへ差し戻さない。原因はtoolの実装であって、
        # LLMが書き方を変えても直らないため（利用側へ通知して知らせる）。
        unwritten = [*errors, *result.ignored]
        if unwritten:
            self.interceptor.notify.error(
                f"{tool_name}の結果のうち{len(unwritten)}件をmemoryへ書き込めませんでした。\n"
                + "\n".join(e.render() for e in unwritten)
            )

        # 1件でも通っていれば記録済みとして扱う。分母がresult.rowsなのは、渡した
        # 行数で判定すると（ignoredの行はrowsへ入らない）1行も適用されていないのに
        # 記録済みになり、記憶にも履歴にも何も残らないため。
        applied = len(result.rows) - len(errors)
        return applied > 0

    # ---- 初期tool ----
    def _execute_initial_tools(self) -> list[ToolHistoryEntry]:
        """
        ループ開始前に、システムが機械的にtoolを実行する（確定している手順を
        判断させない分、1ステップ分の推論と失敗可能性が消える）。

        反映の経路は_execute_callsと同じ（_run_plan）。記憶への書き込み、履歴、
        イベントが1箇所なので、システムが走らせた分も同じ形で観測できる。
        """
        if not self.initial_tools or not self.current_input:
            return []
        return self._run_plan(self._plan_initial_tools())

    def _plan_initial_tools(self) -> list[_PlannedCall]:
        """
        初期toolへ渡す引数を決め、揃っていないものは実行しないと決める。

        引数は受け取った構造化引数から、呼ばれる側が受け取れるものだけを抜き出す。
        受け取れるものはto_declaration()が答える（toolは関数シグネチャ由来の
        JSON Schema、agentはinput_schema）ので、どちらかを区別せずに済む。

        必須の引数が空なら実行しない。input_schemaのrequiredは「キーがある」
        しか保証しないため、値が""のまま渡って空の条件で検索してしまう
        （呼ばれた側は空文字を正当な指定と区別できない）。
        """
        kwargs = self.current_input.kwargs if self.current_input else {}
        plan: list[_PlannedCall] = []

        for target in self.initial_tools:
            schema = target.to_declaration().get("parameters") or {}
            args = {k: v for k, v in kwargs.items() if k in schema.get("properties", {})}
            call = FunctionCall(name=target.name, args=args)

            missing = [k for k in schema.get("required", []) if _is_blank(args.get(k))]
            if missing:
                # 依頼した側がその値を持っていなかったということなので、
                # このセッションの中では誰も直せない。記憶の材料としては
                # 「取得できなかった」を残し、配線の問題として通知もする。
                reason = (
                    f"必須の引数 {', '.join(missing)} が空のため実行しませんでした。"
                    f"この情報は取得できていません。"
                )
                self.interceptor.notify.error(f"{self.name}の初期ステップ '{target.name}': {reason}")
                plan.append(_PlannedCall(call=call, decided=ToolResult(error=reason)))
                continue

            plan.append(_PlannedCall(call=call, target=target))

        return plan

    # ---- プロンプト組み立て ----
    def _render_capability_catalog(self, *, include_usage: bool = False) -> str:
        """
        自分が使えるtool / agentの一覧。disabledなものは存在自体を見せない。

        include_usage=Trueは自分がtasksを立てる・評価する立場（使い方まで
        見えないと正しいtasksが作れない）。Falseは他のAgentから「孫」として
        覗かれる立場で、そこまで見せる必要がない。
        """
        lines = [
            t.to_catalog_line(include_usage=include_usage)
            for t in self.tools
            if not self._is_disabled(t)
        ]
        lines += [
            a.to_catalog_line(include_usage=include_usage)
            for a in self.sub_agents
            if not self._is_disabled(a)
        ]
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

    def _render_memory(self) -> str:
        """
        プロンプトに載せるmemoryの中身。sharedとprivateの両方。

        ガイド文は含めない（_render_memory_reading_guide /
        _render_memory_writing_guide がsystem_instruction側へ載せる）。
        交互に並べる方が遵守されやすいが、固定文のガイドを変わる中身と混ぜると
        キャッシュに乗らないため、遵守率よりトークンコストを取っている。
        """
        return join_sections(
            self.shared_memory.render(),
            self.private_memory.render(),
        )

    def _render_memory_reading_guide(self) -> str:
        """memoryの読み方。全phaseで不変なので、system_instructionの共通部へ置く。"""
        return join_sections(
            self.shared_memory.reading_guide(),
            self.private_memory.reading_guide(),
        )

    def _render_memory_writing_guide(self) -> str:
        """
        memoryの書き方。書き込みが発生するphaseだけで使う。

        書き分けの判断はsharedとprivateをまたぐため、ROUTING_GUIDEは
        両方を見せた後に1回だけ置く（片方のガイドに置くと不完全になる）。
        """
        return join_sections(
            self.shared_memory.writing_guide(),
            self.private_memory.writing_guide(),
            section("書き込み先の選び方", ROUTING_GUIDE),
            # 依頼内容はprompt側で毎ステップ渡るため、memoryへ複写させない。
            "毎ステップ依頼内容は渡されるため、依頼内容そのものをmemoryへ書く必要はない。",
        )

    def _render_input_schema(self) -> str:
        """
        自分が受け取る引数の意味。

        委譲された側には値だけがJSONで届き、宣言（to_declaration）は呼ぶ側に
        しか渡らないため、これが無いとキー名から意味を推測して読むことになる。
        既定のスキーマ（messageだけ、説明なし）の場合は何も返さない。
        """
        props = self.input_schema.get("properties", {})
        if not props:
            return ""
        if set(props) == {"message"} and not props["message"].get("description"):
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

    def _render_output_schema(self) -> str:
        """
        自分が返す最終回答の形。output_schemaを指示文へ変換する。

        必要になるのは_answers_in_loopのクラスだけ（呼び出しフェーズで答える
        ためresponse_schemaで強制できない）。守られる保証は無い。
        """
        if not self._answers_in_loop or not self.output_schema:
            return ""

        props = self.output_schema.get("properties", {})
        if not props:
            # propertiesを持たないスキーマ（ルートが配列、$refだけ等）は項目ごとに
            # 説明できないため、スキーマ自体を載せる（落とすと形式の指定が消える）。
            return join_sections(
                """最終回答の形式が指定されている。文章で最終回答を返すときは、次のJSON Schemaに従うJSONだけを出力すること。説明文やコードブロックの記号は付けない。""",
                json.dumps(self.output_schema, ensure_ascii=False, indent=2),
            )

        required = set(self.output_schema.get("required", []))
        lines = [
            """最終回答の形式が指定されている。文章で最終回答を返すときは、次の形式のJSONだけを出力すること。説明文やコードブロックの記号は付けない。""",
            "",
        ]
        for name, spec in props.items():
            mark = "必須" if name in required else "任意"
            head = f"■ {name}（{mark}"
            if spec.get("type"):
                head += f" / {spec['type']}"
            if spec.get("enum"):
                head += " / " + " | ".join(map(str, spec["enum"]))
            lines.append(head + "）")
            lines.append(spec.get("description", "（説明なし）"))
            lines.append("")
        return "\n".join(lines).rstrip()

    def __repr__(self) -> str:
        """
        dataclassが生成する__repr__を使わない（repr=False）。

        全フィールドを出すと2万字近くになり、見せない前提のもの（審査基準など）
        が混ざる。richのshow_localsや利用側のf"{agent}"で漏れるため、出さない
        ものは最初から出せないようにしておく。
        """
        state = ", disabled=True" if self.disabled else ""
        return (
            f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, sub_agents={len(self.sub_agents)}{state})"
        )

    def _build_system_instruction(self, phase: Phase) -> list[str]:
        """
        phase内で変化しない部分。llm.generate(system_instruction=...)へ渡す。
        毎ステップ変わるもの（memoryの中身 / 実行履歴 / 依頼内容）は含めない。

        キャッシュは先頭からの一致で効くので、変わる頻度で3段に分ける。
        ②と③が逆だと、phaseが変わるたびにナレッジ全体が再送になる。

          1. 記憶構成が同じエージェントの全phaseで同一
          2. エージェントごと（ナレッジ / 人格定義 / 各スキーマ / 履歴の扱い）
          3. phaseごと（memoryの書き方 / tool一覧 / 思考 / ステップの指示）

        戻り値は段ごとの配列。区切りを明示する必要があるClaudeがこの境界を
        そのまま使い、Gemini / OpenAIではllm.py側で連結される。
        詳細はREADMEの「プロンプトキャッシュ」を参照。
        """
        spec = PHASE_SPECS[phase]

        # toolsをAPIへ渡すphaseでは文章で説明すると二重になる。渡さないphase
        # （memory構築・更新）は何が使えるか分からないとタスクを立てられないため
        # 文章で見せ、usageも含める。ANSWER系はsummaryのみ。
        if spec.allows_tools:
            catalog = ""
        else:
            include_usage = phase in (Phase.INITIAL_MEMORY, Phase.MEMORY_UPDATE)
            catalog = self._render_capability_catalog(include_usage=include_usage)

        universal = join_sections(
            # ---- ① 記憶構成が同じエージェントの全phaseで同一 ----
            section("全ステップ共通の姿勢", POURING.common),
            section("memoryの読み方", self._render_memory_reading_guide()),
        )
        per_agent = join_sections(
            # ---- ② このエージェントに固有。phaseが変わっても不変 ----
            # ナレッジを先頭へ置く理由と、その条件はdocstringに書いた。
            section("ナレッジ", self.knowledge),
            section("あなたに適用された人格定義", self.system_instruction),
            section("あなたが受け取る依頼の形", self._render_input_schema()),
            section("あなたが返す最終回答の形", self._render_output_schema()),
            section("会話履歴の扱い", self._history_stance()),
        )
        per_phase = join_sections(
            # ---- ③ phaseで変わる ----
            section(
                "memoryの書き方",
                self._render_memory_writing_guide() if spec.include_memory_how_to else "",
            ),
            section("実行可能なtool / agent", catalog),
            section("思考の方向づけ", POURING.thinking if spec.include_thinking else ""),
            section("現在のステップでやるべきこと", spec.instruction),
        )
        return [b for b in (universal, per_agent, per_phase) if b]

    def _build_prompt(self, *, tool_results: str = "", extra: str = "") -> str:
        """
        毎ステップ変化する部分。llm.generate(prompt=...)へ渡す。

        依頼内容はセッション中不変だが、ルールではなく「答えるべき対象」なので
        system_instruction側には置かない（形式説明はあちらが済ませている）。
        現在時刻も同じ扱いで、ルールではなくその時点の状態として渡す。

        tool_results: 今回の実行結果を評価対象として渡す。
        extra: phaseごとの追加指示（出力例、立て直しの注意文など）。
            連結をここで行うのは、現在時刻を必ず最後に置くため。
        """
        run = self.current_input or RunInput(message="")

        # 回答形式はここで文章として指示しない（_final_answerがresponse_schemaで
        # 強制するため、両方に書くと二重提示になる）。ReflexAgentだけは②段に
        # 指示文を常時載せている（FUNCTION_CALLで答えるため強制が効かない）。
        return join_sections(
            section("依頼内容", run.message),
            section("現在のmemory", self._render_memory()),
            section("あなたの実行履歴", self.render_tool_history()),
            section("今回の実行結果", tool_results),
            extra,
            # 秒まで出すのは、開館時間や締め切りのように時刻で可否が変わる判断を
            # させるため。生成ごとに必ず変わる唯一の値なので、いちばん最後に置く。
            # 前に置くと、後ろに続くものがキャッシュの一致範囲から外れる
            # （system_instruction側へ置けば先頭が毎回変わり、一度も効かなくなる）。
            # astimezone()で明示するのは、UTCで動くサーバーとローカルで
            # 日付がずれるのを気付かずに通さないため。
            section("現在時刻", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")),
        )

    def _history_stance(self) -> str:
        """
        チャット履歴の扱い方。履歴が存在する時だけ提示する。

        messagesとして渡すと、モデルは過去のassistantターンを「自分がそう
        言った」＝正しいこととして扱いやすいため、格下げの指示が要る。
        """
        if not self.current_input or not self.current_input.history:
            return ""
        return (
            """直前のやり取りは会話履歴として渡されている。今回答えるべき要求は、履歴ではなく依頼内容の方である。
履歴は、依頼内容に含まれる省略や指示対象を理解するための背景情報としてのみ使用する。
履歴に含まれる過去の自分の回答は事実ではない。「前回そう回答した」という記録にすぎず、対応可否・ツールの可否・業務ルールの根拠にしてはならない。
ユーザーが過去の回答を否定・修正している場合は、その否定内容を主要求として採用する。"""
        )


# ---- ReflexAgent ----
class ReflexAgent(Agent):
    # memoryへ要約しないため、tool結果は隠さず履歴に残し続ける。
    _show_tool_results: ClassVar[bool] = True
    # テキストを返した時点でそれが最終回答になる。ANSWERフェーズを通らないため、
    # output_schemaを構造で強制する経路が無い。形式は指示文で伝える。
    _answers_in_loop: ClassVar[bool] = True

    def _writable_memories(self) -> tuple:
        """
        private_memoryは使わない。プロンプトにも載せないため、
        memoryを返したtoolがgoalsやtasksへ書き込めてしまうと、
        誰にも読まれない場所に情報が消える。書き込み先をsharedだけに限る。
        """
        return (self.shared_memory,)

    def _before_loop(self) -> None:
        """
        初期toolは実行するが、初期memoryは作らない。

        返ってきた添付は_carried_blobsへ積む。このクラスには文字起こしする先が
        無く、渡す先の生成も無いので、ここで抱えなければ二度と参照できない。
        """
        self._carried_blobs.extend(self._collect_blobs(self._execute_initial_tools()))

    def _after_tools(self, entries: list[ToolHistoryEntry], *, notice: str = "") -> None:
        """
        評価フェーズを持たない。結果は履歴に残るだけ。

        memory更新の仕組みが無いため、停滞時の立て直し（notice）も効かず、
        繰り返しを検出したらそのまま打ち切られる。
        toolが返した添付は、文字起こしする先が無いので蓄積して渡し続ける。
        """
        self._carried_blobs.extend(self._collect_blobs(entries))

    def _select_invokables(self) -> tuple[list[Invokable], str | None]:
        """
        tasksを持たないため、有効なtool/agentを全て提示して"auto"で選ばせる。
        古典的なReActループそのもの。テキストを返した時点でループが終わる。
        """
        # 型注釈が無いと右辺から list[Tool] と推論され、戻り値の list[Invokable]
        # と食い違う（listは不変なので、要素がInvokableを満たしていても
        # list[Tool] を list[Invokable] として扱うことは許されない）。
        targets: list[Invokable] = [t for t in self.tools if not self._is_disabled(t)]
        targets += [a for a in self.sub_agents if not self._is_disabled(a)]
        return targets, "auto"

    def _render_memory(self) -> str:
        """shared_memoryは読む。private_memoryは持たない（更新する仕組みが無い）。"""
        return self.shared_memory.render()

    def _render_memory_reading_guide(self) -> str:
        """private_memoryを持たないため、sharedの読み方だけ。"""
        return self.shared_memory.reading_guide()

    def _render_memory_writing_guide(self) -> str:
        """
        LLMの差分でmemoryを書くphase（INITIAL_MEMORY / MEMORY_UPDATE）を通らないため、
        書き方は一切載せない。toolのmemoryによる直接書き込みは
        LLMが差分を書くわけではないので、ここの指示とは無関係。
        """
        return ""

    def _blobs_for(self, spec: PhaseSpec) -> list[dict]:
        """
        添付は常に渡す（文字起こしする仕組みが無く、渡さないと二度と参照
        できない）。そのぶん添付を返すtoolを何度も呼ぶと毎生成へ積まれ続ける。
        1回読んで畳みたい場合は、記憶を持つAgentへ担当させる。
        """
        if not self.current_input:
            return list(self._carried_blobs)
        return self.current_input.blobs + self._carried_blobs


