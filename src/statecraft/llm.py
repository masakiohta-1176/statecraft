"""
LLMプロバイダ（Gemini / OpenAI / Claude）への接続を抽象化する。

このモジュールの役割は1つで、**プロバイダごとの差異をここで吸収し、
Agent側からは同じ呼び方で使えるようにする**こと。

    response = llm.generate(model="...", prompt="...", tools=[...], ...)
    → LLMResponse（text と function_calls を持つ共通の形）

プロバイダごとに違うのは、たとえば次のような点である。

    - 会話履歴の表現         Geminiは role="model"、他は role="assistant"
    - 履歴に対する厳しさ     先頭がuserでなくてもよいもの、400になるもの
    - ツール定義の形         それぞれ別のキー名・入れ子構造
    - 呼び出し要求の取り出し方  専用フィールド／JSON文字列／ブロック走査
    - 思考の強さの指定方法    段階名／トークン数／effort
    - 出力形式の強制         スキーマで強制できるもの、指示文にするしかないもの
    - 添付の渡し方           画像とPDFでブロックが分かれるもの、分かれないもの
    - 終了理由の語彙         end_turn / stop / STOP と呼び方が違う

これらを各クラスのgenerate()の中で変換し、外へは同じ形で返す。
新しいプロバイダを足す場合は、BaseLLMを継承したクラスを1つ書くだけで、
Agent側には手を入れない。

【構成】
    ThoughtLevel    どれくらい考えさせたいかの段階（プロバイダ非依存の語彙）
    LLMResponse     応答の共通形
    BaseLLM         全プロバイダが満たす契約
    GeminiLLM       以下、プロバイダごとの実装
    OpenAILLM
    ClaudeLLM
"""

import base64
import json
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .tools import Tool
from .utils import with_timing

# ---- プロバイダSDKの遅延import ----
# 各SDKはそのクラスを生成する時に読み込む。
# 先頭でimportすると、使わないプロバイダのインストールまで必須になる。
genai = None  # google.genai
types = None  # google.genai.types（メソッド内から types.X として参照する）


def _require(module_name: str, package: str):
    """
    SDKを読み込む。未インストールなら、何を入れればよいかを含めて落とす。
    ImportErrorのままだと利用側が対処を推測することになる。
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"{module_name} が見つかりません。`pip install {package}` を実行してください。"
        ) from e


def _ensure_genai() -> None:
    """google-genaiをモジュールグローバルへ読み込む。2回目以降は何もしない。"""
    global genai, types
    if types is None:
        genai = _require("google.genai", "google-genai")
        types = _require("google.genai.types", "google-genai")


# ---- 0. データ構造 (ThoughtLevel / LLMResponse / EmbeddingResponse) ----
class ThoughtLevel(IntEnum):
    """
    「どれくらい考えさせたいか」というAgent側の意図。プロバイダの語彙ではない。

    【指定した段階はそのまま送られる】
    どの段階を持つかはモデル単位で違うが、フレームワークは丸めない。
    丸める判断の材料が実態と合わない表しか無く、合わない表で丸めると
    「HIGHを指定したのにLOWで動いていた」が観測できない形で起きる。
    受け取れない段階を送った場合はAPIがエラーを返す。

    IntEnumなのは大小比較のため（「MEDIUM以上なら」と書けるように）。

    NONEは「明示的に思考させない」で、thought_level=None（引数自体を渡さず
    モデルの既定に任せる）とは別の意味になる。
    """

    NONE = 0
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4


@dataclass
class FunctionCall:
    """LLMが返したtool/agent呼び出し1件。プロバイダ差を吸収した共通の形。"""

    name: str
    args: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    LLMからの応答。プロバイダごとの差異を吸収した共通の形。

    各LLMクラス（GeminiLLM等）は、それぞれのAPIレスポンスをこの形へ
    変換して返す。呼び出し側（Agent）はプロバイダを意識せずに扱える。
    """

    # 生成された回答文。呼び出し要求だけの応答では空になる。
    text: str

    # LLMが「このツールを呼んでほしい」と要求した内容。
    # 空リストなら、ツールを呼ばずテキストで答えたということ。
    function_calls: list[FunctionCall] = field(default_factory=list)
    input_tokens: int = 0  # このリクエストで消費した入力トークン数（キャッシュ分を含む合計）
    output_tokens: int = 0  # このリクエストで生成した出力トークン数

    # input_tokensのうち、キャッシュから読まれた分。安く課金される。
    # プロンプトキャッシュが実際に効いているかは、この値でしか確認できない。
    cached_tokens: int = 0
    # キャッシュへ「書き込んだ」分。Claudeのみ報告される。
    # 書き込みは通常より高く課金されるため、毎回これが立つ（＝毎回ミスしている）
    # 状態は、キャッシュを付けたことが逆に高くついているサイン。
    cache_write_tokens: int = 0

    # 生成が終わった理由。プロバイダの語彙を共通語彙へ移したもの。
    #
    # 打ち切りと拒否はどちらも例外ではなく200の空応答として届くため、
    # これが無いと「モデルが何も返さなかった」と区別が付かない。
    stop_reason: str = ""

    execution_time: float = 0.0  # API応答にかかった時間（秒）。with_timingが自動で埋める

    @property
    def has_calls(self) -> bool:
        """
        ツールの呼び出し要求が含まれているかどうか。

        種別フィールドは持たない（function_callsが空かどうかで判定できる）。
        textもfunction_callsも両方空という応答もありうるため、
        呼び出し側は「has_callsがFalse かつ textも空」でその状態を判定する。
        """
        return bool(self.function_calls)


@dataclass
class EmbeddedChunk:
    chunk: str  # 埋め込み対象の元テキスト
    vector: list[float]  # 埋め込みベクトル


@dataclass
class EmbeddingResponse:
    items: list[EmbeddedChunk]  # chunkとvectorのペアのリスト
    input_tokens: int = 0  # 入力トークン数
    execution_time: float = 0.0  # API応答時間（秒）


# 会話履歴の共通形式。各プロバイダの形はここから変換する。
#   [{"role": "user" | "assistant", "content": "..."}, ...]
#
# プロバイダごとに要求される形が違う（Geminiはpartsキーでroleが"model"、
# OpenAI/Claudeはcontentキーでroleが"assistant"）。
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"


# 生成が終わった理由の共通語彙。プロバイダごとの呼び方をここへ移す。
#
#   ""            プロバイダが報告しなかった
#   "end"         通常終了
#   "tool_use"    呼び出し要求を出して停止した
#   "max_tokens"  出力上限で打ち切られた（回答が途中で切れている）
#   "refusal"     安全側の判断で生成されなかった
#
# 上の4つ以外は、プロバイダの生の値を小文字にして素通しする
# （知らない値を"other"へ潰すと、何が起きたのか追えなくなる）。
#
# Claude: anthropic.types.StopReason
#   end_turn / max_tokens / stop_sequence / tool_use / pause_turn / refusal /
#   model_context_window_exceeded
_CLAUDE_STOP_REASONS = {
    "end_turn": "end",
    "stop_sequence": "end",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}

# OpenAI: choices[0].finish_reason
_OPENAI_STOP_REASONS = {
    "stop": "end",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

# Gemini: candidates[0].finish_reason（google.genai.types.FinishReason）
# SAFETY / PROHIBITED_CONTENT / BLOCKLIST / SPII は refusal へ寄せる。
# RECITATION や MALFORMED_FUNCTION_CALL は性質が違うので素通しにする。
_GEMINI_STOP_REASONS = {
    "stop": "end",
    "max_tokens": "max_tokens",
    "safety": "refusal",
    "prohibited_content": "refusal",
    "blocklist": "refusal",
    "spii": "refusal",
}


def _normalize_stop_reason(raw: Any, table: dict[str, str]) -> str:
    """プロバイダの終了理由を共通語彙へ移す。未知の値は小文字にして素通しする。"""
    if not raw:
        return ""
    # Geminiは列挙型で返すため、名前を取り出してから引く。
    name = str(getattr(raw, "name", None) or raw).lower()
    return table.get(name, name)


def _normalize_messages(messages: list[dict] | None) -> list[tuple[str, str]]:
    """
    会話履歴を (role, text) のタプル列へ正規化する。

    不正な形は黙って捨てず例外にする。履歴の欠落は「なぜか過去の文脈を
    踏まえてくれない」という形で現れ、実行時には原因が見えないため。
    """
    normalized = []
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict):
            raise TypeError(f"messages[{i}]はrole/contentを持つ辞書である必要があります")

        role = m.get("role")
        # "model"（Gemini語彙）で渡されても受け付ける
        if role in ("assistant", "model"):
            role = _ROLE_ASSISTANT
        elif role == "user":
            role = _ROLE_USER
        else:
            raise ValueError(
                f"messages[{i}]のroleが不正です: {role!r}"
                f"（'user' または 'assistant' を指定してください）"
            )

        content = m.get("content")
        if not isinstance(content, str):
            raise TypeError(f"messages[{i}]のcontentは文字列で指定してください")
        if not content.strip():
            # 空の発話はClaudeが受け付けない（400）。Gemini / OpenAIでは通るため、
            # 通してしまうと「Geminiでは動いたのにClaudeで落ちる履歴」ができる。
            # ここで落とす方が、どのプロバイダでも同じ入力が同じように動く。
            raise ValueError(
                f"messages[{i}]のcontentが空です"
                "（空の発話は履歴へ入れる前に除いてください）"
            )

        normalized.append((role, content))
    return normalized


def _as_blocks(system_instruction: str | list[str] | None) -> list[str]:
    """
    system_instructionを、段ごとのブロックの配列に正規化する。

    Agentは変わる頻度で3段に分けて渡す（全エージェント共通 / エージェント固有 /
    phase固有）。プロンプトキャッシュの区切りを打てるプロバイダは、この境界を
    そのまま使う。単一の文字列で渡された場合は1段として扱う。
    """
    if not system_instruction:
        return []
    if isinstance(system_instruction, str):
        return [system_instruction]
    return [b for b in system_instruction if b]


def _as_text(system_instruction: str | list[str] | None) -> str:
    """段の区切りを持たないプロバイダ向けに、1つの文字列へ戻す。"""
    return "\n\n".join(_as_blocks(system_instruction))


def _extension_for(mime_type: str) -> str:
    """
    mime_typeから拡張子を推測する。分からなければ空文字。

    添付にファイル名を要求するプロバイダ（OpenAI）向け。blobは名前を
    持たないので、せめて種類が分かる名前を組めるようにする。
    """
    return mimetypes.guess_extension(mime_type) or ""


def _json_only_instruction(response_schema) -> str:
    """
    response_schemaに相当する仕組みを持たないプロバイダ向けに、
    スキーマを指示文へ落とし込む。

    Geminiは構造を強制できるが、OpenAIとClaudeにはルートが配列のスキーマを
    強制する手段が無い。受け取って無視すると「指定したのに効かない」状態に
    なるため、最低限プロンプトとして伝える（受け取り側は寛容にパースする）。
    """
    lines = [
        "",
        "",
        "出力は次のJSON Schemaに厳密に従うJSONのみとすること。",
        "前後に説明文やコードブロックの記号を付けてはならない。",
        json.dumps(response_schema, ensure_ascii=False),
    ]
    return "\n".join(lines)


# ---- 1. 統一規格（基底クラス） ----
class BaseLLM(ABC):
    """
    LLMプロバイダへの接続。

    モデル名は保持せず、generate()の引数として毎回受け取る（どのモデルを
    使うかはAgentとphase_overridesが決める）。
    そのためインスタンスはプロバイダの数だけあればよい。
    """

    @abstractmethod
    def generate(
        self,
        *,
        model: str,  # 使用するモデル名。Agentが指定する
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | list[str] | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str
        | None = "auto",  # 'auto' | 'any'（toolsを使わせたくない場合はtools自体を渡さない = Agentの責務）
        response_schema: Any | None = None,
        # Noneならこのパラメータ自体を送らない。受け付けないモデル
        # （OpenAIの推論モデル、Claudeの現行モデル）があるため。
        # 受け付けるかどうかは判定しない（表を持つと必ず古くなる）。
        temperature: float | None,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        pass


# ---- 2. Gemini 統合クラス（AI Studio / Vertex AI 両対応） ----
class GeminiLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str = "global",
        use_vertex: bool = False,
        # 思考の強さをトークン数で受け取るモデル（gemini-2.5系）向けに、
        # 各段階を何トークンとみなすかを渡す。
        #
        #   thinking_budgets={ThoughtLevel.LOW: 1024, ThoughtLevel.HIGH: 8192}
        #
        # これを渡すかどうかが、そのまま送り方になる。
        #   渡した   … thinking_budget（トークン数）で送る
        #   渡さない … thinking_level（段階名）で送る
        #
        # 系列名（"level" / "budget"）は受け取らない。別に持つと、モデルを
        # 別系列へ変えた時に直し忘れが実行時エラーになる。
        thinking_budgets: dict[ThoughtLevel, int] | None = None,
    ):
        _ensure_genai()
        # 各段階を何トークンとみなすかは、フレームワークではなく利用側が決める。
        # 渡されなかった段階については上限を指定せず、モデルの動的思考に任せる。
        self._thinking_budgets = thinking_budgets or {}
        if use_vertex:
            if not project_id:
                raise ValueError("Vertex AI を使用する場合は project_id が必須です。")
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            self.client = genai.Client(api_key=api_key)

    def _build_thinking_config(self, thought_level: ThoughtLevel | None):
        """
        Agentの意図（ThoughtLevel）を、この系列が受け取れる形に変換する。
        Noneを返した場合はthinking_configを送らないため、モデルの動的思考に任せる。

        【丸めない】
        指定された段階をそのまま要求する。受け取れないモデルへ送った場合は
        APIがエラーを返す（理由はThoughtLevelのdocstringにある）。
        """
        # 思考の指定が無ければ何も送らない（モデルの既定に任せる）。
        if thought_level is None:
            return None

        if not self._thinking_budgets:
            # 段階名をそのまま送る（OpenAI / Claudeと同じ扱い）。
            # 段階名で受け取らないモデルや、NONEに相当する語彙が無いモデルへ
            # 送った場合はAPIがエラーを返す。丸めないのは上に書いた理由による。
            return types.ThinkingConfig(thinking_level=thought_level.name.lower())

        # 以下はトークン数で受け取るモデル向け。数値が必要になる。
        if thought_level is ThoughtLevel.NONE:
            # 0は「思考させない」という意味そのものなので、チューニング値ではない。
            return types.ThinkingConfig(thinking_budget=0)

        budget = self._thinking_budgets.get(thought_level)
        if budget is None:
            # 何トークン割り当てるかの指定が無いので、勝手に決めずモデルに任せる。
            return None
        return types.ThinkingConfig(thinking_budget=budget)

    # 戻り値が list[Any] なのは遅延importの都合。実際に返すのは
    # list[types.Tool] だが、typesは変数なので型注釈には書けない。
    # TYPE_CHECKINGで本物をimportすると、google-genaiが無い環境で
    # 型チェックが通らなくなる。
    def _format_tools(self, tools: list[Any]) -> list[Any]:
        declarations = []
        for t in tools:
            # isinstance(t, Tool)では判定しない。AgentはToolを継承せず
            # Invokableを構造的に満たすだけなので、継承で見るとAgentが
            # 変換されないままAPIへ渡り、宣言情報が失われる。
            if hasattr(t, "to_declaration"):
                d = t.to_declaration()
                declarations.append(
                    types.FunctionDeclaration(
                        name=d["name"],
                        description=d["description"],
                        parameters_json_schema=d["parameters"],
                    )
                )
            else:
                declarations.append(t)
        return [types.Tool(function_declarations=declarations)]

    @with_timing
    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | list[str] | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        # Noneならこのパラメータ自体を送らない。受け付けないモデル
        # （OpenAIの推論モデル、Claudeの現行モデル）があるため。
        # 受け付けるかどうかは判定しない（表を持つと必ず古くなる）。
        temperature: float | None,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # --- ① 会話履歴をGeminiの形へ変換する -------------------------------
        # 共通形式を Content(role=..., parts=[...]) の列へ移す。
        # Geminiはassistant側のroleを "model" と呼ぶため、そこも変換する。
        contents = [
            types.Content(
                role="model" if role == _ROLE_ASSISTANT else "user",
                parts=[types.Part.from_text(text=text)],
            )
            for role, text in _normalize_messages(messages)
        ]

        # --- ② 今回のターンを1つのuserメッセージとして足す -------------------
        # 添付と本文を同じuserターンにまとめ、履歴の後ろへ足す
        # （promptとmessagesは併用される。片方を捨ててはならない）。
        #
        # 添付の種類は絞らない。Geminiは画像・PDFに加えて音声と動画も
        # 同じinline_dataで受け取るため、種類ごとの作り分けが無い。
        #
        # inline_dataはリクエスト全体のサイズ上限（20MB程度）に収まる必要が
        # ある。長い音声や動画はFiles API（Part.from_uri）が別途必要。
        current_parts = []
        for blob in blobs or []:
            current_parts.append(
                types.Part.from_bytes(data=blob["data"], mime_type=blob["mime_type"])
            )
        if prompt:
            current_parts.append(types.Part.from_text(text=prompt))
        if current_parts:
            contents.append(types.Content(role="user", parts=current_parts))

        # --- ③ 思考の強さを、このモデル系列が受け取れる形へ変換する ----------
        thinking_config = self._build_thinking_config(thought_level)

        # --- ④ ツール定義を変換し、呼び出しを強制するかを決める --------------
        # "any"  … 必ずどれかを呼ばせる（Agentがtasksで対象を絞った時）
        # "auto" … 呼ぶか答えるかをモデルに委ねる
        # 使わせたくない場合はtools自体を渡さない（呼び出し側の責務）。
        formatted_tools = self._format_tools(tools) if tools else None
        tool_config = None
        if formatted_tools and tool_choice:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=tool_choice.upper())
            )

        # --- ⑤ リクエスト設定をまとめる --------------------------------------
        config = types.GenerateContentConfig(
            # SDKの自動関数呼び出し（AFC）を必ず切る。有効なままだと
            # SDK自身がPython関数を呼んでしまい、呼び出し手前の確認
            # （無効化／重複実行／before_execute）と計測が全部迂回される。
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=formatted_tools,
            tool_config=tool_config,
            # スキーマが指定された時だけJSONモードにする。
            # Geminiはresponse_schemaで出力構造を実際に強制できる。
            response_mime_type="application/json" if response_schema else "text/plain",
            response_schema=response_schema,
            thinking_config=thinking_config,
            system_instruction=_as_text(system_instruction),
        )

        # --- ⑥ 実際に呼ぶ ----------------------------------------------------
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # --- ⑦ トークン消費量を取り出す --------------------------------------
        # 属性が無い場合やNoneの場合があるため、素直に辿らず存在確認してから読む。
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            # Geminiはキャッシュ分をprompt_token_countの内数として報告する。
            # 暗黙キャッシュが効いていない場合や、SDKが古い場合は属性ごと無い。
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        # --- ⑧ 呼び出し要求を共通の形へ変換する ------------------------------
        calls = [
            FunctionCall(name=c.name, args=dict(c.args or {}))
            for c in (response.function_calls or [])
        ]

        # --- ⑨ テキスト部分だけを取り出す ------------------------------------
        # response.text という近道は、応答に呼び出し要求が含まれていると
        # 警告を出す。テキストのパートだけを拾えば警告なしで同じ結果になる。
        texts = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    texts.append(part.text)

        # --- ⑩ 終了理由を共通語彙へ移す --------------------------------------
        # 安全側の判断や上限での打ち切りは、例外ではなく「本文の無い応答」
        # として返る。ここを拾わないと「何も返さなかった」と区別が付かない。
        candidates = response.candidates or []
        stop_reason = _normalize_stop_reason(
            getattr(candidates[0], "finish_reason", None) if candidates else None,
            _GEMINI_STOP_REASONS,
        )

        return LLMResponse(
            text="".join(texts),
            function_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            stop_reason=stop_reason,
        )

    @with_timing
    def embed(
        self, *, texts: list[str], model: str, dimensions: int | None = None
    ) -> EmbeddingResponse:
        embed_config = None
        if dimensions:
            embed_config = types.EmbedContentConfig(output_dimensionality=dimensions)

        response = self.client.models.embed_content(
            model=model,
            contents=texts,
            config=embed_config,
        )

        # strict=True: 件数が食い違ったら例外にする。既定（短い方で打ち切る）
        # だと、一部が埋め込まれないまま処理が進み、後段で
        # 「なぜか一部だけ検索に出ない」という形でしか現れない。
        items = [
            EmbeddedChunk(chunk=chunk, vector=list(e.values))
            for chunk, e in zip(texts, response.embeddings, strict=True)
        ]

        # TODO: response にトークン使用量の情報が含まれているか未確認。
        # generate_content の usage_metadata と同じ形で入っているか、
        # 実際にレスポンスを print して確認してから input_tokens を実装する。
        return EmbeddingResponse(
            items=items,
        )


# ---- 3. OpenAI クラス ----
# mime_typeと、input_audioのformatで使う名前が食い違うものだけを並べる。
# 対応形式の一覧ではない（一覧を持つと必ず古くなる）。ここに無い音声は
# サブタイプをそのまま送り、可否はAPIに判断させる（audio/flac → "flac"）。
_OPENAI_AUDIO_FORMAT_ALIASES = {
    "audio/mpeg": "mp3",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
}


class OpenAILLM(BaseLLM):
    def __init__(self, *, api_key: str | None = None):
        OpenAI = _require("openai", "openai").OpenAI
        self.client = OpenAI(api_key=api_key)

    def _format_tools(self, tools: list[Any]) -> list[dict]:
        formatted = []
        for t in tools:
            # ToolとAgentの両方を受け取るため、継承ではなくto_declaration()の
            # 有無で判定する（AgentはToolを継承していない）。
            if hasattr(t, "to_declaration"):
                d = t.to_declaration()
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": d["name"],
                            "description": d["description"],
                            "parameters": d["parameters"],
                        },
                    }
                )
            else:
                formatted.append(t)
        return formatted

    def _format_blobs(self, blobs: list[dict]) -> list[dict]:
        """
        添付を、OpenAIのcontentパートへ変換する。

        種類ごとにパートが違う（openai.types.chat の
        ChatCompletionContentPartParam に並んでいる形に合わせる）。

            画像    {"type": "image_url",    "image_url":   {"url": "data:..."}}
            音声    {"type": "input_audio",  "input_audio": {"data": ..., "format": ...}}
            その他  {"type": "file",         "file":        {"filename": ..., "file_data": ...}}

        filenameが必要なのは、file_dataで中身を直接送る場合に要求されるため。
        blobは名前を持たないので連番で作る。

        動画はChat Completionsに該当するパートが無いため例外にする
        （fileパートへ入れても「渡したつもりで見られていない」状態になる）。
        """
        parts = []
        for i, blob in enumerate(blobs):
            mime = blob["mime_type"]
            b64_data = base64.b64encode(blob["data"]).decode("utf-8")
            if mime.startswith("image/"):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                    }
                )
            elif mime.startswith("audio/"):
                # formatは形式の名前で渡す。mime_typeと名前が違うものだけ
                # 読み替え、それ以外はサブタイプをそのまま送る。
                # 受け付けるかどうかはAPIが判断する（対応形式の一覧は持たない）。
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_data,
                            "format": _OPENAI_AUDIO_FORMAT_ALIASES.get(
                                mime, mime.removeprefix("audio/")
                            ),
                        },
                    }
                )
            elif mime.startswith("video/"):
                raise ValueError(
                    f"blobs[{i}]のmime_type {mime!r} はOpenAIへ渡せません。"
                    "Chat Completionsに動画のcontentパートが無いためです。"
                    "動画を渡す必要がある生成は、phase_overridesでllmをGeminiへ"
                    "差し替えてください（Geminiは動画をそのまま受け取れます）。"
                )
            else:
                parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": f"attachment_{i + 1}{_extension_for(mime)}",
                            "file_data": f"data:{mime};base64,{b64_data}",
                        },
                    }
                )
        return parts

    @with_timing
    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | list[str] | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        # Noneならこのパラメータ自体を送らない。受け付けないモデル
        # （OpenAIの推論モデル、Claudeの現行モデル）があるため。
        # 受け付けるかどうかは判定しない（表を持つと必ず古くなる）。
        temperature: float | None,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # response_formatのjson_objectモードはルートをオブジェクトに強制するため使わない。
        # memory差分はルートが配列である前提（キーの捏造を防ぐための設計）なので、
        # スキーマは指示文として渡し、パース側で吸収する。
        system_text = _as_text(system_instruction)
        if response_schema:
            system_text += _json_only_instruction(response_schema)

        api_messages = []
        if system_text:
            api_messages.append({"role": "system", "content": system_text})

        # 過去の履歴 → 今回のターン、の順で組み立てる。
        api_messages.extend(
            {"role": role, "content": text} for role, text in _normalize_messages(messages)
        )

        if prompt or blobs:
            content = self._format_blobs(blobs or [])
            if prompt:
                content.append({"type": "text", "text": prompt})
            api_messages.append({"role": "user", "content": content})

        kwargs = {
            "model": model,
            "messages": api_messages,
            # max_tokensは非推奨で、推論モデル（o系）では受け付けられない。
            # reasoning_effortを送る相手はその系列なので、max_tokensのままでは
            # 思考を指定した瞬間に400になる。
            "max_completion_tokens": max_tokens,
        }
        # 指定された時だけ送る。推論モデル（o系 / gpt-5系）は受け付けないため、
        # そちらを使う場合は利用側がtemperatureを指定しない。
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = "required" if tool_choice == "any" else tool_choice

        # OpenAIはreasoning_effortという語彙で受け取る。ThoughtLevelの
        # 段階名をそのまま送る（Geminiのthinking_levelと同じ扱い）。
        # 対応表は持たない——受け付ける値がモデルごとに違うため。
        if thought_level is not None:
            kwargs["reasoning_effort"] = thought_level.name.lower()

        response = self.client.chat.completions.create(**kwargs)

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            # OpenAIもキャッシュ分をprompt_tokensの内数として報告する。
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) or 0

        message = response.choices[0].message
        # OpenAIは引数をJSON文字列で返すため、ここで辞書へ戻す。
        # 壊れたJSONが来た場合に例外で落とさず、空の引数として扱う
        # （呼び出し自体は実行され、引数不足のエラーとして結果に現れる）。
        calls = []
        for c in message.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(FunctionCall(name=c.function.name, args=args))

        return LLMResponse(
            text=message.content or "",
            function_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            stop_reason=_normalize_stop_reason(
                response.choices[0].finish_reason, _OPENAI_STOP_REASONS
            ),
        )

    @with_timing
    def embed(
        self, *, texts: list[str], model: str, dimensions: int | None = None
    ) -> EmbeddingResponse:
        kwargs = {"model": model, "input": texts}
        if dimensions:
            kwargs["dimensions"] = dimensions

        response = self.client.embeddings.create(**kwargs)

        # strict=True の理由はGeminiLLM.embedと同じ（件数の食い違いを黙って通さない）。
        items = [
            EmbeddedChunk(chunk=chunk, vector=d.embedding)
            for chunk, d in zip(texts, response.data, strict=True)
        ]

        input_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.total_tokens or 0

        return EmbeddingResponse(
            items=items,
            input_tokens=input_tokens,
        )


# ---- 4. Claude クラス ----
class ClaudeLLM(BaseLLM):
    def __init__(self, *, api_key: str | None = None):
        Anthropic = _require("anthropic", "anthropic").Anthropic
        self.client = Anthropic(api_key=api_key)

    def _format_tools(self, tools: list[Any]) -> list[dict]:
        formatted = []
        for t in tools:
            # ToolとAgentの両方を受け取るため、継承ではなくto_declaration()の
            # 有無で判定する（AgentはToolを継承していない）。
            if hasattr(t, "to_declaration"):
                d = t.to_declaration()
                formatted.append(
                    {
                        "name": d["name"],
                        "description": d["description"],
                        "input_schema": d["parameters"],
                    }
                )
            else:
                formatted.append(t)
        return formatted

    def _format_blobs(self, blobs: list[dict]) -> list[dict]:
        """
        添付を、Claudeのcontentブロックへ変換する。

        種類ごとにブロックが違う。すべてをimageで送ってはならない。

            画像    {"type": "image",    "source": {"type": "base64", ...}}
            PDF     {"type": "document", "source": {"type": "base64", ...}}
            テキスト {"type": "document", "source": {"type": "text",   ...}}

        全てimageブロックへ入れるとPDFが400になる。Geminiは種類を問わず
        受けるため、この取り違えはClaudeでだけ落ちる形で現れる。

        音声と動画は渡せない。Messages APIに該当するcontent blockが無いため、
        フレームワーク側で作れない（理由を言って落とす）。

        画像は image/* をそのまま入れ、形式の一覧は持たない（一覧を持つと
        古くなった時点で使えるはずの形式を拒否することになる）。

        扱えない種類は例外にする。黙って落とすと、添付を見ていない回答が
        できてしまう。
        """
        blocks = []
        for i, blob in enumerate(blobs):
            mime = blob["mime_type"]
            if mime.startswith("image/"):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": base64.b64encode(blob["data"]).decode("utf-8"),
                        },
                    }
                )
            elif mime == "application/pdf":
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.b64encode(blob["data"]).decode("utf-8"),
                        },
                    }
                )
            elif mime.startswith("text/"):
                # テキストのdocumentはbase64ではなく素の文字列で渡す
                # （anthropic.types.PlainTextSourceParam）。
                try:
                    text = blob["data"].decode("utf-8")
                except UnicodeDecodeError as e:
                    raise ValueError(
                        f"blobs[{i}]（{mime}）をUTF-8として読めませんでした。"
                        "テキストの添付はUTF-8で渡してください。"
                    ) from e
                blocks.append(
                    {
                        "type": "document",
                        "source": {"type": "text", "media_type": "text/plain", "data": text},
                    }
                )
            elif mime.startswith(("audio/", "video/")):
                kind = "音声" if mime.startswith("audio/") else "動画"
                raise ValueError(
                    f"blobs[{i}]のmime_type {mime!r} はClaudeへ渡せません。"
                    f"Messages APIに{kind}のcontent blockが無いためです。"
                    f"{kind}を渡す必要がある生成は、phase_overridesでllmをGeminiへ"
                    f"差し替えてください（Geminiは{kind}をそのまま受け取れます）。"
                )
            else:
                raise ValueError(
                    f"blobs[{i}]のmime_type {mime!r} はClaudeへ渡せません。"
                    "渡せるのは image/* / application/pdf / text/* です。"
                )
        return blocks

    # Claudeのキャッシュ区切りは1リクエストに4個まで。段が増えたらここで詰まる。
    # 上限はリクエスト全体で4個であり、systemだけの枠ではない
    # （toolsやmessagesへ区切りを打つ場合は、その分をここから譲る必要がある）。
    _MAX_CACHE_BREAKPOINTS = 4

    def _cached_system(self, blocks: list[str]) -> list[dict]:
        """
        system_instructionを段ごとのブロックにし、各段の末尾へキャッシュの区切りを打つ。

        Gemini / OpenAIは一致するプレフィックスを自動でキャッシュするが、
        Claudeは cache_control で明示的に区切らないとキャッシュされない。

        段の境界ごとに区切るのは、再利用できる範囲が違うため。
        1段目は全エージェント・全phase、2段目はそのエージェントの全phase、
        3段目は同じphaseの繰り返しで当たる。末尾に1つだけ打つと完全一致でしか
        当たらず、段を分けた意味が消える。

        最小トークン数（モデルにより1024/2048程度）を下回る段では区切りが
        無視される。エラーにはならず、単に効かない。

        【段を分けても効かない条件がある】
        照合は tools → system → messages の順に先頭から行われるため、
        提示するtool/agentの集合や並び順が変わると、systemの区切りが一致して
        いても①から書き直しになる。効果が出るのは、モデルとtool定義が同一の
        リクエスト同士だけ。cache_write_tokensばかり立つ場合は、まずtoolsが
        リクエストごとに動いていないかを見る。
        """
        head = list(blocks[: self._MAX_CACHE_BREAKPOINTS])
        tail = blocks[self._MAX_CACHE_BREAKPOINTS :]
        if tail:
            # 区切りの上限を超えた分は、最後の段へまとめる（区切りは打てない）。
            head[-1] = "\n\n".join([head[-1], *tail])
        return [{"type": "text", "text": b, "cache_control": {"type": "ephemeral"}} for b in head]

    @with_timing
    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | list[str] | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        # Noneならこのパラメータ自体を送らない。受け付けないモデル
        # （OpenAIの推論モデル、Claudeの現行モデル）があるため。
        # 受け付けるかどうかは判定しない（表を持つと必ず古くなる）。
        temperature: float | None,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # 過去の履歴 → 今回のターン、の順で組み立てる。
        # 値がAnyなのは、contentが文字列（履歴）とブロックのリスト
        # （添付を含む今回のターン）の両方を取るため。
        api_messages: list[dict[str, Any]] = [
            {"role": role, "content": text} for role, text in _normalize_messages(messages)
        ]

        if prompt or blobs:
            content: list[dict[str, Any]] = self._format_blobs(blobs or [])
            if prompt:
                content.append({"type": "text", "text": prompt})
            api_messages.append({"role": "user", "content": content})

        # Claudeはmessagesの形に他より厳しく、次の2つは400になる。
        # 送る前に落として理由を言う（APIのエラーは、どの入力が原因かを
        # 呼び出し側から辿りにくい）。
        if not api_messages:
            raise ValueError(
                "Claudeはmessagesが空のリクエストを受け付けません。"
                "promptか会話履歴のどちらかを渡してください。"
            )
        if api_messages[0]["role"] != _ROLE_USER:
            raise ValueError(
                "Claudeはmessagesの先頭がuserである必要があります"
                f"（先頭が{api_messages[0]['role']!r}の履歴が渡されました）。"
                "assistantの発話から始まる履歴は、先頭を除いてから渡してください。"
            )

        kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        # 指定された時だけ送る。現行モデルはサンプリング指定を受け付けない
        # （値に関わらず400になる）ため、そちらを使う場合は利用側が
        # temperatureを指定しない。どのモデルが受け付けるかは判定しない。
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Claudeにも構造化出力（output_config.format）はあるが使っていない。
        #   ・memory差分はルートが配列である前提で、formatがそれを強制できるか
        #     未確認
        #   ・formatは下のeffortと同じoutput_configに入るため、両方使うなら
        #     代入ではなくマージに直す必要がある
        blocks = _as_blocks(system_instruction)
        if response_schema:
            instruction = _json_only_instruction(response_schema)
            # 最後の段（phase固有）へ足す。phaseが決まればスキーマも決まるので、
            # 別の段にするとその段だけキャッシュが当たらなくなる。
            blocks = [*blocks[:-1], blocks[-1] + instruction] if blocks else [instruction]
        if blocks:
            kwargs["system"] = self._cached_system(blocks)

        # 思考の指定はtoken数ではなくeffortで行う
        # （budget_tokensは現行モデルでは廃止。送ると400）。
        #
        # NONE（思考OFF）が使えないモデルがある。思考が常時ONの系列へ
        # disabledを送ると400になる。丸めないのはThoughtLevelの方針どおり。
        #
        # 思考を切ると、tool_useブロックではなく可視テキストへツール呼び出しを
        # 書く挙動が報告されている（呼び出されず、エラーも出ないため
        # ステップが静かに空回りする）。弱めたいだけならLOWを使う。
        if thought_level is ThoughtLevel.NONE:
            kwargs["thinking"] = {"type": "disabled"}
        elif thought_level is not None:
            kwargs["thinking"] = {"type": "adaptive"}
            # ThoughtLevelの段階名をそのまま送る（Gemini / OpenAIと同じ扱い）。
            # 受け付ける段階はモデルごとに違うため対応表は持たない。
            # 受け取れない段階（Claudeにはminimalが無い等）はAPIがエラーを返す。
            kwargs["output_config"] = {"effort": thought_level.name.lower()}
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            if tool_choice:
                # "any"（必ずどれかを呼ばせる）を受け付けないモデルがある。
                # "auto"へ落としたりはしない——「選ばせない」つもりの実行が
                # 「呼ぶかもしれない」に変わり、気付けないままずれる。
                kwargs["tool_choice"] = {"type": tool_choice}

        response = self.client.messages.create(**kwargs)

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        cache_write_tokens = 0
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            output_tokens = usage.output_tokens or 0
            # Claudeだけ意味が違う。input_tokensはキャッシュ分を含まず、
            # 読み込み分・書き込み分が別建てで報告される。
            # 他プロバイダ（内数として報告）と揃えるため、合計をinput_tokensとする。
            cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
            input_tokens = (usage.input_tokens or 0) + cached_tokens + cache_write_tokens

        # Claudeは複数のブロック（text / tool_use）を並べて返す。
        # content[0]をtextと決めつけると、tool_useが先頭に来た場合に落ちる。
        texts = []
        calls = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(FunctionCall(name=block.name, args=dict(block.input or {})))

        return LLMResponse(
            text="".join(texts),
            function_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            # 拒否（refusal）も上限打ち切り（max_tokens）も、例外ではなく
            # 200の応答として届く。textとfunction_callsだけを見ていると
            # 「モデルが何も返さなかった」と同じに見える。
            stop_reason=_normalize_stop_reason(
                getattr(response, "stop_reason", None), _CLAUDE_STOP_REASONS
            ),
        )


