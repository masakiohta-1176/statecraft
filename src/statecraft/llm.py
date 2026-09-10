"""
LLMプロバイダ（Gemini / OpenAI / Claude）への接続を抽象化する。


このモジュールの役割は1つで、**プロバイダごとの差異をここで吸収し、
Agent側からは同じ呼び方で使えるようにする**こと。


    response = llm.generate(model="...", prompt="...", tools=[...], ...)
    → LLMResponse（text と function_calls を持つ共通の形）


プロバイダごとに違うのは、たとえば次のような点である。


    - 会話履歴の表現         Geminiは role="model"、他は role="assistant"
    - ツール定義の形         それぞれ別のキー名・入れ子構造
    - 呼び出し要求の取り出し方  専用フィールド／JSON文字列／ブロック走査
    - 思考の強さの指定方法    段階名／トークン数／effort
    - 出力形式の強制         スキーマで強制できるもの、指示文にするしかないもの


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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


from .tools import Tool
from .utils import with_timing


# ==========================================
# 各プロバイダのSDKは、そのクラスを実際に生成する時に読み込む。
#
# モジュールの先頭でimportすると、Geminiしか使わない場合でも
# anthropicとopenaiのインストールが必須になる。
# 使わないプロバイダの依存を強制しないため、遅延させる。
# ==========================================
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




# ==========================================
# 0. データ構造 (ThoughtLevel / LLMResponse / EmbeddingResponse)
# ==========================================
class ThoughtLevel(IntEnum):
    """
    「どれくらい考えさせたいか」というAgent側の意図。


    プロバイダの語彙ではなく意図を表す。モデルごとにサポートする段階が違う
    （HIGHが無い、minimalがある等）が、その差異を吸収するのは各LLMクラスの責務であり、
    Agentは意図を伝えるだけでよい。


    IntEnumにしているのは大小比較のため。「サポートされている中で最も近い段階へ丸める」
    という処理が、通常のEnumでは比較できず書けない。
    NONEは「明示的に思考させない」であり、thought_level=None（引数自体を渡さず
    モデルのデフォルトに任せる）とは別の意味を持つ。
    """


    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3




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
    input_tokens: int = 0  # このリクエストで消費した入力トークン数
    output_tokens: int = 0  # このリクエストで生成した出力トークン数
    execution_time: float = 0.0  # API応答にかかった時間（秒）。with_timingが自動で埋める


    @property
    def has_calls(self) -> bool:
        """
        ツールの呼び出し要求が含まれているかどうか。


        「テキスト応答」と「呼び出し要求」を区別するための種別フィールドは
        持たせていない。function_callsが空かどうかで判定できるため、
        種別を別に持つと同じ情報が2箇所に存在することになる。


        なお、textもfunction_callsも両方空という応答もありうる（モデルが
        何も返さなかった場合）。呼び出し側は
        「has_callsがFalse かつ textも空」でその状態を判定する。
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
# OpenAI/Claudeはcontentキーでroleが"assistant"）。呼び出し側に
# その差を意識させると「OpenAIでは動いたのにGeminiで壊れる」が起きるため、
# 共通形式を1つ決めて各クラスが変換する（FunctionCallと同じ方針）。
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"




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


        normalized.append((role, content))
    return normalized




def _json_only_instruction(response_schema) -> str:
    """
    response_schemaに相当する仕組みを持たないプロバイダ向けに、
    スキーマを指示文へ落とし込む。


    Geminiはresponse_schemaで構造を強制できるが、OpenAIとClaudeには
    ルートが配列のスキーマをそのまま強制する手段が無い。
    受け取っておいて無視すると、指定したのに効かない状態になるため、
    最低限プロンプトとして伝える（受け取り側は寛容にパースする）。
    """
    lines = [
        "",
        "出力は次のJSON Schemaに厳密に従うJSONのみとすること。",
        "前後に説明文やコードブロックの記号を付けてはならない。",
        json.dumps(response_schema, ensure_ascii=False),
    ]
    return "\n".join(lines)




# ==========================================
# 1. 統一規格（基底クラス）
# ==========================================
class BaseLLM(ABC):
    """
    LLMプロバイダへの接続。


    モデル名を保持しない。どのモデルを使うかはAgent（およびphaseごとの上書き）が
    決めるため、generate()の引数として毎回受け取る。
    プロバイダのSDK自体も接続とモデル指定を分けている（クライアントは
    モデルに紐づかず、リクエストごとにモデル名を渡す）ので、その形に合わせる。


    結果として、インスタンスはプロバイダの数だけあればよい。
    「flashで低め」「flashで高め」「proで高め」のような組み合わせごとに
    インスタンスを作る必要はない。
    """


    @abstractmethod
    def generate(
        self,
        *,
        model: str,  # 使用するモデル名。Agentが指定する
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str
        | None = "auto",  # 'auto' | 'any'（toolsを使わせたくない場合はtools自体を渡さない = Agentの責務）
        response_schema: Any | None = None,
        temperature: float,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        pass




# ==========================================
# 2. Gemini 統合クラス（AI Studio / Vertex AI 両対応）
# ==========================================
# 思考設定の受け取り方（style）ごとに、実際に受け取れる段階。
#
# これはモデル名の対応表ではなく、styleの対応表である点が重要。
# Geminiのモデルはほぼ毎月増えるが、styleは2種類しかなく増えない。
# だから「新しいモデルが出るたびに追記する」という運用は発生しない。
#
#   level  … gemini-3系。thinking_levelで段階名を受け取る。
#            "none"に相当する語彙が無いため、NONEを含めない（丸めでLOWへ上がる）。
#   budget … gemini-2.5系。thinking_budgetでトークン数を受け取る。0で無効化できる。
_STYLE_SUPPORTED: dict[str, tuple[ThoughtLevel, ...]] = {
    "level": (ThoughtLevel.LOW, ThoughtLevel.HIGH),
    "budget": (ThoughtLevel.NONE, ThoughtLevel.LOW, ThoughtLevel.MEDIUM, ThoughtLevel.HIGH),
}




class GeminiLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str = "global",
        use_vertex: bool = False,
        # 思考設定の受け取り方。"level"（gemini-3系）か "budget"（gemini-2.5系）。
        # モデル名ではなく系列に紐づく情報なので、接続時に1回決めれば足りる。
        # 世代を混在させる場合（3系と2.5系を同時に使う）は、styleごとに
        # インスタンスを2つ作る。
        #
        # 未指定（None）なら思考設定を一切送らず、モデルの既定動作に任せる
        # （安全側のデフォルト。未対応の値を送ってエラーにするより、
        # 何も送らない方が動く可能性が高い）。
        thinking_style: str | None = None,
        thinking_budgets: dict[ThoughtLevel, int] | None = None,
    ):
        _ensure_genai()
        if thinking_style is not None and thinking_style not in _STYLE_SUPPORTED:
            # 誤字は生成のたびにではなく、接続を作った瞬間に検出させる
            # （早期に大きな声で落とす）。
            valid = " / ".join(f'"{s}"' for s in _STYLE_SUPPORTED)
            raise ValueError(f"thinking_styleは {valid} のいずれかです: {thinking_style!r}")
        self.thinking_style = thinking_style
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


        サポートされていない段階を要求された場合は、最も近い段階へ丸める。
        同距離で並んだ場合は低い方を選ぶ（トークン消費を予測可能にするため）。
        """
        if thought_level is None or self.thinking_style is None:
            return None


        supported = _STYLE_SUPPORTED[self.thinking_style]
        if thought_level not in supported:
            # 要求された段階を別名へ写してから丸める。
            # lambdaは変数の「今の値」ではなく変数そのものを見に行くため、
            # 代入先のthought_levelを直接参照させると、
            # 評価される時点でどの値を指しているかが読み取りづらくなる。
            requested = thought_level
            # キーをタプルにすることで、距離が同じなら値が小さい方が勝つ
            thought_level = min(supported, key=lambda s: (abs(s - requested), s))


        if self.thinking_style == "level":
            # gemini-3系は段階名をそのまま受け取るので、数値の指定は不要。
            return types.ThinkingConfig(thinking_level=thought_level.name.lower())


        # 以下はbudgetスタイル（gemini-2.5系）。数値が必要になる。
        if thought_level is ThoughtLevel.NONE:
            # 0は「思考させない」という意味そのものなので、チューニング値ではない。
            return types.ThinkingConfig(thinking_budget=0)


        budget = self._thinking_budgets.get(thought_level)
        if budget is None:
            # 何トークン割り当てるかの指定が無いので、勝手に決めずモデルに任せる。
            return None
        return types.ThinkingConfig(thinking_budget=budget)


    # 戻り値の型を list[Any] にしているのは、遅延importの都合による。
    # 実際に返すのは list[google.genai.types.Tool] だが、typesは
    # モジュール先頭で None として宣言した「変数」であり、
    # 型注釈の中では使えない（型チェッカーが
    # 「型式では変数を使用できません」として弾く）。
    #
    # TYPE_CHECKINGブロックで本物をimportして注釈に使う手もあるが、
    # それをすると google-genai が入っていない環境で型チェックが通らなくなり、
    # 「使わないプロバイダのSDKを要求しない」という遅延importの目的と衝突する。
    # 型の正確さより、依存を増やさないことを優先している。
    def _format_tools(self, tools: list[Any]) -> list[Any]:
        declarations = []
        for t in tools:
            # isinstance(t, Tool)で判定してはならない。AgentはToolを継承せず
            # Invokableを構造的に満たすだけなので、継承で判定するとAgentが
            # 変換されずそのままAPIへ渡り、宣言情報が失われる。
            # to_declaration()を持つかどうか＝Invokableかどうかで判定する。
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
        system_instruction: str | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        temperature: float,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # --- ① 会話履歴をGeminiの形へ変換する -------------------------------
        # 共通形式 [{"role": "user"|"assistant", "content": str}] を、
        # Geminiが要求する Content(role=..., parts=[...]) の列へ移す。
        # Geminiはassistant側のroleを "model" と呼ぶため、そこも変換する。
        contents = [
            types.Content(
                role="model" if role == _ROLE_ASSISTANT else "user",
                parts=[types.Part.from_text(text=text)],
            )
            for role, text in _normalize_messages(messages)
        ]


        # --- ② 今回のターンを1つのuserメッセージとして足す -------------------
        # 添付（画像・PDF）と本文を同じuserターンにまとめる。
        # 履歴の「後ろ」に足すことで、モデルから見て
        # 「過去のやり取り → 今回の依頼」という並びになる。
        #
        # promptとmessagesは併用される。messagesがある時にpromptを
        # 捨ててしまうと、履歴だけ送って今回の依頼が消えることになる。
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
        # tool_choice="any"  … 必ずどれかを呼ばせる（Agentがtasksで対象を絞った時）
        # tool_choice="auto" … 呼ぶか答えるかをモデルに委ねる
        # ツールを使わせたくない場合は、tools自体を渡さない（呼び出し側の責務）。
        formatted_tools = self._format_tools(tools) if tools else None
        tool_config = None
        if formatted_tools and tool_choice:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=tool_choice.upper())
            )


        # --- ⑤ リクエスト設定をまとめる --------------------------------------
        config = types.GenerateContentConfig(
            # SDKの自動関数呼び出し（AFC）を必ず切る。
            #
            # 有効なままだと、SDKが「このツールを呼べ」という応答を受け取った
            # 時点で、SDK自身がPython関数を呼び出してしまう。
            # このフレームワークは呼び出しの手前で
            #   ・無効化されていないか
            #   ・同じ引数で既に実行済みでないか
            #   ・実行前の判定（Interceptor）を通るか
            # を確認し、実行時間の計測や記憶への直接書き込みも行う。
            # SDKに先に実行されると、それら全部が迂回される。
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
            system_instruction=system_instruction,
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
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0


        # --- ⑧ 呼び出し要求を共通の形へ変換する ------------------------------
        calls = [
            FunctionCall(name=c.name, args=dict(c.args or {}))
            for c in (response.function_calls or [])
        ]


        # --- ⑨ テキスト部分だけを取り出す ------------------------------------
        # SDKには response.text という近道があるが、応答に呼び出し要求が
        # 含まれていると「テキスト以外のパートがある」という警告を出す。
        # 応答は複数のパート（テキスト／呼び出し要求／添付）に分かれているので、
        # テキストのパートだけを自分で拾って連結すれば、
        # 警告なしで同じ結果が得られる。
        texts = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    texts.append(part.text)


        return LLMResponse(
            text="".join(texts),
            function_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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


        # strict=True: 送ったテキスト数と返ってきたベクトル数が食い違ったら例外にする。
        # 既定の挙動（短い方で打ち切る）だと、一部のテキストが黙って
        # 埋め込まれないまま処理が続き、後段で「なぜか一部だけ検索に出ない」
        # という形でしか現れなくなる。
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




# ==========================================
# 3. OpenAI クラス
# ==========================================
_OPENAI_REASONING_EFFORT = {
    ThoughtLevel.NONE: "minimal",
    ThoughtLevel.LOW: "low",
    ThoughtLevel.MEDIUM: "medium",
    ThoughtLevel.HIGH: "high",
}




# ==========================================
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


    @with_timing
    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        temperature: float,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # response_formatのjson_objectモードはルートをオブジェクトに強制するため使わない。
        # memory差分はルートが配列である前提（キーの捏造を防ぐための設計）なので、
        # スキーマは指示文として渡し、パース側で吸収する。
        if response_schema:
            system_instruction = (system_instruction or "") + _json_only_instruction(
                response_schema
            )


        api_messages = []
        if system_instruction:
            api_messages.append({"role": "system", "content": system_instruction})


        # 過去の履歴 → 今回のターン、の順で組み立てる。
        api_messages.extend(
            {"role": role, "content": text} for role, text in _normalize_messages(messages)
        )


        if prompt or blobs:
            content = []
            for blob in blobs or []:
                b64_data = base64.b64encode(blob["data"]).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{blob['mime_type']};base64,{b64_data}"},
                    }
                )
            if prompt:
                content.append({"type": "text", "text": prompt})
            api_messages.append({"role": "user", "content": content})


        kwargs = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = "required" if tool_choice == "any" else tool_choice


        # OpenAIは思考の強さをreasoning_effortという語彙で受け取る。
        # 段階名の対応だけなので、トークン数のようなチューニング値は不要。
        if thought_level is not None:
            kwargs["reasoning_effort"] = _OPENAI_REASONING_EFFORT[thought_level]


        response = self.client.chat.completions.create(**kwargs)


        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0


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




# ==========================================
# 4. Claude クラス
# ==========================================
# ThoughtLevelをClaudeのeffortへ対応させる。
# Claudeは思考の深さをトークン数ではなくeffortという段階で受け取る。
_CLAUDE_EFFORT = {
    ThoughtLevel.LOW: "low",
    ThoughtLevel.MEDIUM: "medium",
    ThoughtLevel.HIGH: "high",
}


# サンプリングパラメータ（temperature等）を受け付けないモデル。
# 該当モデルへtemperatureを送ると400になるため、送信自体を止める。
_CLAUDE_NO_SAMPLING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)




class ClaudeLLM(BaseLLM):
    def __init__(self, *, api_key: str | None = None):
        Anthropic = _require("anthropic", "anthropic").Anthropic
        self.client = Anthropic(api_key=api_key)


    @staticmethod
    def _accepts_sampling(model: str) -> bool:
        """
        そのモデルがtemperature等のサンプリング指定を受け付けるか。


        モデルがリクエストごとに変わる（phaseごとに切り替えられる）ため、
        接続時ではなく生成時に判定する。
        """
        return not model.startswith(_CLAUDE_NO_SAMPLING_PREFIXES)


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


    @with_timing
    def generate(
        self,
        *,
        model: str,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system_instruction: str | None = None,
        blobs: list[dict] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: str | None = "auto",
        response_schema: Any | None = None,
        temperature: float,
        max_tokens: int,
        thought_level: ThoughtLevel | None = None,
    ) -> LLMResponse:
        # 過去の履歴 → 今回のターン、の順で組み立てる。
        #
        # 値の型をAnyにしているのは、contentが文字列とリストの両方を取るため。
        # 履歴はテキストだけなので文字列を入れるが、今回のターンで画像を含める場合は
        # ブロックのリストになる（Messages APIがどちらの形も受け付ける）。
        api_messages: list[dict[str, Any]] = [
            {"role": role, "content": text} for role, text in _normalize_messages(messages)
        ]


        if prompt or blobs:
            content: list[dict[str, Any]] = []
            for blob in blobs or []:
                b64_data = base64.b64encode(blob["data"]).decode("utf-8")
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": blob["mime_type"],
                            "data": b64_data,
                        },
                    }
                )
            if prompt:
                content.append({"type": "text", "text": prompt})
            api_messages.append({"role": "user", "content": content})


        kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        # 現行モデルではサンプリング指定が400になるため、受け付けるモデルにだけ送る。
        if self._accepts_sampling(model):
            kwargs["temperature"] = temperature


        # Claudeにはresponse_schema相当が無いため、指示文として渡す。
        if response_schema:
            system_instruction = (system_instruction or "") + _json_only_instruction(
                response_schema
            )
        if system_instruction:
            kwargs["system"] = system_instruction


        # 思考の指定はtoken数ではなくeffortで行う。
        # budget_tokensは現行モデルでは廃止されており、送ると400になる。
        if thought_level is ThoughtLevel.NONE:
            kwargs["thinking"] = {"type": "disabled"}
        elif thought_level is not None:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": _CLAUDE_EFFORT[thought_level]}
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = {"type": tool_choice}


        response = self.client.messages.create(**kwargs)


        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0


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
        )





