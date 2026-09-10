# StateCraft


複数のAIエージェントが協調して問い合わせに回答するPython製フレームワーク。


LLMに多段の手順を踏ませる時の失敗を、**プロンプトでお願いするのではなく構造で防ぐ**。


- 使わせたくないツールは、選択肢として提示しない
- 出力形式は依頼せず、スキーマで強制する
- 実行対象は記憶に書いたタスクから決まる（LLMが自由に選ばない）
- ツールの結果は記憶へ要約し、以降のプロンプトから外す


軽量なモデルでも多段の手順が完遂する。


**なぜこの作りなのかは [設計思想.md](docs/設計思想.md)。** サンプルの読み方は [sampleの読み方.md](docs/sampleの読み方.md)。


---


## 目次


- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [Agent](#agent)
- [Tool](#tool)
- [記憶](#記憶)
- [Network](#network)
- [Interceptor](#interceptor)
- [API側の失敗に備える](#api側の失敗に備える)
- [Phase](#phase)
- [よくある構成](#よくある構成)
- [実装状況](#実装状況)


---


## インストール


```bash
pip install -e .
```


使うプロバイダのSDKは別に入れる。全部は要らない。


```bash
pip install -e ".[gemini]"    # Gemini
pip install -e ".[openai]"    # OpenAI
pip install -e ".[claude]"    # Claude
```


対応するクラスを初めて使う時にSDKのimportを試みるので、入れていないプロバイダがあっても他は動く。


APIキーは環境変数か、起動時に読み込む。


```python
import os


api_key = os.environ["GEMINI_API_KEY"]
```


Python 3.11 以上（`StrEnum` を使用）。


---


## クイックスタート


```python
from statecraft import Agent, GeminiLLM, Network, Tool




def search_docs(keyword: str) -> str:
    """社内ドキュメントを検索する"""
    return "..."




llm = GeminiLLM(api_key=api_key, thinking_style="level")


worker = Agent(
    name="worker",
    summary="ドキュメントを調べて判断材料を揃える",
    llm=llm,
    model="gemini-3.5-flash",
    system_instruction="あなたは調査担当です。...",
    tools=[Tool(func=search_docs, summary="社内ドキュメントを検索する")],
)


front = Agent(
    name="front",
    summary="利用者との窓口",
    llm=llm,
    model="gemini-3.5-flash",
    system_instruction="あなたは受付です。...",
    sub_agent_names=["worker"],
)


net = Network(agents=[front, worker])
response = front.respond(message="就業規則の有給の項目を教えて")


print(response.text)
print(net.total_tokens())  # (入力, 出力)
```


`Network` は1リクエストにつき1インスタンス。ここで共有記憶が作られ、全Agentへ注入される。委譲先の名前の解決、循環の検出、配線の検証もここで行われる。


自己完結したサンプルは `sample.py`。


```bash
python sample.py "貸出は何冊まで？"
python sample.py "貸出は何冊まで？" --fresh    # 前回の記憶を引き継がない
```


---


## Agent


ReActループを回す。`tools` と `sub_agent_names` を持ち、記憶へ判断を積み上げながら進む。


### フィールド


| | 型 | 説明 |
|---|---|---|
| `name` | `str` | **必須。** 委譲時に名前で参照される |
| `summary` | `str` | **必須。** 委譲候補として常時提示される1行 |
| `llm` | `BaseLLM` | **必須。** 接続。モデル名は持たない |
| `model` | `str` | **必須。** 使うモデル名 |
| `system_instruction` | `str` | 人格・役割の定義 |
| `usage` | `str` | 委譲先として選ばれた時だけ渡す使い方の制約 |
| `evaluation` | `str` | このAgentの回答を受け取った側への、扱い方の指示 |
| `knowledge` | `str` | 常時プロンプトへ載せる本文 |
| `tools` | `list[Tool]` | 使えるツール |
| `sub_agent_names` | `list[str]` | 委譲先の名前。`Network` が実体へ解決する |
| `input_schema` | `dict` | 委譲時に受け取る引数の形（JSON Schema） |
| `output_schema` | `dict \| None` | 最終回答に強制する形式 |
| `initial_tool_name` | `str \| None` | ループ前にシステムが1回だけ実行するツール名 |
| `max_steps` | `int` | ループの上限（既定 10） |
| `max_memory_retries` | `int` | 記憶の差分が不正だった時の再生成回数（既定 3） |
| `max_stalled_steps` | `int` | 同じ実行が何回続いたら打ち切るか（既定 2） |
| `temperature` | `float` | 既定 1.0 |
| `max_tokens` | `int` | 既定 8192 |
| `thought_level` | `ThoughtLevel \| None` | `NONE` / `LOW` / `MEDIUM` / `HIGH`。`None` ならモデルの動的思考 |
| `phase_overrides` | `dict[Phase, GenerationConfig]` | phaseごとの生成設定の上書き |
| `disabled` | `bool` | `True` の間は委譲候補として提示されない |
| `execution_message` | `str` | 委譲された時に画面へ出す文言 |
| `describe_execution` | `Callable[[dict], str] \| None` | 文言を動的に組む場合 |
| `interceptor` | `Interceptor` | イベントの通知先 |
| `shared_memory` | `SharedMemory` | 通常は `Network` が注入する |


### ReflexAgent


`Agent` を継承し、**記憶を持たない**。記憶の構築・更新フェーズが無いので生成回数が少ない。


| | `Agent` | `ReflexAgent` |
|---|---|---|
| 記憶 | 持つ | 持たない（共有記憶は読む） |
| 実行対象の決定 | 記憶に書いたタスクから | 有効な全ツールから自分で選ぶ |
| 1往復の生成回数 | 多い（計画・実行・評価・回答） | 少ない（実行・回答） |


**`ReflexAgent` が向くのは、担う役割が1〜2個で速度を優先したい場合。** 単発の分類、定型の変換、審査など。


複数のタスクを積んで進める役割には `Agent` を使う。記憶が無いと「何を終えて何が残っているか」を保持できない。


### phaseごとにモデルを変える


`phase_overrides` で上書きする。指定しなかった phase は `model` の既定値を使う。


```python
from statecraft import GenerationConfig, Phase


worker = Agent(
    model="gemini-3.5-flash-lite",  # 既定
    phase_overrides={
        Phase.INITIAL_MEMORY: GenerationConfig(model="gemini-3.7-flash"),
        Phase.MEMORY_UPDATE: GenerationConfig(model="gemini-3.7-flash"),
    },
    ...
)
```


`GenerationConfig` は `model` / `llm` / `temperature` / `max_tokens` / `thought_level` を持つ（すべて任意）。


**接続（`llm`）は1つ作って全Agentで共有できる。** モデル名を持たないため。プロバイダを混在させる場合だけインスタンスを分ける。


| phase | 割り当ての目安 |
|---|---|
| `INITIAL_MEMORY` / `MEMORY_UPDATE` | 読み取りが中心。軽いモデルだと項目が落ちる |
| `FUNCTION_CALL` | `tasks` に書かれた対象を呼ぶだけなら軽くて足りる、SQL生成など引数の品質が重視される場合はここを上位モデルにする |
| `ANSWER` | 文章を作る場合は上位。`output_schema` で状態だけ返す場合は軽くてよい |


### 初期ツール


「呼ぶかどうか」も「何を呼ぶか」も決まっている手順は、LLMに選ばせない。


```python
Agent(
    initial_tool_name="search_docs",
    input_schema={
        "type": "object",
        "properties": {"keyword": {"type": "string"}},  # ツールの引数名と一致させる
        "required": ["keyword"],
    },
    ...
)
```


ループ前にシステムが1回実行し、その結果を材料に初期記憶を作る。**生成1回とその失敗可能性が消える。** 引数は `input_schema` の同名プロパティから自動で渡される（名前が一致しないと `Network` 構築時に落ちる）。


同じ引数での再実行は拒否されるので、LLMが直後に同じ呼び出しをしても二重に実行されない。


---


## Tool


Python関数をそのまま渡す。**引数のJSON Schemaは型ヒントから自動生成される。**


```python
def issue_ticket(title: str, member_id: str, tags: list[str]) -> str:
    """チケットを発行する"""
    ...




Tool(
    func=issue_ticket,
    summary="チケットを発行する",
    param_descriptions={"title": "件名", "member_id": "会員番号"},
)
```


`list[str]` は `{"type": "array", "items": {"type": "string"}}` に、`str | None` は `{"type": "string"}` になる。


### フィールド


| | 型 | 説明 |
|---|---|---|
| `func` | `Callable` | **必須。** ツール本体 |
| `summary` | `str` | **必須。** 常時提示される1行 |
| `param_descriptions` | `dict[str, str]` | 引数の説明 |
| `usage` | `str` | 呼ばれる候補になった時だけ渡す制約 |
| `evaluation` | `str` | **結果と一緒に渡す**、その結果の読み方 |
| `write_to_memory` | `bool` | 戻り値をLLMを介さず記憶へ直接書く |
| `expose_error_details` | `bool` | 例外の詳細をLLMへ返すか（既定 `False`） |
| `disabled` | `bool` | `True` の間は提示されない |
| `execution_message` | `str` | 実行中に画面へ出す文言。`{引数名}` で差し込める |
| `describe_execution` | `Callable[[dict], str] \| None` | 文言を動的に組む場合 |


### summary / usage / evaluation


3つとも文言だが、**渡るタイミングが違う**。


| | いつ渡るか | 何を書くか |
|---|---|---|
| `summary` | 常時 | 何をするものか（1行） |
| `usage` | 呼ばれる候補になった時 | スキーマで表現できない制約 |
| `evaluation` | **結果と一緒に1回** | その結果をどう読むか |


`evaluation` は結果が返った瞬間だけ現れる。


```
  - search_rules(topic='貸出')
    【貸出】一般利用者は5冊まで、貸出期間は14日間。...
    【この結果の解釈】条文をそのまま採用せず、問われている条件に該当するかを確かめること。
```


各ツールの結果の見方を人格定義へ書くと、ツール数に比例してプロンプトが伸び、ツールを足すたびにオーケストレーター側を書き換えることになる。**`evaluation` はそのツールの定義に置く。**


`Agent` も同じ3つを持つ。委譲先として選ばれた時に `usage` が渡り、回答が返った時に `evaluation` が渡る。


### write_to_memory


一字一句正確に残したい値（識別子、受付番号、SQLなど）は、LLMの要約を経由させず記憶へ直接書ける。記憶差分と同じ形で返す。


```python
def issue_ticket(title: str, member_id: str) -> list[dict]:
    ticket = "TKT-00123"
    return [
        {"field": "vars", "text": f"受付番号: {ticket}"},
        {"field": "facts", "text": f"『{title}』のチケットを発行した。"},
    ]




Tool(func=issue_ticket, summary="...", write_to_memory=True)
```


`id` は省略できる（システムが採番する）。履歴側では「※結果はmemoryへ記録済み」となり、中身が二重に載らない。


---


## 記憶


ツールの結果は記憶へ要約され、**以降のプロンプトから中身が外れる**。5,000文字の規程を取得しても、次のステップで見えるのは要約された1〜2行になる。


LLMは記憶を直接書き換えず、**差分の配列**を返す。


```json
[
  {"field": "facts", "id": "fact-1", "text": "一般利用者は5冊まで"},
  {"field": "tasks", "id": "task-1", "text": "在庫を確認する", "status": "next", "target_names": ["check_stock"]}
]
```


`field` に書き込み先を指定させる。取りうる値は `enum` で列挙されるので、存在しないプロパティは出力できない。既存の `id` を指定すると更新になる。**削除する経路は用意していない**（不要になった情報は上書きさせる）。


### SharedMemory


セッション内の全Agentが同一実体を共有する。


| | 内容 |
|---|---|
| `facts` | 確定した情報。対象ごとの断片 |
| `back_grounds` | 対象そのもののモデル（定義・構成・制約・状態・関係） |
| `state_briefing` | 断片どうしの繋がりと、現時点の理解。時系列で積み上がる |
| `vars` | 一字一句失われると困る値（ID・URL・SQL・キー） |
| `actions` | 何を確認するために何を行い、結果どうなったか |
| `decisions` | 何を採用し、何を採用しなかったか |
| `hypotheses` | 根拠はあるが未確定の読み取り |
| `open_questions` | 自分で調べれば解決できる未解決の論点 |
| `requests` | **システムが書く。** 今回の要求 |
| `agent_answers` | **システムが書く。** 各Agentの回答 |


`facts` / `back_grounds` / `state_briefing` は3つで1組。`facts` は断片なので、それだけを読んだ相手は論点を追えない。残り2つが補う。


`requests` と `agent_answers` は差分スキーマの `enum` に現れないため、LLMは指定できない。


### PrivateMemory


Agentごとに別の実体。共有されない。


| | 内容 |
|---|---|
| `goals` | このAgentが達成すべきこと |
| `tasks` | 追加実行が必要な作業のキュー |
| `task_notes` | 作業に必要だが外へ出さない知識、tool/agentの使い方の取り決め |


### tasks


**実行指示として機能する。** `status` が `next` / `next_parallel` のものだけが実行対象になる。


| `status` | 意味 |
|---|---|
| `next` | 次に実行する |
| `next_parallel` | 次に同時実行する（相互に依存しない場合のみ） |
| `conditional` | 前段の結果次第で実行する |
| `done` | 完了した |
| `unnecessary` | 前提が崩れた、または不要になった |


`target_names` に書かれた対象だけが `FUNCTION_CALL` フェーズでAPIへ渡され、`tool_choice="any"` で呼び出しが強制される。**LLMが決めるのは引数だけ。**


1回のフェーズで2つ以上の呼び出しが返った場合、実行はスレッドで並列に走る。関数本体は同時に呼ばれるので、外部の状態を書き換えるツールを並列で呼ばせる場合は利用側で保護する。記憶への書き込みと `Interceptor` のコールバックはフレームワーク側で直列化してあるため、そちらは考えなくてよい（コールバックの順序は終了順になる）。


実行済みのタスクが `next` のまま残っていると、記憶の更新時に差し戻される。未実行のタスクが残ったまま回答へ進む場合は、その旨が最終回答の生成へ伝わる。


### 引き継ぎ


`Network` は1リクエストで破棄されるので、次のターンへ渡すものは利用側で保存する。


```python
# 保存
saved = {
    f: [{"id": e.id, "text": e.text} for e in getattr(net.shared_memory, f)]
    for f in ("back_grounds", "vars")
}


# 復元（Network構築後に代入する）
net.shared_memory.back_grounds = [MemoryEntry(**r) for r in saved["back_grounds"]]
```


| | 引き継ぐ | なぜ |
|---|---|---|
| `back_grounds` | **する** | 対象のモデル。質問が変わっても成り立つ |
| `vars` | **する** | 値そのものが変わらない |
| `facts` | しない | 別の件では無関係な事実が混ざる |
| `actions` | しない | 「もう調べた」と誤認する |
| `decisions` | しない | 前のターンの解釈に縛られる |
| `hypotheses` / `open_questions` | しない | 前のターンの未解決論点を追い続ける |


**フレームワークは方針を持たない。** 何をセッションとみなすかはアプリケーションごとに違うため。`sample.py` に実装例がある。


---


## Network


1リクエストにつき1インスタンス。


```python
net = Network(agents=[front, worker, planner])


net.get("front")  # net["front"] でも同じ
net.total_tokens()  # (入力, 出力)
net.token_usage()  # {"front": (入力, 出力), ...}
net.shared_memory  # 全Agentが共有する記憶の実体
```


構築時に行われること。


- `sub_agent_names` を実体へ解決する
- 共有記憶を全Agentへ注入する
- 循環参照を検出する
- `initial_tool_name` のツールが存在し、引数名が `input_schema` と一致するかを確認する
- `phase_overrides` で `llm` だけを上書きして `model` を書き忘れていないかを確認する


### 渡さないAgentもある


`agents` に渡すと**共有記憶が注入される**。逆に、共有記憶を持たせたくないAgentは渡さない。


```python
# 関門として使うAgent。委譲候補にもしないし、記憶も共有しない
auditor = ReflexAgent(name="auditor", ..., shared_memory=SharedMemory())


net = Network(agents=[front, worker])   # auditor は入れない
```


これは検証に引っかからない（誰からも呼ばれないAgentがあること自体はエラーにしていない）。


構築時に落ちるのは、`sub_agents` へ**実体を直接指定したのに、その実体が `agents` に含まれていない**場合。委譲先が別の共有記憶を持つと、書いた内容がどこにも反映されないまま完走してしまうため。


---


## Interceptor


イベントの通知と、実行前の拒否。


```python
from statecraft import Interceptor


ic = Interceptor()
ic.on.execute_start(lambda message: print(message))
ic.on.before_execute(guard)


net = Network(agents=[...], interceptor=ic)
```


### イベント


| イベント | 種別 | 受け取るもの |
|---|---|---|
| `before_execute` | **check** | `name`, `kwargs`（キーワード引数） |
| `generation_failed` | **check** | `agent`, `phase`, `model`, `attempt`, `error` |
| `execute_start` / `execute_end` / `execute_blocked` | notify | `message: str` |
| `respond_start` / `respond_end` | notify | `message: str` |
| `memory_updated` | notify | `message: str` |
| `memory_diff` | notify | `MemoryDiffEvent`（構造のまま） |
| `generated` | notify | `GenerationEvent`（構造のまま） |
| `error` | notify | `message: str` |


`before_execute` は**ツールとエージェントの両方で発火する**ので、委譲そのものも審査できる。`generation_failed` は [API側の失敗に備える](#api側の失敗に備える) を参照。


`memory_diff` は記憶の差分を構造のまま渡す。材料（`tool_results`）・出力（`rows`）・結果（`errors`）が揃うので、「この入力に対してこの記憶の作り方で合っているか」を検証できる。失敗時にも発火する。


```python
def record(event) -> None:
    print(f"{event.agent} / {event.phase} / {event.attempt}回目 / 適用={event.applied}")
    for row in event.rows:
        print(f"  {row.get('field')}[{row.get('id')}] {row.get('text')}")




ic.on.memory_diff(record)
```


`generated` は生成1回分の実測値を渡す。`agent` / `phase` / `model` / `elapsed`（秒）/ `input_tokens` / `output_tokens` を持つので、遅い原因が生成回数か1回の重さかを切り分けられる。`phase_overrides` の結果が `model` に出る。


```python
def measure(event) -> None:
    print(f"{event.agent} / {event.phase.value} / {event.model} / {event.elapsed:.1f}秒")




ic.on.generated(measure)
```


### 実行を止める


```python
def guard(*, name: str, kwargs: dict) -> bool | str:
    # ツール名で止める
    if name == "send_mail":
        return "外部への送信はこの環境では実行できません。"


    # 引数の内容で止める
    if name == "issue_ticket":
        if not kwargs.get("member_id", "").startswith("M-"):
            return "受け取った会員番号は正しい形式ではありません。利用者から取得してください。"


    # 外部の状態で止める
    if name == "delete_record" and not is_business_hours():
        return "営業時間外のため実行できません。"


    # 理由を伝えたくない場合
    if name == "check_stock" and kwargs.get("title") in RESTRICTED:
        return False


    return True




ic.on.before_execute(guard)
```


| 戻り値 | 結果 |
|---|---|
| `True` | 許可 |
| 文字列 | 拒否。**その文字列が理由としてLLMへ渡る** |
| `False` / `None` / 戻り値なし / 例外 | 拒否。理由は渡らない |


**判定できなかったものが「許可」として通ることはない。**


理由を返すかどうかは拒否ごとに選ぶ。引数を直せば通るもの（形式の誤り）や条件が分かれば諦められるもの（営業時間外）は伝える。理由そのものが審査基準になる場合は `False` を返す。


**理由に「何なら通るか」は書かない。** 形式を教えると、その形式に合わせて値を作られる。


```
NG: 「会員番号は M- で始まります」
      → 「12345」を「M-12345」へ書き換えて実行される


OK: 「受け取った会員番号は正しい形式ではありません。利用者から取得してください。
      値を推測したり、形式に合わせて書き換えたりしてはいけません」
```


形式チェックだけでは形を真似た値が通る。**実在の確認はツール本体で行う。**


### 複数のコールバック


`before_execute` は複数登録でき、**全員が `True` を返した時だけ許可**される。観測だけを行う関数でも `return True` が必須。


判定は登録順に呼ばれ、1つが拒否した時点で後続は呼ばれない。**観測用は拒否用より先に登録する**（後ろに置くと、止められた実行の引数だけがログに残らない）。


```python
def watch(*, name: str, kwargs: dict) -> bool:
    print(f"[引数] {name}({kwargs})")
    return True  # 書き忘れると全ての実行が拒否される




ic.on.before_execute(watch)  # 先に登録する
ic.on.before_execute(guard)
```


### 判定をエージェントに行わせる


`guard` の中から別のAgentを直接 `execute` できる。


```python
auditor = ReflexAgent(name="auditor", ..., shared_memory=SharedMemory(), max_steps=1)




def guard(*, name: str, kwargs: dict) -> bool | str:
    if name in NEEDS_AUDIT:
        result = auditor.execute(kwargs={"message": f"次の実行を審査せよ: {name} {kwargs}"})
        if not result.success:
            return False  # 審査できなかったものは通さない
        return json.loads(result.value).get("allowed") is True
    return True
```


`auditor` を `Network` に渡さないことで、委譲候補として提示されず、共有記憶も汚さない。`interceptor` を渡さなければ、審査中に審査が再び走ることもない。


### 進捗メッセージ


文言は tool / agent 自身が持つ。受け取る側に分岐を書かない。


```python
Tool(func=check_stock, summary="...", execution_message="『{title}』の在庫を照会しています...")
Agent(name="worker", execution_message="調査しています...", ...)


ic.on.execute_start(lambda message: send_to_frontend(message))
```


引数は `{名前}` で差し込める。渡さなければ開発者向けの汎用文が使われる。差し込みに失敗しても例外にはならない（そのキーが空文字になる）。


同じ Interceptor に**宛先の違う購読者を並べられる**。`execute_start` は利用者の画面へ、`execute_blocked` や `error` は内部の文言なのでログへ。


---


## API側の失敗に備える


生成はAPI越しなので失敗する。落ち方は2種類あり、**対処が逆になる**。


| 落ち方 | 例 | 復活するか | 対処 |
|---|---|---|---|
| 一時的な過負荷 | `503 UNAVAILABLE` | 待てば通る | 待って同じモデルで再試行 |
| 枠の使い切り | `429 RESOURCE_EXHAUSTED` | **その日は復活しない** | 別のモデルへ逃がす |


`429` はレスポンスに `retryDelay: 29s` が入っていても、それが1日あたりの上限（`GenerateRequestsPerDayPerProjectPerModel`）であれば待っても同じエラーが返る。同じモデルで何度やり直しても越えられない。


どちらも `generation_failed` で扱う。生成が例外で終わった時に、**もう一度生成するかどうか**を判定するイベント。


| 戻り値 | 意味 |
|---|---|
| `True` | 同じモデルで再試行する |
| 文字列 | **そのモデル名**で再試行する |
| それ以外（`False` / `None` / 例外） | 諦める |


待つ場合はコールバックの中で待つ。何秒待つか、何回試すか、どのモデルへ逃がすかはフレームワークが持たない。


```python
def retry_on_overload(
    *, agent: str, phase, model: str, attempt: int, error: Exception
) -> bool | str:
    text = str(error)


    # 枠を使い切った場合は待っても同じエラーが返る。モデルを変える
    if "RESOURCE_EXHAUSTED" in text and model != "gemini-3.5-flash-lite":
        return "gemini-3.5-flash-lite"


    # 一時的な過負荷なら、待てば同じモデルで通る
    if "503" in text and attempt < 3:
        time.sleep(2.0 * attempt)
        return True


    return False




ic.on.generation_failed(retry_on_overload)
```


`model` は**その回に実際に使ったモデル**なので、切り替えた後は切り替え先が渡る。`generated` の `model` にも実際に成功したモデルが出る。


**登録しなければ再試行しない。** `before_execute`（登録が無ければ許可）とは既定が逆になる。実行の審査は「止める人がいなければ実行する」が正しく、再試行は「やると言う人がいなければやらない」が正しい。


文字列がモデル名を意味するのはこのイベントだけで、`before_execute` では文字列は拒否の理由になる。またこのコールバックだけは複数スレッドから同時に呼ばれる場合がある（待つ処理を直列化すると、無関係な実行まで止まるため）。


再試行しないと決めた場合、例外はそのまま外へ出る。委譲先の生成であれば `execute` が受け止めて `ToolResult(error=...)` になり、依頼元のAgentが状況を見て判断する（別の対象へ切り替える、`tasks` を `unnecessary` にする、など）。入口（`respond`）の生成であれば呼び出し側まで届くので、そこは利用側で受ける。


### 軽いモデルへ逃がす代償


上位モデルで作らせていた記憶が、軽いモデルでは落ちることがある。実際に `state_briefing`（今の状況と、回答に足りていないものの記述）が1件も書かれずに完走した例がある。回答そのものは出るが、情報が足りない場面で「何が足りないか」が残らなくなる。


落としてよい phase と落としたくない phase は別なので、逃がし先を phase で分けられる。


```python
def retry_on_overload(*, agent, phase, model, attempt, error):
    if "RESOURCE_EXHAUSTED" not in str(error):
        return False
    # 記憶を作るphaseは、軽いモデルへ落とすより諦めた方がよい場合もある
    if phase in (Phase.INITIAL_MEMORY, Phase.MEMORY_UPDATE):
        return False
    return "gemini-3.5-flash-lite"
```


---


## Phase


1往復の中で、生成は phase に分かれて行われる。


| phase | 何をするか | ツールをAPIへ渡すか | 出力形式の強制 |
|---|---|---|---|
| `INITIAL_MEMORY` | 目標とタスクを立てる。ツールの無効化もここだけ | — | — |
| `FUNCTION_CALL` | `tasks` の対象を呼ぶ | ✅ | — |
| `MEMORY_UPDATE` | 結果を評価して記憶へ反映 | — | — |
| `ANSWER` | 最終回答 | — | ✅ |
| `CUTOFF_ANSWER` | `max_steps` を使い切った場合の回答 | — | ✅ |
| `STALLED_ANSWER` | 停滞を検出して打ち切った場合の回答 | — | ✅ |


`ReflexAgent` は `INITIAL_MEMORY` と `MEMORY_UPDATE` を通らない。


`FUNCTION_CALL` の間は `system_instruction` が変化しないため、プロンプトキャッシュに乗る。毎ステップ変わるもの（記憶・実行履歴・依頼内容）は `prompt` 側にある。


---


## よくある構成


エージェントは**ハブとスポーク**に配置する。


```
利用者
  │
  ▼
Front（窓口）           会話の受け答えと、依頼の構造化
  │
  ▼
Orchestrator（統括）    記憶を持ち、知識の取得と全体の判断を担う
  │            │
  │ tool       │ 委譲
  ▼            ▼
知識取得       Specialist（専門家）
検索・参照     専門性・創造性を要する作業
```


| 層 | 担うもの | 担わないもの |
|---|---|---|
| 窓口 | 会話、依頼の構造化 | 調査、可否の判断 |
| 統括 | 知識の取得、情報収集、全体の判断 | 専門的な作業そのもの |
| 専門家 | 専門性・創造性を要する作業 | **知識の保持** |


専門家へ委譲してよいのは「作業」であり、「質問」ではない。仕様やルールを専門家に**尋ねてはいけない**（→ [知識をエージェントへ閉じ込めない](docs/設計思想.md#知識をエージェントへ閉じ込めない重要)）。


### ツールの粒度


| | 例 |
|---|---|
| **まとめる** | SQLを実行して結果をシートへ書く（間に判断が入らない） |
| **分ける** | 検索して候補を出す → 選んだ候補で本処理（結果を見て次が変わる） |


判断の基準は、**間にLLMの判断が必要かどうか**。不要なら1つにまとめた方が生成回数が減る。


---


## モジュール


すべて `src/statecraft/` の下にある。


| | 責務 |
|---|---|
| `agent.py` | ReActループ、プロンプト構築、委譲 |
| `memory.py` | 記憶の定義、差分の適用、スキーマの自動生成 |
| `prompts.py` | phaseの定義と共通の文言 |
| `tools.py` | 関数のtool化、スキーマ導出、実行制御 |
| `invokable.py` | 「呼び出せるもの」の型（ToolとAgentが独立に満たす） |
| `llm.py` | LLMプロバイダの抽象化（Gemini / OpenAI / Claude） |
| `interceptor.py` | イベントの通知と、実行前の拒否 |
| `network.py` | リクエスト単位の配線と検証 |
| `utils.py` | 共通ユーティリティ |


---


## 実装状況


**動作を確認済み** — Geminiでの1往復、委譲、初期ツール、入力/出力スキーマ、実行前の拒否（機械的な条件とエージェント判定の両方）、記憶への直接書き込み、同時実行、ツールの無効化、停滞の検出、phaseごとのモデル切替、記憶の差分イベント、生成の計測、生成失敗時の再試行判定（待って再試行／枠切れ時のモデル切替）


**未検証** — OpenAI / Claude のプロバイダ実装、会話履歴、添付ファイル、記憶の引き継ぎ


**使う時に踏みうる制限**


- `phase_overrides` で `thought_level` を「指定なし」へ戻せない。`None` が「上書きしない」を意味するため、Agentの既定が `HIGH` の時に特定の phase だけモデルの動的思考へ任せることができない
- `respond()` は生成の例外を外へ出す。`execute()`（委譲）は `ToolResult(error=...)` として受け止めるので、入口だけ扱いが違う
- `tasks` が同じ対象を複数指している場合、実行されなかったタスクが実行済みと判定されうる（判定は「成功した実行の対象名」で行うため）


残りの宿題は [TODO.md](TODO.md)。


---


## ドキュメント


| | 内容 |
|---|---|
| [設計思想.md](docs/設計思想.md) | なぜこの作りなのか。分類の破綻、マルチエージェントの捉え方、知識をエージェントへ閉じ込めない原則 |
| [sampleの読み方.md](docs/sampleの読み方.md) | `sample.py` の全体像と、試せる質問集 |





