"""
Agentが持つ「記憶」の定義と、その更新の仕組み。

記憶は倉庫ではなく判断の履歴である。ツールの結果はそのまま保存されず、
LLMが要約して記憶へ書き、元の全文はプロンプトから外れる。

【2種類の記憶】
    SharedMemory   セッション内の全Agentが同じ実体を読み書きする
    PrivateMemory  各Agentが個別に持つ。自分の作業の進行管理だけを入れる

【更新のされ方】
LLMは記憶を直接書き換えず、差分の配列を返す。

    [{"field": "facts", "id": "fact-1", "text": "..."}, ...]
       ↓ apply_diff()
    該当するプロパティへ追記、または同じidがあれば差し替え

差分のJSONスキーマ（build_diff_schema）は記憶の定義から自動生成されるため、
プロパティを増やせばスキーマも追従する。

各プロパティの説明（何を入れる場所か／どう書くか）は
SHARED_MEMORY_GUIDE / PRIVATE_MEMORY_GUIDE が持ち、そのままLLMへ提示される。
"""

import dataclasses
import re
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, ClassVar, get_args, get_origin


@dataclass
class MemoryEntry:
    """
    記憶の1件。ほぼ全てのプロパティがこの形の配列として持たれる。

    idを持たせているのは、LLMが既存の項目を更新できるようにするため。
    同じidで書き込めば差し替えになり、新しいidなら追記になる。
    """

    id: str  # この項目の識別子。LLMが付ける（例: "fact-1"）
    text: str  # 内容そのもの


class TaskStatus(StrEnum):
    """
    タスクの状態。

    Enumにしているのは、文字列のタイプミスが「実行対象が無い」と判断されて
    静かに通るのを防ぐため。

    値はauto()で名前から導出する（NEXT_PARALLEL → "next_parallel"）。
    この値はLLMへ渡すスキーマのenumになり、LLMが書いた文字列の復元にも使う。
    """

    NEXT = auto()  # 次に実行する
    NEXT_PARALLEL = auto()  # 次に、他と同時に実行する（相互依存しない場合）
    CONDITIONAL = auto()  # 前の結果次第で実行する
    DONE = auto()  # 完了した
    UNNECESSARY = auto()  # 前提が崩れた、または他の作業で不要になった


@dataclass
class TaskEntry:
    """
    tasks専用のエントリ。他のプロパティと違い、text以外の情報を構造で持つ。

    tasksは「次に何を実行するか」のシステムへの指示である。
    target_namesに実在するtool / agent名が入った場合だけ、それが提示される
    （自然言語で「規程を確認する」と書いても実行はされない）。
    """

    id: str
    # 「何を判断できるようにするための実行か」という目的を書く。
    # 実行して何が起きたかはactionsへ書き、ここには重複させない。
    text: str
    status: TaskStatus
    # 呼び出すツール名またはエージェント名。同時実行なら複数入る。
    # ここに書かれた名前だけが次のステップで提示される（LLMに選ばせない）。
    target_names: list[str] = field(default_factory=list)


# ---- LLMへ渡す固定文 ----
# Shared/Privateの心構えと、各プロパティのガイド。
# STANCE.shared.read / STANCE.private.write のように名前空間で引く。
@dataclass
class Stance:
    read: str
    write: str


@dataclass
class StanceRegistry:
    shared: Stance
    private: Stance


STANCE = StanceRegistry(
    shared=Stance(
        read="""
SharedMemoryは、このセッションに関わる全エージェントが共有する、たった1つの記憶領域である。
誰かが誰かに報告して伝える形式ではなく、全員が同じ実体を直接読み書きする
「ホワイトボード」だと考えること。

ここに書かれている内容は、たとえ他のエージェントが書いたものであっても、
「自分がすでに知っていること」として読むこと。他人の報告書を参照している
のではない。全エージェントは別々の人格ではなく、同じ1つの理解を共有する、
1つの思考の延長である。
""",
        write="""
この情報が、自分以外のエージェントの判断にも影響しうるならSharedMemoryに書く。
SharedMemoryは全エージェントが毎回読むため、自分だけの作業メモや、他エージェントには
不要な思考過程を書き込まない。
""",
    ),
    private=Stance(
        read="""
PrivateMemoryは、あなた自身だけが持つ、他のエージェントからは見えない記憶領域である。
これまでの自分自身の作業経過や考えの積み重ねとして読むこと。
""",
        write="""
この情報が、自分がタスクを進める上でのみ必要で、他のエージェントには
意味を持たないならPrivateMemoryに書く。
""",
    ),
)

# ---- 書き分けの判断 ----
# ここに1箇所だけ置く。プロパティ間の関係であって各プロパティの性質ではないため、
# 個々のhow_toへは書かない（書くと説明がプロパティ数の二乗で増える）。
ROUTING_GUIDE = """
書き込み先は、内容の性質で決まる。

- 一字一句失われると困る値そのもの（ID・URL・SQL）      → vars
- 今回調べて確定した、この件についての情報              → facts
- 対象がどういう概念・性質・制約で成り立っているか      → back_grounds
- 根拠はあるが、まだ確定していない読み取り              → hypotheses
  （対象についてのものと、依頼者の認識についてのものの両方）
- 自分が実際に行った調査・実行と、その結果              → actions
- 何を採用し、何を採用しなかったかの判断                → decisions
- まだ答えが出ていない論点                              → open_questions
- ここまでを踏まえた、現時点の理解                      → state_briefing
- これから追加で実行する必要のある作業                  → tasks
- 作業には必要だが、外に出す必要が全く無い知識          → task_notes
- 手元のtool/agentを、どう使い、どういう時に呼ばないか  → task_notes

特に取り違えやすい境界:
- facts と back_grounds : 今回調べて分かったことか、以前から一般に成り立つ知識か
- facts と hypotheses   : 根拠が確定しているか、読み取りにとどまるか
- facts と decisions    : 判断の材料か、判断そのものか
- actions と tasks      : すでに行ったことか、これから行うことか
- state_briefing と decisions : 要求に対して何が言えるかの提示か、何を採用したかの決定か
- open_questions と state_briefing : 自分で調べれば埋まるか、相手に聞かないと埋まらないか
- vars と task_notes    : 相手にも渡すべき値か、自分が作業で使うだけの値か
- decisions と task_notes : 依頼をどう解釈したかの判断か、手元のtool/agentをどう使うかの取り決めか

要求と情報を突き合わせて初めて出てくるものは、すべてstate_briefingに書く。
情報そのものではないため、factsやback_groundsには現れない。
  - そのままでは満たせないが、別の手段でなら満たせる（代替案）
  - あと1つ何かが分かれば答えられる（不足の明示）
  - 今回は求められていないが、求められれば対応できる
いずれも提示であって決定ではない。

factsは、判断に使いやすいよう対象ごとの断片として持つ。文章として繋がってはいない。
そのため、factsだけを読んだ相手は論点を追えず、話が飛んで見える。
これを補うのが次の2つで、3つで1組になっている。

  back_grounds   … 対象そのもののモデルを補う（なぜそうなっているのか、他に何が言えるのか）
  state_briefing … 断片どうしの繋がりと現在の理解を補う（なぜその話になっているか）

factsを増やしたときは、この2つが追いついているかを必ず確認する。
factsだけが増えていく状態は、読む相手にとって情報が増えたことにならない。
"""


# システムだけが書き込むプロパティには how_to を持たせない。
# fieldのenumに現れないため書きようがなく、書き方を説明しても
# 守らせる対象が存在しない。読むための説明（description）だけを持つ。
SHARED_MEMORY_GUIDE = {
    "state_briefing": {
        "description": """その時点までに分かったことを踏まえた、現在の理解。時系列で積み上がる。
factsは断片として持つため、それだけを読んだ相手には論点が飛んで見える。断片どうしを繋ぎ、なぜ今この理解に至っているのかを追えるようにするのがここ。
最終的に回答を組み立てる側は、factsの羅列ではなくここを読んで筋を掴む。""",
        "how_to": """毎回、新しいidで1件追加する。既存のidを使って書き換えない（理解がどう変わってきたかが読めなくなる）。変化が無いと感じる場合も、その時点の理解を1件追加する。

1件の中では、次の2つを分けて書く。

【要求に対応する情報】要求のどこに対応するかを添える。
  - 直接答えられること
  - 直接は満たせないが、別の手段でなら満たせること（代替案）
  - あと1つ何かが分かれば答えられること（何が足りないのかを明記する）
【周辺情報】保持している理由を添える。
  - 判断や取り違えの防止に必要な情報
  - 今回は求められていないが、求められれば対応できること

これらは情報そのものではなく、要求と情報を突き合わせて初めて出てくるため、factsやback_groundsには現れない。書かなければ、「できない」で終わった事実だけが読む側へ渡る。

いずれも提示であって決定ではない。採用すると決めた時点で、その判断をdecisionsへ書く。
求められていないことを調べるために追加の実行はしない。調べる過程で分かった場合にだけ書く。
最終回答の文面そのものは書かない。

書けたかどうかは、「これを読んだだけで話の筋が追えるか」で判断する。factsを見に行かないと意味が通らないなら、繋ぐ文脈が足りていない。""",
    },
    "requests": {
        "description": """このセッションで達成すべき要求。全エージェント共通の判断基準であり、依頼が下位のエージェントへ渡っても変わらない。
ここには要求の核（実際に求められている成果）と、解釈のための経緯・背景が混在しうる。要求は核だけである。背景に現れた論点を、明示的に求められていない限り要求として扱わない。

ここに書かれていない用語・概念を補って解釈しない。似た概念、その分野でよく使われる概念であっても、明示されていないものを足すと、別の要求を解いていることになる。
依頼者の表現が曖昧であっても、既知の用語へ言い換えて確定させない。曖昧なまま扱えないなら、何が確定していないのかを明示する。""",
    },
    "vars": {
        "description": "一字一句失われると困る値そのもの。ID・URL・SQL・各種キーなど。",
        "how_to": """推測した値は入れない。idは何の値かが分かる名前にする（url ではなく 仕様書のURL のように）。textにも、値だけでなく何の値かを書く。""",
    },
    "facts": {
        "description": """後続の判断材料になる、確定した情報。
判断に使いやすいよう、対象ごとの断片として持つ。文章として繋がっている必要はない。断片どうしを繋ぐ文脈はstate_briefingが、語の意味はback_groundsが担う。""",
        "how_to": """ここへ書けるのは、確認できたものだけである。
依頼者が述べたことは、それだけでは確認されていない。もっともらしく述べられていても、確認していないなら「分からない」が正確な状態であり、hypothesesへ書く。
依頼者の前提が誤っていることもある。述べられた内容をfactsへ写すと、それ以降は誰も疑わなくなる。

回答文ではなくメモとして書く。である調・体言止めを基本とし、敬体（です・ます）や説明口調（〜してください）は使わない。
主語を必ず書き、何についての情報かが単体で分かるようにする。同じ対象の情報は1件へ統合し、同じ主語の短文を並べない。
例: 「対象機能の指定上限：標準で10件。超過分は無視される。上位プランでは50件。」

【確認できた、と言える根拠の種類を明示する】
「確認できた」の中身は一様ではない。何を根拠にそう言えるのかを、textの中に含める。根拠の種類によって、以降どう扱われるべきかが変わる（一般知識由来のものは、対象への当てはめがまだ済んでいない場合がある。その場合はfactsではなくhypothesesが正しい置き場所になる）。

根拠の種類の例（これに限らない）:
  - そのテキストそのものが存在した（規程・仕様に明記されている）
      例:「規程書に、標準上限は10件と明記されている」
  - 構造・状態からそう判断した（明記はないが、確認した事実から導ける）
      例:「在架の資料は貸出中の表示がないため、予約の対象にならない」
  - 一般知識として、対象がそういうものである（個別の確認ではない）
      例:「禁帯出資料は館内閲覧のみという制度が一般に存在する」
  - データを実際に観測した（ログ・実データを見て確認した）
      例:「昨日のログを確認したところ、実際の値は0件だった」

根拠が薄いまま断定しない。根拠を示せないなら、それはfactsではなくhypothesesである。""",
    },
    "back_grounds": {
        "description": """質問の対象そのものについてのモデル。
語の意味を引くための辞書ではなく、個別の問いに答えるための材料でもない。対象がどういう概念で、どういう性質と制約を持ち、何とどう関係するのかを、その対象の設計を書くつもりで言語化する。

対象のモデルが書けていれば、想定していなかった問いが来ても、そこから導いて答えられる。個別の可否をいくら積み上げても、書かれていない問いには答えられない。問いは常に想定の外から来る、という前提で書く。""",
        "how_to": """対象の名前を明記し、単体で読んで意味が通るように書く。
次の観点で、対象がどう成り立っているのかを言語化する。
  - それが何であるか（定義）
  - どういう単位・属性で構成されるか
  - どういう制約があり、なぜその制約があるか
  - 制約を満たさない場合に用意されている手段
  - どういう状態を取りうるか
  - 何と関係するか。名前が似ているが別物の概念との違い

  不十分:「A形式は受け付けられない」「統合できる単位はB〜C」
    → 個別の可否。並んでいるだけでは、理由も関係も分からず、少しずれた条件を聞かれた時点で答えられない
  モデル:「受け付けの単位は規定値で定められており、規定外は受け付けられない。規定外を扱う場合は、規定内の単位を複数統合して規定に合わせる。統合できる単位には下限と上限があり、その範囲外は統合の対象にならない」
    → 規定外のどんな値を問われても、統合で吸収できるかを導いて判断できる

【何を理解する必要があるか、はここに書かない】
「〜についての理解が必要」「〜を確認する必要がある」は、これから何を調べるべきかを述べているだけで、対象の説明になっていない。
ここに書くのは、調べた結果として分かった対象の性質そのものである。
  不十分:「予約制度や手続き方法についての理解が必要」
    → 何が必要かを述べているだけで、対象について何も説明していない
  モデル:「予約は貸出中の資料に対してのみ受け付ける。在架の資料はその場で借りられるため対象にならない。確保後に一定期間受け取られなければ取り消される」

制約について書く時、「そういう決まりだから」は理由になっていない。その制約が何を守るためにあるのかまで書く。
理由が書けていないと、決まりの文面から少しでも外れた条件を問われた時に答えられず、「規定でそうなっています」としか返せなくなる。

【同じ対象を2件に分けない】
対象ごとに1件へまとめる。書こうとしている内容が既にある項目と重なっているなら、
新しいidで足さず、その項目のidを使って書き換える（分かったことを既存の内容へ足して1件にする）。
説明が2件に分かれていると、読む側はどちらが最新か判断できないまま両方を読むことになる。

前のセッションから引き継がれた項目も対象である。既にあるものと似た内容を書こうとしていると
気付いたら、それは新しい項目ではなく既存項目の更新である。

書けたかどうかは、「この説明だけを読んで、まだ来ていない問いに答えられるか」で判断する。今回の問いに答えるための説明にとどまっているなら、対象のモデルになっていない。
案件ごとに変わる値や、今回の調査で分かった可否は書かない（仕様として定まっている値は、モデルの説明に必要なら含めてよい）。""",
    },
    "actions": {
        "description": "何を確認するために何を行い、その結果どうなったかの記録。",
        "how_to": """ツール名・引数・内部の仕組みは書かず、自然言語で書く。
何も得られなかった試行も省略しない。同じ調査を繰り返さないための材料になる。""",
    },
    "decisions": {
        "description": """何を採用し、何を採用しなかったかの判断履歴。
どこまで調べれば十分か、という判断もここに含む。

一度決めた解釈を固定し、以降のステップで判断をやり直さないための場所でもある。""",
        "how_to": """どの論点について、何を根拠に、何を採用し何を棄却したかを、単体で読んで分かる形で書く。情報のメモにしない。

必ず残す場面:
  - 複数の候補を比較した時
  - 要求をそのまま満たせず、方針を変えた時
  - これ以上調べない、あるいはさらに調べると決めた時
  - 解釈を確定させた時（下記）

【解釈を固定する】
次に何をするかは、毎回その時点の記憶から決め直される。記憶は増え続けるため、同じ問いに毎回同じ答えが出る保証がない。
一度確定した解釈をここへ書いておかないと、2つ目以降の作業で別の解釈に流れ、最初の判断と食い違ったまま進む。

固定するもの:
  - どの論点を誰が担当するのか。なぜそう判断したか
  - 依頼をどの種別のものとして扱うと決めたか
  - 曖昧な語を、どちらの意味として確定させたか
  - 対象の範囲をどこまでと決めたか

「〜と判断した」で終わらせず、「〜であるため、〜として扱う」と、根拠と適用先まで書く。根拠が書かれていないと、次に別の材料が出てきた時に、静かに上書きされる。

固定した解釈は、それを覆す事実が出た時にだけ変える。変えるなら、なぜ変えるのかを新しい判断として書く。黙って別の解釈で進めない。

【調べ足りているかの判断】
これ以上調べないと決めた時は、なぜ十分と言えるのかを書く。
基準は「要求の対象に届いたか」であって、「たくさん調べたか」ではない。
対象に届いているなら、そこで止める。周辺の概念は放っておくと際限なく広がり、要求と関係のない深さまで進む。広げるなら、それが要求のどこに効くのかを書けることを条件にする。

逆に、分かったことが一般論にとどまっていて、それが今回の対象に当てはまるかを確認していない場合は、まだ足りない。
一般論をそのまま個別の答えとして扱うと、成立しているか分からない前提の上で回答することになる。しかもそれは失敗として現れず、断定した回答の形で出ていく。
当てはまるかを確かめられるなら確かめる。確かめられないなら、一般論であることを明示したうえでhypothesesへ書き、確定した事実として扱わない。

特に、「これ以上検証しない」という判断も、ここに含む。検証する材料や手段が無いと分かった場合は、その旨と根拠を書く。

  例:「対象の規定は取得できた。要求は規定そのものについての問い合わせであり、個別の例外も確認されなかったため、実データの観測は不要と判断した。関連する周辺制度まで検索を広げることはしない」
  例:「知識には『指標Xの低下は要因Aによる』とあるが、これは一般論であり、今回の対象が要因Aの条件に該当するかは確認できていない。該当を確かめないまま原因として提示すると、成立していない前提で回答することになるため、実データを観測する方針とした」""",
    },
    "hypotheses": {
        "description": """根拠はあるが、まだ確定していない読み取り。
対象についての読み取りだけでなく、依頼者の認識についての読み取りもここに含む。""",
        "how_to": """推測だと分かる形で書き、何を根拠にそう考えたかを併記する。確定させないまま、それを前提とした作業を始めない。

特に、一般論として成り立つことを今回の対象へ当てはめたものは、確認するまで仮説である。
知識として書かれているからといって、今回の対象がその条件に該当している保証はない。該当を確かめずにfactsへ書くと、以降のすべての判断が、確認されていない前提の上で進む。

【依頼者の認識についての読み取り】
依頼者が取り違えていそうな点も、ここへ書く。
  - 似た概念を混同している可能性
  - 前提としている事実が、調べた内容と食い違っている
  - 原因についての思い込み

何をどう取り違えていそうか、何を見てそう思うのかを書く。取り違えていた場合に要求がどう変わるのかまで書けると、何を確認すべきかが決まる（確認しないと答えられないという状態そのものは、state_briefingへ書く）。

ここへ書いておくことで、回答では「その可能性がある」として扱われる。factsへ書くと、確定していない取り違えを断定して指摘することになる。
取り違えに気付けるのは、似た概念の違いがback_groundsに書かれている場合に限られる。気付けないのは、たいてい違いが言語化されていないため。

これは依頼者を正すために書くのではない。こちらが誤った要求を解かないために書く。取り違えに気付かないまま進むと、正しい手順で別の問いに答えることになる。""",
    },
    "open_questions": {
        "description": "まだ答えが出ていない論点のうち、自分で調べれば解決できるもの。",
        "how_to": """何について、なぜ未解決なのか、何が分かれば解決するのかまで書く。
ここに書くのは、追加で調べれば埋まるものだけにする。
相手に聞かなければ分からないことをここに書くと、どれだけ探しても見つからないものを探し続けることになる。それはstate_briefingへ「これが分かれば答えられる」として書き、回答で相手に尋ねる材料にする。

委譲先が判断すべきことも、ここには書かない。相手の技能や持っている情報で解決することは、自分の不足としてではなくtasksとして委譲する。
「相手が何をどう扱うか分からない」ことを自分の不足として扱うと、委譲すれば済むものを自分で調べ始めるか、手前で止まって不足として返すことになる。

【解決したら書き換える】
答えが出た論点を、未解決のまま残さない。
解決したら同じidで内容を書き換え、解決したことが分かる形にする。
答えそのものはfactsへ書き、ここには解決した事実だけを残す。

  例:「（解決済み）在庫状態は貸出中と確認できた。fact-1を参照」

記憶から項目を消す手段は無いため、書き換えないと残り続ける。
放置すると、次のステップでも未解決の論点として読まれ、すでに答えの出ていることを再び調べ始める。
次のセッションへ引き継いだ場合は、そのまま持ち込まれる。""",
    },
    "agent_answers": {
        "description": """各エージェントが出した回答。単独のエージェントの出力であり、検証されていない結論を含みうる。そのまま最終的な結論として採用せず、判断材料として読む。""",
    },
}

PRIVATE_MEMORY_GUIDE = {
    "goals": {
        "description": """このエージェントが今回達成すべきこと。
依頼された内容をそのまま写す場所ではなく、自分の責務に照らして判定した結果を書く場所である。""",
        "how_to": """まず、求められている成果物が何かを確定する。そのうえで、それが自分の扱えるものかを判定する。

委譲されたことは、自分が担当してよい根拠にならない。扱えないなら、扱えないと判定したこと自体をgoalsにする。
依頼文に含まれる語だけで自分の担当だと判断しない。語が一致していても、求められている成果物が違うことがある。

求められているものを、自分が処理できる形へ書き換えない。近い作業へ置き換えると、求められていないものを正確に作って返すことになる。これは失敗として現れないため、誰も気付けない。

依頼にあることだけを書く。求められていない先回りをしない。論点が複数あるなら分けて持つ。""",
    },
    "tasks": {
        "description": "追加のtool/agent実行が必要な作業のキュー。",
        "how_to": """追加の実行が必要な場合だけ作る。すでに得た情報で答えられるなら作らず、追加実行しないという判断をdecisionsへ残す。
textには「何を判断できるようにするための実行か」という目的を書く。実行して何が起きたかはactionsへ書き、ここでは重複させない。

statusの意味:
  next            … 次に実行する
  next_parallel   … 次に同時実行する。相互に依存しない場合のみ。片方の結果を見ないと対象や引数が決まらないものは同時にしない
  conditional     … 前段の結果次第で実行する
  done            … 完了した
  unnecessary     … 前提が崩れた、または他の作業により不要になった

【statusは必ず更新する】
tasksは記録用のメモではなく、次に何を実行するかをシステムへ指示する場所である。
システムが実行するのはnextとnext_parallelだけで、それ以外は実行されない。

実行した後は、同じidでstatusを更新する。完了したものを未完了のまま残さない。
残したままにすると、次のステップで同じ対象がまた選ばれる。

conditionalにしたタスクは、前提が満たされた時点でnextへ上げる。
これを忘れると、そのタスクは一度も実行されない。
しかも実行対象が無くなった状態として扱われ、そのまま回答へ進む。
「予約する」というタスクをconditionalのまま残したまま、
予約が完了したかのように回答する——という失敗がこれで起きる。
実行されないまま残っているタスクが無いか、毎回必ず確認する。

前提が満たされないと確定した場合は、conditionalのまま放置せずunnecessaryにする。
放置と、実行しないという判断は別のものである。
「確認する」「整理する」だけの曖昧なものや、後で役に立つかもしれないという理由の先回りは作らない。

【委譲するのは作業であって、質問ではない】
委譲先へ「〜とは何か」「〜はどうなっているか」を尋ねるタスクを作らない。
知りたいことがあるなら、それは自分で調べる。相手が詳しそうだから聞く、というのは、自分が持つべき知識を相手に預けていることになる。
相手の責務は、その知識を持つことではなく、自分にはできない作業を行うことである。

渡すのは、作ってほしい成果物と、満たすべき条件。返ってくるべきものは意見ではなく成果物である。
成果物なら、受け取った側が中身を評価して記憶へ書ける。意見だと、それが正しいかを判定する手段がない。

【条件を創作しない】
依頼に存在しない条件を足さない。すでに知っている情報から絞り込みを作ると、依頼された範囲とは違うものが返ってくる。
「そのほうが正確そう」「対象が明確になりそう」は、条件を足す理由にならない。渡すのは、依頼に明示された条件だけにする。

【まとめるか、分けるか】
独立した論点は分ける。逆に、1つの依頼を出力項目ごと・条件ごと・確認手順ごとに刻まない。
「まず確認し、次に本実行する」と分けると、間で文脈が失われ、実行回数だけが増える。分けてよいのは、独立した成果物が複数ある場合か、1回にまとめると意味が壊れる場合だけ。

【制約は適用範囲まで伝える】
使ってよい範囲と使ってはいけない範囲がある情報は、その区別まで書く。
「使うな」とだけ伝えると、途中の処理にも使えないと解釈されるか、逆に最終的な結果にまで現れる。
例:「この値は順位の判定にのみ使い、最終的な出力には含めない」""",
    },
    "task_notes": {
        "description": """tasksを進めるために自分が持っておくべきことのうち、外へ出す必要が全く無いもの。
作業に必要な技術的な詳細と、手元のtool/agentをどう使うかの取り決めを持つ。
他のエージェントからは見えず、共有もされない。""",
        "how_to": """書くのは、作業には必要だが、受け取る相手にとって意味を持たない技術的な詳細である。
例: 参照先のテーブル名やカラム名、内部の識別子、処理の途中で必要になる形式の取り決め。

【tool/agentごとに、使い方を固定する】
自分が持つtool/agentについて、「これはこう使う」「こういう時は呼ばない」を1つずつ書いて固定する。
  例:「ツールXは名称から識別子を引くためのもの。識別子が既に分かっている場合は呼ばない」
  例:「ツールYが返すのは日次の値のみ。期間の合計を求められた場合は、取得した値をこちらで合算せず、範囲を指定して取得し直す」
  例:「エージェントZは成果物の作成専用。仕様や前提についての質問には使わない」
  例:「対象が一意に定まっていない状態で、ツールWは呼ばない」

提示される宣言文は毎ステップ同じだが、実際に使って分かった制約はそこに書かれていない。選べる対象が多いほど、毎回その場で解釈し直すことになり、同じtoolの使い方がステップごとにぶれる。
一度こう使うと決めたら、ここへ書いて読み返す。そうしないと、材料が増えた時に別の使い方へ流れる。

特に「呼ばない条件」を明示する。使えるものが目の前にあると、必要でない場面でも呼ぶ理由が立ってしまう。求められていないことを足しても失敗としては現れず、正確に作られた不要なものは誤りとして検出できないまま外へ出ていく。

【判断のしかた】
次のどちらかに当てはまるなら、ここへ書く。
  - その内容が最終的な回答に現れて困る
  - 自分の作業のための取り決めであり、他のエージェントには意味を持たない

逆に、相手が読んで判断に使う情報であれば、ここではなくsharedの該当プロパティへ書く。ここに書いたものは相手に届かないため、伝えるべき内容を書くと、伝わらないまま作業が進む。

【書き写さない】
ここの内容を、sharedのプロパティへ写さない。写した時点で共有され、外に出さないという前提が失われる。
同じ理由で、最終的な回答や、他のエージェントへ返す成果物にも含めない。

作業に使わないものを念のため残す場所ではない。何のために必要なのかを、内容と一緒に書く。""",
    },
}


# ---- BaseMemory ----
# render / reading_guide / writing_guide の実装を1回だけ書き、Shared/Privateは
# _guide / _reading_stance / _writing_stance を差し替えるだけにする。
@dataclass
class BaseMemory:
    _guide: ClassVar[dict] = {}
    _reading_stance: ClassVar[str] = ""
    _writing_stance: ClassVar[str] = ""
    # システムだけが書き込むプロパティ。LLMからの更新差分は、ここに含まれるものを
    # 無条件で破棄する。プロンプトで「書くな」と依頼するのではなく、
    # 受け取り側で弾くことで構造的に保証する。
    _system_owned: ClassVar[frozenset] = frozenset()

    @classmethod
    def is_writable(cls, field_name: str) -> bool:
        """LLMからの更新を受け付けてよいプロパティかどうか。"""
        return field_name not in cls._system_owned

    @classmethod
    def writable_fields(cls) -> dict:
        """
        LLMが書き込めるプロパティ名と、その要素の型の対応。

        要素の型は list[MemoryEntry] のような注釈から取り出す（対応表を
        別に持つと、フィールドを増やした時に更新漏れが起きる）。

        list以外（Optional[MemoryEntry]のような単一値）は対象にしない。
        get_argsだけで判定すると Optional[X] が (X, NoneType) を返して
        単一値まで含まれ、apply_diffでNoneへ追記してTypeErrorになる。
        """
        result = {}
        for f in dataclasses.fields(cls):
            if not cls.is_writable(f.name):
                continue
            if get_origin(f.type) is not list:
                continue
            result[f.name] = get_args(f.type)[0]
        return result

    @staticmethod
    def _render_entry(e) -> str:
        """1件分のエントリを文字列にする。TaskEntryならstatus/target_namesも表示する。"""
        if isinstance(e, TaskEntry):
            targets = ", ".join(e.target_names) if e.target_names else "(対象なし)"
            return f"  - [{e.id}] ({e.status.value}) {e.text} → {targets}"
        return f"  - [{e.id}] {e.text}"

    def render(self) -> str:
        """
        自分の中身だけを、読みやすいテキストに変換する。空の項目は出さない。

        各プロパティの意味はreading_guide()、書き方はwriting_guide()が返す。
        毎ステップ変わる中身と、変わらない説明を分けてあるのは、同じ場所に
        置くと説明の分までプロンプトキャッシュに乗らなくなるため。
        """
        lines = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not value:
                continue  # 空リスト・Noneはスキップ

            if isinstance(value, list):
                body = "\n".join(self._render_entry(e) for e in value)
            else:  # Optional[MemoryEntry] の単一値（requestsなど）
                body = self._render_entry(value)

            lines.append(f"■ {f.name}")
            lines.append(body)
            lines.append("")

        return "\n".join(lines)

    def reading_guide(self) -> str:
        """
        読む側に必要なもの。心構えと、各プロパティの意味。中身は含まない。

        中身が空でも常に全プロパティを出す。出し分けると、memoryが埋まるたびに
        この文字列が変わり、プロンプトキャッシュに乗らなくなる。
        """
        lines = [self._reading_stance]
        for f in dataclasses.fields(self):
            description = self._guide.get(f.name, {}).get("description")
            if not description:
                continue
            lines.append(f"■ {f.name}")
            lines.append(f"(説明: {description})")
            lines.append("")
        return "\n".join(lines)

    def writing_guide(self) -> str:
        """
        書く側に必要なもの。心構えと、各プロパティへどう書くか。

        プロパティの意味はreading_guide()が既に出しているので繰り返さない。
        読むphaseでは不要なので、書き込みが発生するphaseでだけ使う。
        """
        lines = [self._writing_stance]
        for f in dataclasses.fields(self):
            how_to = self._guide.get(f.name, {}).get("how_to")
            if not how_to:
                continue
            lines.append(f"■ {f.name}")
            lines.append(f"(書き方: {how_to})")
            lines.append("")
        return "\n".join(lines)


@dataclass
class SharedMemory(BaseMemory):
    """
    全エージェントで共有される記憶（ホワイトボード方式）。
    誰かに報告して受け渡すのではなく、全員が同じ実体を直接読み書きする。
    """

    _guide: ClassVar[dict] = SHARED_MEMORY_GUIDE
    _reading_stance: ClassVar[str] = STANCE.shared.read
    _writing_stance: ClassVar[str] = STANCE.shared.write
    # ターンをまたぐ引き継ぎの方針は持たない。どのプロパティを次のターンへ
    # 残すかは、何をセッションとみなすかで変わるため利用側が決める
    # （各プロパティは通常の属性なので、保存して次のターンで代入する）。
    # requests: Frontで構造化された依頼だけが入る。ユーザー原文を共有記憶へ
    #   持ち込まないことでプロンプトインジェクションの経路を断つ。
    # agent_answers: 回答確定時にシステムが記録する。
    _system_owned: ClassVar[frozenset] = frozenset({"requests", "agent_answers"})

    state_briefing: list[MemoryEntry] = field(default_factory=list)  # 理解の変遷を時系列で追記
    requests: MemoryEntry | None = None  # 常に最新の1件のみ（上書き）
    vars: list[MemoryEntry] = field(default_factory=list)  # 不変値（URL/ID/SQL等）
    facts: list[MemoryEntry] = field(default_factory=list)  # 構造化された事実のメモ
    back_grounds: list[MemoryEntry] = field(default_factory=list)  # factsを解釈するための背景知識
    actions: list[MemoryEntry] = field(default_factory=list)  # 実際に行った調査・実行の記録
    decisions: list[MemoryEntry] = field(default_factory=list)  # 採用した判断方針の記録
    hypotheses: list[MemoryEntry] = field(default_factory=list)  # 根拠はあるが未確定の推測
    open_questions: list[MemoryEntry] = field(default_factory=list)  # まだ解決していない論点
    agent_answers: list[MemoryEntry] = field(
        default_factory=list
    )  # 各エージェントの回答（追記専用）


@dataclass
class PrivateMemory(BaseMemory):
    """
    個々のAgentインスタンスだけが持つ記憶。他のAgentとは共有されない。

    持つのは自分の作業の進行管理と、外へ出す必要が全く無い知識だけ。
    判断に使う知識は原則SharedMemoryへ書く（ここに置くと他のエージェントから
    見えず、突き合わせも検証もできない情報が生まれる）。

    task_notesが唯一の例外で、検証性より開示範囲の制御を優先する。

      ・参照先のテーブル名やカラム名のように、作業には必要だが依頼元には
        意味を持たない詳細が、成果物や最終回答へ紛れ込むのを防ぐ
      ・手元のtool/agentをどう使うかを固定し、毎ステップ解釈し直すことで
        使い方がぶれるのを防ぐ

    仕切りとしては最低限のものなので、本当に漏洩を防ぐなら利用側で
    回答内容を検査する。
    """

    _guide: ClassVar[dict] = PRIVATE_MEMORY_GUIDE
    _reading_stance: ClassVar[str] = STANCE.private.read
    _writing_stance: ClassVar[str] = STANCE.private.write

    goals: list[MemoryEntry] = field(default_factory=list)  # このエージェントが達成すべき目標
    tasks: list[TaskEntry] = field(default_factory=list)  # 実行待ちのタスクキュー
    # 作業に必要だが外へ出さない知識（テーブル名・カラム名・内部の識別子など）
    task_notes: list[MemoryEntry] = field(default_factory=list)

    def actionable_tasks(self) -> list[TaskEntry]:
        """
        Function Calling時に実際に呼び出すべきタスク（NEXT/NEXT_PARALLEL）だけを返す。
        DONE/UNNECESSARY/CONDITIONALはここでは無視する
        （state更新時にはrender()で全件を見せるので、そちらで参照される）。
        """
        return [t for t in self.tasks if t.status in (TaskStatus.NEXT, TaskStatus.NEXT_PARALLEL)]


# ---- 差分の適用 ----
# 1行が1件の書き込みで、次の形をしている。
# {"field": "facts", "id": "fact-1", "text": "..."}
# プロパティ名からshared / privateのどちらかは一意に決まるため、
# LLMにスコープ（"shared.facts" のような階層表記）は書かせない。
@dataclass
class DiffResult:
    """
    apply_diffの結果。

    rowsを返すのは、id_prefixで採番したidを呼び出し元へ届けるためである。
    採番は書き込み先のbucketを見て決まるためapply_diffの中でしか行えず、
    呼び出し元が持っているのは採番前の行になる。errorsだけを返すと、
    観測側（memory_diffイベント）にはid=Noneしか見えず、一字一句を正確に
    残すための経路（toolのmemory）で着地点が追えない。

    rowsは採番済みの行。observerへはこちらを渡す。
    """

    errors: list  # 適用に失敗した行。差し戻して直させる
    rows: list  # 実際に適用を試みた行。採番されたidが入っている

    # 反映しなかったが、差し戻しても直せない行（システム所有のプロパティへの
    # 書き込みなど）。errorsと同じDiffErrorで持つのは、違いが「誤りかどうか」
    # ではなく「差し戻して直させるかどうか」だけだから。
    #
    # 分けているのは、ここを捨てると「書いたのに入っていない」が誰にも
    # 見えなくなるため。LLMへは次のステップで1回だけ伝える。
    ignored: list

    # 何番目の適用か（1始まり、プロセス内で単調増加）。
    #
    # 適用は_diff_lockで直列化されるが通知はロックの外なので、観測側が
    # 受け取る順序は適用順と一致しない。差分を保存して後から再適用する用途では
    # 通知順に並べると最後に残る値が逆になるため、この番号で並べ直す。
    apply_seq: int = 0


@dataclass
class DiffError:
    """
    差分の適用に失敗した1件。

    失敗しても例外を投げず、この形で集めて返す。呼び出し側（Agent）は
    これをLLMへ差し戻して「この行だけ直して」と再生成させる。
    1行の失敗で全体を止めないための仕組み。
    """

    field: str  # 書き込み先として指定されていたプロパティ名
    entry: Any  # 失敗した行そのもの（LLMが何を書いたか分かるように保持する）
    message: str  # なぜ失敗したか。LLMが直せるように具体的に書く

    def render(self) -> str:
        """LLMへ差し戻すための1行。プロンプトへそのまま載る。"""
        return f"  - {self.field}: {self.message}（対象: {self.entry}）"


DIFF_EXAMPLE = """[
  {"field": "facts", "id": "fact-1", "text": "対象機能は標準条件では最大10件まで指定可能。"},
  {"field": "actions", "id": "action-1", "text": "上限値を確認するため仕様を照会し、判断材料が揃った。"},
  {"field": "tasks", "id": "task-1", "text": "承認履歴を照会するため",
   "status": "next", "target_names": ["fetch_approval"]}
]"""


# 記憶のプロパティではないが、差分と同じ行の形で受け取る「制御指示」。
#
#   {"field": "disable", "id": "send_email", "text": "今回は使わないため"}
#
# 宣言されたツールは以降のステップで一覧にも候補にも現れなくなる。
#
# 別のトップレベルキー（{"memory": [...], "control": [...]}）にしないのは、
# 差分のルートを配列に保つため。オブジェクトにすると、LLMが実装されていない
# 操作（"remove" など）のキーを勝手に足してくる。
# enumへ1つ足すだけなので、スキーマの形も深さも変わらない。
DISABLE_FIELD = "disable"


def build_diff_schema(*memories, allow_disable: bool = False) -> dict:
    """
    差分のJSON Schema。memoryのインスタンスでもクラスでも受け取れる。

    複数のmemory（shared / private）を渡すと、書き込み可能なプロパティ名を
    まとめて1つのenumにする。

    階層を作らずフラットな行の配列にする。プロパティごとに配列をネストさせる形
    （{"facts": [...], "tasks": [...]}）だと、プロパティ数だけスキーマが深くなり、
    中身を埋めず型だけを返す（[{}]）挙動が出やすい。

    allow_disable: tool/agentの無効化を許すphaseでのみTrue。
        Falseのphaseではenumに現れないため、そもそも指示できない。
    """
    fields = {}
    for m in memories:
        cls = m if isinstance(m, type) else type(m)
        fields.update(cls.writable_fields())

    names = sorted(fields)
    if allow_disable:
        names.append(DISABLE_FIELD)

    # 各フィールドの意味はスキーマ自身に持たせる。プロンプト側で
    # 「idは既存のものを指定すると更新になる」と説明しても、
    # 説明とスキーマが離れている分だけ食い違う余地が残る。
    properties = {
        # 取りうるプロパティ名を列挙する。存在しない名前やシステム所有の
        # プロパティは、そもそも出力できない。
        "field": {
            "type": "string",
            "enum": names,
            "description": "書き込み先のプロパティ名",
        },
        "id": {
            "type": "string",
            "description": """この項目の識別子。既存の項目と同じidを指定すると、その項目の更新になる。新しい項目を追加する場合は、既存と重複しない名前を付ける""",
        },
        "text": {"type": "string", "description": "記録する内容"},
    }

    # tasksを持つmemoryが含まれる時だけ、タスク用のフィールドを足す。
    if any(t is TaskEntry for t in fields.values()):
        properties["status"] = {
            "type": "string",
            "enum": [s.value for s in TaskStatus],
            # tasksの行では実質的に必須。requiredへは入れられない（フラットな配列で
            # 全fieldが同じ形を共有するため、facts等でも必須になってしまう）。
            # 代わりに、指定の無いtasks行はapply_diffが破棄して差し戻す。
            "description": """tasksの場合は必ず指定する、そのタスクの現在の状態。指定しないとその行は破棄される。実行し終えたタスクは必ずdoneへ更新する（nextのまま残すと同じ実行が繰り返される）""",
        }
        properties["target_names"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": """tasksの場合のみ指定する、このタスクで呼び出すtool名またはagent名。提示されている名前だけを指定する。同時実行の場合は複数指定する""",
        }

    # ルートを配列にする。オブジェクトで包むと、そこにキーを置ける余地が生まれ、
    # removeやdeleteのような定義していないキーを捏造してくる。
    # 配列にはキーを置く場所自体が無いので、構造的に起こりえない。
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": properties,
            # statusはtasks以外では不要なため、スキーマ上は必須にできない。
            # tasksでのstatus欠落はapply_diff側でエラーとして検出する。
            "required": ["field", "id", "text"],
        },
    }


def _build_entry(entry_type, raw: dict):
    """1件分の生データからエントリを作る。不正なら例外を投げる。"""
    if not isinstance(raw, dict) or "id" not in raw or "text" not in raw:
        raise ValueError("id と text を持つオブジェクトである必要があります")

    # 空のidを許すと、_upsertが同一idとみなして既存項目を上書きしてしまう。
    # 「memoryは破棄されない」という保証が静かに破れるため、ここで弾く。
    entry_id = str(raw["id"]).strip()
    if not entry_id:
        raise ValueError("idが空です")

    if entry_type is TaskEntry:
        raw_status = raw.get("status")
        try:
            status = TaskStatus(raw_status)
        except ValueError as e:
            valid = " / ".join(s.value for s in TaskStatus)
            # from e で繋げ、Enumの変換で落ちたことを辿れるようにする。
            raise ValueError(f"statusが不正です（指定可能: {valid}）") from e

        # 文字列をlist()に通すと1文字ずつ分解され（"fetch" → ['f','e',...]）、
        # 名前解決が必ず失敗するタスクが静かに生まれる。数値ならlist()自体が
        # TypeErrorになり、ValueErrorしか捕まえない呼び出し側を貫通する。
        raw_targets = raw.get("target_names") or []
        if isinstance(raw_targets, str) or not isinstance(raw_targets, (list, tuple)):
            raise ValueError("target_namesは文字列の配列で指定してください")

        return TaskEntry(
            id=entry_id,
            text=str(raw["text"]),
            status=status,
            target_names=[str(t) for t in raw_targets],
        )

    return MemoryEntry(id=entry_id, text=str(raw["text"]))


def _make_id(bucket: list, prefix: str) -> str:
    """
    まだ使われていないidを作る（"prefix-1", "prefix-2", ... と探す）。

    ツールが返したmemoryを直接書き込む場合に使う（tool自身は既存の記憶を
    知らないため、重複しないidを決められない）。
    prefixにはツール名が入るので、idに使えない文字は _ へ置き換える。
    """
    safe = re.sub(r"[^\w-]", "_", prefix)
    n = 1
    while any(e.id == f"{safe}-{n}" for e in bucket):
        n += 1
    return f"{safe}-{n}"


def _upsert(bucket: list, entry) -> None:
    """
    同じidの項目があれば差し替え、無ければ末尾へ追記する。

    「差し替え」しかしないのが要点で、削除する経路は用意していない。
    LLMが記憶を消せるようにすると、判断の履歴が失われる。
    不要になった情報は、消すのではなく新しい内容で上書きさせる。
    """
    for i, existing in enumerate(bucket):
        if existing.id == entry.id:
            bucket[i] = entry
            return
    bucket.append(entry)


# apply_diffの直列化用。書き込み先はプロセス内のオブジェクトなので、
# 1つのロックで足りる（待たされる時間は配列操作の分だけ）。
_diff_lock = threading.RLock()

# 適用の順番。ロックの中で進めるので、これが実際に適用された順になる。
# 通知の順序は適用順と一致しないため、順序を知りたい側はこちらを使う
# （DiffResult.apply_seq の説明を参照）。
_apply_seq = 0


def apply_diff(
    rows: list,
    *memories: BaseMemory,
    id_prefix: str | None = None,
    handled_disable: bool = False,
) -> DiffResult:
    """
    差分を該当するmemoryへ適用し、失敗した行と採番済みの行を返す。

    rowsは [{"field": ..., "id": ..., "text": ...}, ...] という配列そのもの。

    id_prefix: 指定するとidの無い行に連番を自動採番する。
        tool由来の書き込み（ToolResult.memory）で使う。tool自身は既存のmemoryを
        知らないため、重複しないidを決められない。
        LLM由来の差分ではNoneのままにし、idは必須のままにする
        （既存項目の更新に同じidを使わせる必要があるため）。

    handled_disable: 呼び出し側が無効化指示を既に処理済みなら True。
        Falseのまま無効化の行が来た場合は、指示が消えたことになるので
        ignoredへ載せる（このphaseでは書けない、という誤りとして扱う）。

    - fieldの値から書き込み先のmemoryを特定する
    - システム所有のプロパティへの書き込みは、エラーにせずignoredへ回す
      （差し戻しても直せないため、リトライのトークンを使わせない）
    - 1行の失敗で全体を止めない。通る分は適用し、失敗分だけを返す

    並列に走ったエージェントが同じSharedMemoryを書きうるため、適用の全体を
    ロックで囲んでいる。既存項目の差し替えは「同じidを探して置き換える」
    操作であり、途中で割り込まれると片方の書き込みが消える。
    """
    global _apply_seq
    with _diff_lock:
        result = _apply_diff(
            rows, *memories, id_prefix=id_prefix, handled_disable=handled_disable
        )
        # 採番はロックの中で行う。外に出すと、適用した順と番号の順がずれる。
        _apply_seq += 1
        result.apply_seq = _apply_seq
        return result


def _apply_diff(
    rows: list,
    *memories: BaseMemory,
    id_prefix: str | None = None,
    handled_disable: bool = False,
) -> DiffResult:
    errors = []
    ignored = []
    # 採番後の行を観測側へ渡すために集める。入力の行を書き換えないのは、
    # toolが定数のリストを返した場合に、2回目の呼び出しで前回のidが
    # 残ったまま渡されるため（既存項目の更新として扱われて上書きになる）。
    resolved = []

    # ルートが配列でなければ、1行ずつ処理する前に打ち切る。
    # LLMがオブジェクトを返した場合などが該当する。
    if not isinstance(rows, list):
        return DiffResult([DiffError("(全体)", rows, "配列である必要があります")], [], [])

    for row in rows:
        # --- ① 1行がオブジェクトの形をしているか ---------------------------
        if not isinstance(row, dict):
            errors.append(DiffError("(不明)", row, "オブジェクトである必要があります"))
            continue

        # --- ② 書き込み先の名前が文字列か ----------------------------------
        field_name = row.get("field")
        # 辞書のキーとして使うため、hashできない型が来ると
        # `field_name in fields_map` でTypeErrorになり、
        # 「1行の失敗で全体を止めない」という約束が破れる。
        if not isinstance(field_name, str):
            errors.append(DiffError(str(field_name), row, "fieldは文字列で指定してください"))
            continue

        # --- ③ その名前を持つ記憶を探す ------------------------------------
        # shared / privateを順に見て、書き込み可能なプロパティとして
        # 持っているものと、その要素の型（MemoryEntry / TaskEntry）を得る。
        owner = None
        entry_type = None
        for m in memories:
            fields_map = type(m).writable_fields()
            if field_name in fields_map:
                owner, entry_type = m, fields_map[field_name]
                break

        # --- ④ 制御指示（disable）はここでは扱わない ------------------------
        # 処理済みなら何も言わずに飛ばす。未処理のまま来た場合は、無効化を
        # 書けないphaseで書いたということなので、消えた事実を残す。
        if field_name == DISABLE_FIELD:
            if not handled_disable:
                ignored.append(
                    DiffError(field_name, row, "このphaseでは無効化を指示できません")
                )
            continue

        # --- ⑤ 見つからなかった場合の扱いを分ける --------------------------
        if owner is None:
            # システムだけが書き込むプロパティ（requests / agent_answers）なら、
            # エラーにせずignoredへ回す。LLMへ差し戻しても直せない
            # （そもそも書かせる気がない）ので、リトライのトークンを使わせない。
            if any(not type(m).is_writable(field_name) for m in memories):
                ignored.append(
                    DiffError(
                        str(field_name), row, "システムが管理するプロパティのため書き込めません"
                    )
                )
                continue
            # 本当に存在しない名前なら、直せる誤りなので差し戻す。
            errors.append(DiffError(str(field_name), row, "存在しないプロパティです"))
            continue

        # --- ⑥ 書き込む ----------------------------------------------------
        bucket = getattr(owner, field_name)
        # ツール由来の書き込みでidが無い場合は、ここで採番する。
        # （ツールは既存の記憶を知らないので、重複しないidを決められない）
        if id_prefix and not row.get("id"):
            row = {**row, "id": _make_id(bucket, f"{id_prefix}-{field_name}")}
        resolved.append(row)

        try:
            _upsert(bucket, _build_entry(entry_type, row))
        except ValueError as e:
            # 行の中身が不正だった場合（idが空、statusが不正など）。
            # 例外は外へ出さず、この行だけを失敗として集める。
            errors.append(DiffError(field_name, row, str(e)))

    return DiffResult(errors, resolved, ignored)


