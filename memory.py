import dataclasses
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, get_args, get_origin


@dataclass
class MemoryEntry:
    """全プロパティで共通して使う、id・textの最小単位。"""

    id: str
    text: str


class TaskStatus(Enum):
    """タスクの状態。文字列の直書きだとタイプミスで静かに壊れるためEnumにする。"""

    NEXT = "next"  # 次に実行
    NEXT_PARALLEL = "next_parallel"  # 次に同時実行
    CONDITIONAL = "conditional"  # 条件付き
    DONE = "done"  # 完了
    UNNECESSARY = "unnecessary"  # 不要


@dataclass
class TaskEntry:
    """
    tasks専用のエントリ。MemoryEntryのid/textに加えて、
    状態（status）と呼び出し対象（target_names）を構造化フィールドとして持つ。
    「〜を実行するため」という目的だけをtextに書き、「何が起きたか」の詳細は
    actionsに書く（tasksとactionsで内容を重複させない）。
    """

    id: str
    text: str
    status: TaskStatus
    target_names: list[str] = field(
        default_factory=list
    )  # 呼び出すtool/agent名（同時実行なら複数）


# ==========================================
# Shared/Privateそのものについての心構え、および各プロパティのガイド。
# セッションごとに変わらないフレームワーク固有の固定情報なので、
# SharedMemory/PrivateMemoryのクラス定義より前に、ここへ定数として置く。
#
# STANCE.shared.read / STANCE.shared.write / STANCE.private.read / STANCE.private.write
# という名前空間アクセスで使う。
# ==========================================
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


# ==========================================
# 書き分けの判断は、ここに1箇所だけ置く。
#
# 各プロパティのhow_toへ「これはfactsではなくhypothesesへ」と書いていくと、
# 説明がプロパティ数の二乗で増え、しかも同じ境界を両側から二重に説明することになる。
# 片方だけ直して食い違う、という壊れ方をする。
#
# 「どこへ書くか」は各プロパティの性質ではなく、プロパティ間の関係なので、
# 個々のプロパティではなくmemory全体の説明として持つ。
# ==========================================
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


特に取り違えやすい境界:
- facts と back_grounds : 今回調べて分かったことか、以前から一般に成り立つ知識か
- facts と hypotheses   : 根拠が確定しているか、読み取りにとどまるか
- facts と decisions    : 判断の材料か、判断そのものか
- actions と tasks      : すでに行ったことか、これから行うことか
- state_briefing と decisions : 要求に対して何が言えるかの提示か、何を採用したかの決定か
- open_questions と state_briefing : 自分で調べれば埋まるか、相手に聞かないと埋まらないか


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
        "description": "その時点までに分かったことを踏まえた、現在の理解。時系列で積み上がる。\n"
        "factsは断片として持つため、それだけを読んだ相手には論点が飛んで見える。"
        "断片どうしを繋ぎ、なぜ今この理解に至っているのかを追えるようにするのがここ。\n"
        "最終的に回答を組み立てる側は、factsの羅列ではなくここを読んで筋を掴む。",
        "how_to": "毎回、新しいidで1件追加する。既存のidを使って書き換えない"
        "（理解がどう変わってきたかが読めなくなる）。"
        "変化が無いと感じる場合も、その時点の理解を1件追加する。\n"
        "\n"
        "1件の中では、次の2つを分けて書く。\n"
        "\n"
        "【要求に対応する情報】要求のどこに対応するかを添える。\n"
        "  - 直接答えられること\n"
        "  - 直接は満たせないが、別の手段でなら満たせること（代替案）\n"
        "  - あと1つ何かが分かれば答えられること（何が足りないのかを明記する）\n"
        "【周辺情報】保持している理由を添える。\n"
        "  - 判断や取り違えの防止に必要な情報\n"
        "  - 今回は求められていないが、求められれば対応できること\n"
        "\n"
        "これらは情報そのものではなく、要求と情報を突き合わせて初めて出てくるため、"
        "factsやback_groundsには現れない。"
        "書かなければ、「できない」で終わった事実だけが読む側へ渡る。\n"
        "\n"
        "いずれも提示であって決定ではない。"
        "採用すると決めた時点で、その判断をdecisionsへ書く。\n"
        "求められていないことを調べるために追加の実行はしない。"
        "調べる過程で分かった場合にだけ書く。\n"
        "最終回答の文面そのものは書かない。\n"
        "\n"
        "書けたかどうかは、「これを読んだだけで話の筋が追えるか」で判断する。"
        "factsを見に行かないと意味が通らないなら、繋ぐ文脈が足りていない。",
    },
    "requests": {
        "description": "このセッションで達成すべき要求。全エージェント共通の判断基準であり、"
        "依頼が下位のエージェントへ渡っても変わらない。\n"
        "ここには要求の核（実際に求められている成果）と、"
        "解釈のための経緯・背景が混在しうる。要求は核だけである。"
        "背景に現れた論点を、明示的に求められていない限り要求として扱わない。\n"
        "\n"
        "ここに書かれていない用語・概念を補って解釈しない。"
        "似た概念、その分野でよく使われる概念であっても、"
        "明示されていないものを足すと、別の要求を解いていることになる。\n"
        "依頼者の表現が曖昧であっても、既知の用語へ言い換えて確定させない。"
        "曖昧なまま扱えないなら、何が確定していないのかを明示する。",
    },
    "vars": {
        "description": "一字一句失われると困る値そのもの。ID・URL・SQL・各種キーなど。",
        "how_to": "推測した値は入れない。idは何の値かが分かる名前にする"
        "（url ではなく 仕様書のURL のように）。"
        "textにも、値だけでなく何の値かを書く。",
    },
    "facts": {
        "description": "後続の判断材料になる、確定した情報。\n"
        "判断に使いやすいよう、対象ごとの断片として持つ。"
        "文章として繋がっている必要はない。"
        "断片どうしを繋ぐ文脈はstate_briefingが、語の意味はback_groundsが担う。",
        "how_to": "ここへ書けるのは、確認できたものだけである。\n"
        "依頼者が述べたことは、それだけでは確認されていない。"
        "もっともらしく述べられていても、確認していないなら"
        "「分からない」が正確な状態であり、hypothesesへ書く。\n"
        "依頼者の前提が誤っていることもある。"
        "述べられた内容をfactsへ写すと、それ以降は誰も疑わなくなる。\n"
        "\n"
        "回答文ではなくメモとして書く。である調・体言止めを基本とし、"
        "敬体（です・ます）や説明口調（〜してください）は使わない。\n"
        "主語を必ず書き、何についての情報かが単体で分かるようにする。"
        "同じ対象の情報は1件へ統合し、同じ主語の短文を並べない。\n"
        "例: 「対象機能の指定上限：標準で10件。超過分は無視される。"
        "上位プランでは50件。」\n"
        "\n"
        "【確認できた、と言える根拠の種類を明示する】\n"
        "「確認できた」の中身は一様ではない。何を根拠にそう言えるのかを、"
        "textの中に含める。根拠の種類によって、以降どう扱われるべきかが変わる"
        "（一般知識由来のものは、対象への当てはめがまだ済んでいない場合がある。"
        "その場合はfactsではなくhypothesesが正しい置き場所になる）。\n"
        "\n"
        "根拠の種類の例（これに限らない）:\n"
        "  - そのテキストそのものが存在した（規程・仕様に明記されている）\n"
        "      例:「規程書に、標準上限は10件と明記されている」\n"
        "  - 構造・状態からそう判断した（明記はないが、確認した事実から導ける）\n"
        "      例:「在架の資料は貸出中の表示がないため、予約の対象にならない」\n"
        "  - 一般知識として、対象がそういうものである（個別の確認ではない）\n"
        "      例:「禁帯出資料は館内閲覧のみという制度が一般に存在する」\n"
        "  - データを実際に観測した（ログ・実データを見て確認した）\n"
        "      例:「昨日のログを確認したところ、実際の値は0件だった」\n"
        "\n"
        "根拠が薄いまま断定しない。根拠を示せないなら、"
        "それはfactsではなくhypothesesである。",
    },
    "back_grounds": {
        "description": "質問の対象そのものについてのモデル。\n"
        "語の意味を引くための辞書ではなく、個別の問いに答えるための材料でもない。"
        "対象がどういう概念で、どういう性質と制約を持ち、"
        "何とどう関係するのかを、その対象の設計を書くつもりで言語化する。\n"
        "\n"
        "対象のモデルが書けていれば、想定していなかった問いが来ても、"
        "そこから導いて答えられる。個別の可否をいくら積み上げても、"
        "書かれていない問いには答えられない。"
        "問いは常に想定の外から来る、という前提で書く。",
        "how_to": "対象の名前を明記し、単体で読んで意味が通るように書く。\n"
        "次の観点で、対象がどう成り立っているのかを言語化する。\n"
        "  - それが何であるか（定義）\n"
        "  - どういう単位・属性で構成されるか\n"
        "  - どういう制約があり、なぜその制約があるか\n"
        "  - 制約を満たさない場合に用意されている手段\n"
        "  - どういう状態を取りうるか\n"
        "  - 何と関係するか。名前が似ているが別物の概念との違い\n"
        "\n"
        "  不十分:「A形式は受け付けられない」「統合できる単位はB〜C」\n"
        "    → 個別の可否。並んでいるだけでは、理由も関係も分からず、"
        "少しずれた条件を聞かれた時点で答えられない\n"
        "  モデル:「受け付けの単位は規定値で定められており、規定外は受け付けられない。"
        "規定外を扱う場合は、規定内の単位を複数統合して規定に合わせる。"
        "統合できる単位には下限と上限があり、その範囲外は統合の対象にならない」\n"
        "    → 規定外のどんな値を問われても、統合で吸収できるかを導いて判断できる\n"
        "\n"
        "制約について書く時、「そういう決まりだから」は理由になっていない。"
        "その制約が何を守るためにあるのかまで書く。\n"
        "理由が書けていないと、決まりの文面から少しでも外れた条件を問われた時に"
        "答えられず、「規定でそうなっています」としか返せなくなる。\n"
        "\n"
        "書けたかどうかは、「この説明だけを読んで、まだ来ていない問いに答えられるか」"
        "で判断する。今回の問いに答えるための説明にとどまっているなら、"
        "対象のモデルになっていない。\n"
        "案件ごとに変わる値や、今回の調査で分かった可否は書かない"
        "（仕様として定まっている値は、モデルの説明に必要なら含めてよい）。",
    },
    "actions": {
        "description": "何を確認するために何を行い、その結果どうなったかの記録。",
        "how_to": "ツール名・引数・内部の仕組みは書かず、自然言語で書く。\n"
        "何も得られなかった試行も省略しない。同じ調査を繰り返さないための材料になる。",
    },
    "decisions": {
        "description": "何を採用し、何を採用しなかったかの判断履歴。\n"
        "どこまで調べれば十分か、という判断もここに含む。\n"
        "\n"
        "一度決めた解釈を固定し、"
        "以降のステップで判断をやり直さないための場所でもある。",
        "how_to": "どの論点について、何を根拠に、何を採用し何を棄却したかを、"
        "単体で読んで分かる形で書く。情報のメモにしない。\n"
        "\n"
        "必ず残す場面:\n"
        "  - 複数の候補を比較した時\n"
        "  - 要求をそのまま満たせず、方針を変えた時\n"
        "  - これ以上調べない、あるいはさらに調べると決めた時\n"
        "  - 解釈を確定させた時（下記）\n"
        "\n"
        "【解釈を固定する】\n"
        "次に何をするかは、毎回その時点の記憶から決め直される。"
        "記憶は増え続けるため、同じ問いに毎回同じ答えが出る保証がない。\n"
        "一度確定した解釈をここへ書いておかないと、"
        "2つ目以降の作業で別の解釈に流れ、"
        "最初の判断と食い違ったまま進む。\n"
        "\n"
        "固定するもの:\n"
        "  - どの論点を誰が担当するのか。なぜそう判断したか\n"
        "  - 依頼をどの種別のものとして扱うと決めたか\n"
        "  - 曖昧な語を、どちらの意味として確定させたか\n"
        "  - 対象の範囲をどこまでと決めたか\n"
        "\n"
        "「〜と判断した」で終わらせず、"
        "「〜であるため、〜として扱う」と、根拠と適用先まで書く。"
        "根拠が書かれていないと、"
        "次に別の材料が出てきた時に、静かに上書きされる。\n"
        "\n"
        "固定した解釈は、それを覆す事実が出た時にだけ変える。"
        "変えるなら、なぜ変えるのかを新しい判断として書く。"
        "黙って別の解釈で進めない。\n"
        "\n"
        "【調べ足りているかの判断】\n"
        "これ以上調べないと決めた時は、なぜ十分と言えるのかを書く。\n"
        "基準は「要求の対象に届いたか」であって、「たくさん調べたか」ではない。\n"
        "対象に届いているなら、そこで止める。周辺の概念は放っておくと際限なく広がり、"
        "要求と関係のない深さまで進む。広げるなら、"
        "それが要求のどこに効くのかを書けることを条件にする。\n"
        "\n"
        "逆に、分かったことが一般論にとどまっていて、"
        "それが今回の対象に当てはまるかを確認していない場合は、まだ足りない。\n"
        "一般論をそのまま個別の答えとして扱うと、"
        "成立しているか分からない前提の上で回答することになる。"
        "しかもそれは失敗として現れず、断定した回答の形で出ていく。\n"
        "当てはまるかを確かめられるなら確かめる。確かめられないなら、"
        "一般論であることを明示したうえでhypothesesへ書き、確定した事実として扱わない。\n"
        "\n"
        "特に、「これ以上検証しない」という判断も、ここに含む。"
        "検証する材料や手段が無いと分かった場合は、その旨と根拠を書く。\n"
        "\n"
        "  例:「対象の規定は取得できた。要求は規定そのものについての問い合わせであり、"
        "個別の例外も確認されなかったため、実データの観測は不要と判断した。"
        "関連する周辺制度まで検索を広げることはしない」\n"
        "  例:「知識には『指標Xの低下は要因Aによる』とあるが、これは一般論であり、"
        "今回の対象が要因Aの条件に該当するかは確認できていない。"
        "該当を確かめないまま原因として提示すると、成立していない前提で回答することに"
        "なるため、実データを観測する方針とした」",
    },
    "hypotheses": {
        "description": "根拠はあるが、まだ確定していない読み取り。\n"
        "対象についての読み取りだけでなく、"
        "依頼者の認識についての読み取りもここに含む。",
        "how_to": "推測だと分かる形で書き、何を根拠にそう考えたかを併記する。"
        "確定させないまま、それを前提とした作業を始めない。\n"
        "\n"
        "特に、一般論として成り立つことを今回の対象へ当てはめたものは、"
        "確認するまで仮説である。\n"
        "知識として書かれているからといって、今回の対象がその条件に"
        "該当している保証はない。該当を確かめずにfactsへ書くと、"
        "以降のすべての判断が、確認されていない前提の上で進む。\n"
        "\n"
        "【依頼者の認識についての読み取り】\n"
        "依頼者が取り違えていそうな点も、ここへ書く。\n"
        "  - 似た概念を混同している可能性\n"
        "  - 前提としている事実が、調べた内容と食い違っている\n"
        "  - 原因についての思い込み\n"
        "\n"
        "何をどう取り違えていそうか、何を見てそう思うのかを書く。"
        "取り違えていた場合に要求がどう変わるのかまで書けると、"
        "何を確認すべきかが決まる"
        "（確認しないと答えられないという状態そのものは、state_briefingへ書く）。\n"
        "\n"
        "ここへ書いておくことで、回答では「その可能性がある」として扱われる。"
        "factsへ書くと、確定していない取り違えを断定して指摘することになる。\n"
        "取り違えに気付けるのは、似た概念の違いがback_groundsに"
        "書かれている場合に限られる。"
        "気付けないのは、たいてい違いが言語化されていないため。\n"
        "\n"
        "これは依頼者を正すために書くのではない。"
        "こちらが誤った要求を解かないために書く。"
        "取り違えに気付かないまま進むと、正しい手順で別の問いに答えることになる。",
    },
    "open_questions": {
        "description": "まだ答えが出ていない論点のうち、自分で調べれば解決できるもの。",
        "how_to": "何について、なぜ未解決なのか、何が分かれば解決するのかまで書く。\n"
        "ここに書くのは、追加で調べれば埋まるものだけにする。\n"
        "相手に聞かなければ分からないことをここに書くと、"
        "どれだけ探しても見つからないものを探し続けることになる。"
        "それはstate_briefingへ「これが分かれば答えられる」として書き、"
        "回答で相手に尋ねる材料にする。\n"
        "\n"
        "委譲先が判断すべきことも、ここには書かない。"
        "相手の技能や持っている情報で解決することは、"
        "自分の不足としてではなくtasksとして委譲する。\n"
        "「相手が何をどう扱うか分からない」ことを自分の不足として扱うと、"
        "委譲すれば済むものを自分で調べ始めるか、"
        "手前で止まって不足として返すことになる。",
    },
    "agent_answers": {
        "description": "各エージェントが出した回答。単独のエージェントの出力であり、"
        "検証されていない結論を含みうる。"
        "そのまま最終的な結論として採用せず、判断材料として読む。",
    },
}


PRIVATE_MEMORY_GUIDE = {
    "goals": {
        "description": "このエージェントが今回達成すべきこと。\n"
        "依頼された内容をそのまま写す場所ではなく、"
        "自分の責務に照らして判定した結果を書く場所である。",
        "how_to": "まず、求められている成果物が何かを確定する。"
        "そのうえで、それが自分の扱えるものかを判定する。\n"
        "\n"
        "委譲されたことは、自分が担当してよい根拠にならない。"
        "扱えないなら、扱えないと判定したこと自体をgoalsにする。\n"
        "依頼文に含まれる語だけで自分の担当だと判断しない。"
        "語が一致していても、求められている成果物が違うことがある。\n"
        "\n"
        "求められているものを、自分が処理できる形へ書き換えない。"
        "近い作業へ置き換えると、"
        "求められていないものを正確に作って返すことになる。"
        "これは失敗として現れないため、誰も気付けない。\n"
        "\n"
        "依頼にあることだけを書く。求められていない先回りをしない。"
        "論点が複数あるなら分けて持つ。",
    },
    "tasks": {
        "description": "追加のtool/agent実行が必要な作業のキュー。",
        "how_to": "追加の実行が必要な場合だけ作る。すでに得た情報で答えられるなら作らず、"
        "追加実行しないという判断をdecisionsへ残す。\n"
        "textには「何を判断できるようにするための実行か」という目的を書く。"
        "実行して何が起きたかはactionsへ書き、ここでは重複させない。\n"
        "\n"
        "statusの意味:\n"
        "  next            … 次に実行する\n"
        "  next_parallel   … 次に同時実行する。相互に依存しない場合のみ。"
        "片方の結果を見ないと対象や引数が決まらないものは同時にしない\n"
        "  conditional     … 前段の結果次第で実行する\n"
        "  done            … 完了した\n"
        "  unnecessary     … 前提が崩れた、または他の作業により不要になった\n"
        "\n"
        "実行した後は、同じidでstatusを更新する。完了したものを未完了のまま残さない。\n"
        "「確認する」「整理する」だけの曖昧なものや、"
        "後で役に立つかもしれないという理由の先回りは作らない。\n"
        "\n"
        "【委譲するのは作業であって、質問ではない】\n"
        "委譲先へ「〜とは何か」「〜はどうなっているか」を尋ねるタスクを作らない。\n"
        "知りたいことがあるなら、それは自分で調べる。"
        "相手が詳しそうだから聞く、というのは、"
        "自分が持つべき知識を相手に預けていることになる。\n"
        "相手の責務は、その知識を持つことではなく、"
        "自分にはできない作業を行うことである。\n"
        "\n"
        "渡すのは、作ってほしい成果物と、満たすべき条件。"
        "返ってくるべきものは意見ではなく成果物である。\n"
        "成果物なら、受け取った側が中身を評価して記憶へ書ける。"
        "意見だと、それが正しいかを判定する手段がない。\n"
        "\n"
        "【条件を創作しない】\n"
        "依頼に存在しない条件を足さない。"
        "すでに知っている情報から絞り込みを作ると、"
        "依頼された範囲とは違うものが返ってくる。\n"
        "「そのほうが正確そう」「対象が明確になりそう」は、条件を足す理由にならない。"
        "渡すのは、依頼に明示された条件だけにする。\n"
        "\n"
        "【まとめるか、分けるか】\n"
        "独立した論点は分ける。逆に、1つの依頼を"
        "出力項目ごと・条件ごと・確認手順ごとに刻まない。\n"
        "「まず確認し、次に本実行する」と分けると、"
        "間で文脈が失われ、実行回数だけが増える。"
        "分けてよいのは、独立した成果物が複数ある場合か、"
        "1回にまとめると意味が壊れる場合だけ。\n"
        "\n"
        "【制約は適用範囲まで伝える】\n"
        "使ってよい範囲と使ってはいけない範囲がある情報は、その区別まで書く。\n"
        "「使うな」とだけ伝えると、途中の処理にも使えないと解釈されるか、"
        "逆に最終的な結果にまで現れる。\n"
        "例:「この値は順位の判定にのみ使い、最終的な出力には含めない」",
    },
    "notes": {
        "description": "作業を進めるうえで保持しておくべき情報",
        "how_to": "他エージェントや回答には関係しないが、今後自身のタスクを進めるうえで必要になるナレッジなどはここに記載する"
    }
}


# ==========================================
# BaseMemory: render/introの実装はここに1回だけ書く。
# SharedMemory/PrivateMemoryはこれを継承し、自分の_guide/_reading_stance/
# _writing_stanceを指すだけで、両方のメソッドをそのまま使える。
# ==========================================
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


        要素の型は list[MemoryEntry] のような注釈から取り出す。
        「tasksだけはTaskEntry」といった対応表を別に持つと、
        フィールドを増やした時に更新漏れが起きるため、注釈を唯一の正とする。


        list以外（Optional[MemoryEntry]のような単一値）は差分更新の対象にしない。
        get_argsだけで判定すると Optional[X] は (X, NoneType) を返して空にならず、
        単一値フィールドまで対象に含まれてしまう。その状態でapply_diffを通すと、
        Noneに対して追記しようとしてTypeErrorになる。
        """
        result = {}
        for f in dataclasses.fields(cls):
            if not cls.is_writable(f.name):
                continue
            if get_origin(f.type) is not list:
                continue
            result[f.name] = get_args(f.type)[0]
        return result

    @classmethod
    def diff_schema(cls) -> dict:
        """
        更新差分のJSON Schema。llm.generate(response_schema=...)へ渡す。


        プロパティで階層を作らず、フラットな行の配列にする。
        プロパティごとに配列をネストさせる形（{"facts": [...], "tasks": [...]}）だと、
        プロパティ数だけスキーマが深く大きくなり、モデルが中身を埋めずに
        型だけをオウム返しする（[{}] しか返らない）挙動を起こしやすい。


        フラットにすればプロパティが増えてもスキーマの大きさは変わらない。
        書き込み先はfieldで指定させ、取りうる値をenumで列挙することで、
        存在しないプロパティ名やシステム所有のプロパティは出力できなくなる。
        """
        return build_diff_schema(cls)

    @staticmethod
    def _render_entry(e) -> str:
        """1件分のエントリを文字列にする。TaskEntryならstatus/target_namesも表示する。"""
        if isinstance(e, TaskEntry):
            targets = ", ".join(e.target_names) if e.target_names else "(対象なし)"
            return f"  - [{e.id}] ({e.status.value}) {e.text} → {targets}"
        return f"  - [{e.id}] {e.text}"

    def render(self, *, include_how_to: bool = True) -> str:
        """
        自分の中身を、ガイド文言と合わせて読みやすいテキストに変換する。
        空のプロパティは出力しない。


        include_how_to: Trueなら「書き方」の指示まで含める
            （初期構築・更新など、書き込みが発生する場面向け）。
            Falseなら説明のみにする
            （Function Calling時など、読むだけでよい場面向け。トークン節約）。
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

            info = self._guide.get(f.name, {})
            lines.append(f"■ {f.name}")
            if info.get("description"):
                lines.append(f"(説明: {info['description']})")
            if include_how_to and info.get("how_to"):
                lines.append(f"(書き方: {info['how_to']})")
            lines.append(body)
            lines.append("")

        return "\n".join(lines)

    def intro(self, *, include_how_to: bool = True) -> str:
        """
        読む時の心構え（常に含める）と、書く時の心構え
        （include_how_to=Trueの時だけ含める）を組み合わせて返す。
        """
        if include_how_to:
            return self._reading_stance + "\n" + self._writing_stance
        return self._reading_stance


@dataclass
class SharedMemory(BaseMemory):
    """
    全エージェントで共有される記憶（ホワイトボード方式）。
    誰かに報告して受け渡すのではなく、全員が同じ実体を直接読み書きする。
    """

    _guide: ClassVar[dict] = SHARED_MEMORY_GUIDE
    _reading_stance: ClassVar[str] = STANCE.shared.read
    _writing_stance: ClassVar[str] = STANCE.shared.write
    # ターンをまたぐ引き継ぎの方針は、ここには持たない。
    # どのプロパティを次のターンへ残すかは、そのアプリケーションが
    # 何をセッションとみなすかで変わるため、利用側が決める。
    #
    # 各プロパティは通常の属性として公開されているので、
    # 取り出して保存し、次のターンで代入すればよい（README参照）。
    # フレームワークが既定を持つと、その既定に合わない使い方が
    # 「間違った使い方」に見えてしまう。
    # requests: Frontで構造化された依頼だけが入る。ユーザー原文を共有記憶へ
    #   持ち込まないことでプロンプトインジェクションの経路を断つ。
    # agent_answers: 回答確定時にシステムが記録する。
    _system_owned: ClassVar[frozenset] = frozenset({"requests", "agent_answers"})

    state_briefing: list[MemoryEntry] = field(
        default_factory=list
    )  # 理解の変遷を時系列で追記
    requests: Optional[MemoryEntry] = None  # 常に最新の1件のみ（上書き）
    vars: list[MemoryEntry] = field(default_factory=list)  # 不変値（URL/ID/SQL等）
    facts: list[MemoryEntry] = field(default_factory=list)  # 構造化された事実のメモ
    back_grounds: list[MemoryEntry] = field(
        default_factory=list
    )  # factsを解釈するための背景知識
    actions: list[MemoryEntry] = field(
        default_factory=list
    )  # 実際に行った調査・実行の記録
    decisions: list[MemoryEntry] = field(default_factory=list)  # 採用した判断方針の記録
    hypotheses: list[MemoryEntry] = field(
        default_factory=list
    )  # 根拠はあるが未確定の推測
    open_questions: list[MemoryEntry] = field(
        default_factory=list
    )  # まだ解決していない論点
    agent_answers: list[MemoryEntry] = field(
        default_factory=list
    )  # 各エージェントの回答（追記専用）


@dataclass
class PrivateMemory(BaseMemory):
    """
    個々のAgentインスタンスだけが持つ記憶。他のAgentとは共有されない。


    持つのは「自分の作業の進行管理」だけで、知識は一切持たない。
    知識を置ける場所をここに用意すると、そのエージェントの内側にしか無い情報が
    生まれる。他のエージェントから見えず、突き合わせも検証もできない状態は、
    知識をエージェントへ閉じ込めないという原則と正面から衝突する。


    「自分だけが知っていればよい知識」という区分は作らない。
    知識は全てSharedMemoryへ書き、判断の根拠を1箇所に保つ。
    """

    _guide: ClassVar[dict] = PRIVATE_MEMORY_GUIDE
    _reading_stance: ClassVar[str] = STANCE.private.read
    _writing_stance: ClassVar[str] = STANCE.private.write

    goals: list[MemoryEntry] = field(
        default_factory=list
    )  # このエージェントが達成すべき目標
    tasks: list[TaskEntry] = field(default_factory=list)  # 実行待ちのタスクキュー

    def actionable_tasks(self) -> list[TaskEntry]:
        """
        Function Calling時に実際に呼び出すべきタスク（NEXT/NEXT_PARALLEL）だけを返す。
        DONE/UNNECESSARY/CONDITIONALはここでは無視する
        （state更新時にはrender()で全件を見せるので、そちらで参照される）。
        """
        return [
            t
            for t in self.tasks
            if t.status in (TaskStatus.NEXT, TaskStatus.NEXT_PARALLEL)
        ]


# ==========================================
# 差分の適用
#
# LLMは {プロパティ名: [エントリ, ...]} という形の差分を返す。
# shared / private のどちらに属するかはプロパティ名から一意に決まるため、
# LLMにスコープを書かせない（GAS版の "global.facts" のような階層キーは不要）。
# ==========================================
@dataclass
class DiffError:
    """差分の適用に失敗した1件。LLMへ差し戻す材料になる。"""

    field: str
    entry: Any
    message: str

    def render(self) -> str:
        return f"  - {self.field}: {self.message}（対象: {self.entry}）"


DIFF_EXAMPLE = """[
  {"field": "facts", "id": "fact-1", "text": "対象機能は標準条件では最大10件まで指定可能。"},
  {"field": "actions", "id": "action-1", "text": "上限値を確認するため仕様を照会し、判断材料が揃った。"},
  {"field": "tasks", "id": "task-1", "text": "承認履歴を照会するため",
   "status": "next", "target_names": ["fetch_approval"]}
]"""


# memoryのプロパティではないが、同じ行の形で受け取る制御指示。
# tool/agentの無効化をここへ載せる。
#
# GAS版は差分JSONへ _runtime というトップキーを足していたが、
# ルートを配列にしてキーを置く場所を無くした設計（キー捏造の防止）と両立しない。
# fieldのenumへ1つ足すだけなら、スキーマの形も階層も変わらない。
DISABLE_FIELD = "disable"


def build_diff_schema(*memories, allow_disable: bool = False) -> dict:
    """
    差分のJSON Schema。memoryのインスタンスでもクラスでも受け取れる。


    複数のmemory（shared / private）を渡すと、書き込み可能なプロパティ名を
    まとめて1つのenumにする。どちらに属するかはプロパティ名から一意に決まるため、
    LLMにスコープを書かせる必要がない。


    allow_disable: tool/agentの無効化を許すphaseでのみTrueにする。
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
            "description": "この項目の識別子。既存の項目と同じidを指定すると、その項目の更新になる。"
            "新しい項目を追加する場合は、既存と重複しない名前を付ける",
        },
        "text": {"type": "string", "description": "記録する内容"},
    }

    # tasksを持つmemoryが含まれる時だけ、タスク用のフィールドを足す。
    if any(t is TaskEntry for t in fields.values()):
        properties["status"] = {
            "type": "string",
            "enum": [s.value for s in TaskStatus],
            "description": "tasksの場合のみ指定する、そのタスクの現在の状態",
        }
        properties["target_names"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "tasksの場合のみ指定する、このタスクで呼び出すtool名またはagent名。"
            "提示されている名前だけを指定する。同時実行の場合は複数指定する",
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
        except ValueError:
            valid = " / ".join(s.value for s in TaskStatus)
            raise ValueError(f"statusが不正です（指定可能: {valid}）")

        # target_namesは配列でなければならない。
        # 文字列をlist()に通すと1文字ずつ分解され（"fetch" → ['f','e','t','c','h']）、
        # エラーにならないまま名前解決が必ず失敗するタスクが生まれる。
        # 数値を渡された場合はlist()自体がTypeErrorになり、ValueErrorしか
        # 捕まえていない呼び出し側を貫通する。どちらもここで弾く。
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
    重複しないidを作る。tool由来の書き込みでは、tool自身がidを決められない
    （既存のmemoryを知らない）ため、システム側で連番を振る。
    """
    safe = re.sub(r"[^\w-]", "_", prefix)
    n = 1
    while any(e.id == f"{safe}-{n}" for e in bucket):
        n += 1
    return f"{safe}-{n}"


def _upsert(bucket: list, entry) -> None:
    """同じidがあれば差し替え、無ければ追記する。memoryは破棄されない。"""
    for i, existing in enumerate(bucket):
        if existing.id == entry.id:
            bucket[i] = entry
            return
    bucket.append(entry)


def apply_diff(
    rows: list,
    *memories: BaseMemory,
    id_prefix: Optional[str] = None,
) -> list[DiffError]:
    """
    差分を該当するmemoryへ適用し、失敗した行だけを返す。


    rowsは [{"field": ..., "id": ..., "text": ...}, ...] という配列そのもの。


    id_prefix: 指定するとidの無い行に連番を自動採番する。
        tool由来の書き込み（write_to_memory）で使う。tool自身は既存のmemoryを
        知らないため、重複しないidを決められない。
        LLM由来の差分ではNoneのままにし、idは必須のままにする
        （既存項目の更新に同じidを使わせる必要があるため）。


    - fieldの値から書き込み先のmemoryを特定する
    - システム所有のプロパティへの書き込みは、エラーにせず黙って捨てる。
      LLMに差し戻しても直せない（そもそも書かせる気がない）ため、
      リトライのトークンを消費させない
    - 1行の失敗で全体を止めない。通る分は適用し、失敗分だけを返して
      次の生成で修正させる
    """
    errors = []

    if not isinstance(rows, list):
        return [DiffError("(全体)", rows, "配列である必要があります")]

    for row in rows:
        if not isinstance(row, dict):
            errors.append(DiffError("(不明)", row, "オブジェクトである必要があります"))
            continue

        field_name = row.get("field")
        # 辞書のキーとして使うため、hashできない型（list等）が来ると
        # `field_name in fields_map` の時点でTypeErrorになり、
        # 「1行の失敗で全体を止めない」という保証が破れる。
        if not isinstance(field_name, str):
            errors.append(
                DiffError(str(field_name), row, "fieldは文字列で指定してください")
            )
            continue

        owner = None
        entry_type = None
        for m in memories:
            fields_map = type(m).writable_fields()
            if field_name in fields_map:
                owner, entry_type = m, fields_map[field_name]
                break

        # 制御指示はmemoryへの書き込みではないため、ここでは扱わない。
        # 呼び出し側が事前に取り出す前提だが、取り残された場合に
        # 「存在しないプロパティ」と誤って差し戻さないよう明示的に無視する。
        if field_name == DISABLE_FIELD:
            continue

        if owner is None:
            # システム所有なら黙って破棄、本当に存在しない名前ならエラーとして返す。
            if any(not type(m).is_writable(field_name) for m in memories):
                continue
            errors.append(DiffError(str(field_name), row, "存在しないプロパティです"))
            continue

        bucket = getattr(owner, field_name)
        if id_prefix and not row.get("id"):
            row = {**row, "id": _make_id(bucket, f"{id_prefix}-{field_name}")}

        try:
            _upsert(bucket, _build_entry(entry_type, row))
        except ValueError as e:
            errors.append(DiffError(field_name, row, str(e)))

    return errors
