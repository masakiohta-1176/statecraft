"""
実行の観測と、実行前の拒否を担う仕組み。


ツールやエージェントが実行される前後にコールバックを差し込める。
用途は2つあり、性質がまったく違う。


    notify（通知）… 起きたことを伝えるだけ。戻り値は見ない
    check（判定）  … 実行してよいかを尋ねる。拒否権を持つ


使い方は「種類 → イベント名」の2段階になっている。


    interceptor.on.execute_start(callback)          # 登録する
    interceptor.notify.execute_start("...")         # 通知を出す（Tool/Agentが呼ぶ）
    interceptor.check.before_execute(name=, kwargs=)  # 判定を仰ぐ（Tool/Agentが呼ぶ）


利用側が書くのは1行目だけで、2行目以降はフレームワークが呼ぶ。


【何に使えるか】
    ・進捗を利用者の画面へ出す（execute_start）
    ・記憶がどう作られたかを検証用に記録する（memory_diff）
    ・特定の条件下で実行を止める（before_execute）
    ・異常をログへ残す（error）


【文言を持たない】
Interceptorは通知の文言を一切持たない。「何を伝えるか」はツールや
エージェント自身が組み立て、Interceptorはそれを配るだけ。
だから同じイベントに対して、利用者向けの表示と開発者向けのログを
別々に登録できる（誰に何を見せるかは購読する側が決める）。
"""


import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass


logger = logging.getLogger(__name__)




@dataclass
class CheckResult:
    """
    判定の結果。許可されたかと、拒否した場合の理由を持つ。


    boolだけを返す形にしていないのは、拒否の理由を実行側へ返せるようにするため。
    理由が返らないと、止められた側は「ブロックされた」しか分からず、
    原因を探すために追加の実行を始めてしまう。
    引数の形式が違うといった直せる誤りでも、それが伝わらない。


    ただし理由を返すかどうかは、拒否する側が選べるようにしてある。
    審査基準そのものが理由になる場合、それを伝えると基準を回避されるため。
    """


    allowed: bool
    # 拒否した理由。空文字なら理由を伝えない拒否。
    # ここに書いた内容はLLMへ渡るため、そのまま読まれる前提で書く。
    reason: str = ""




@dataclass
class RetryDecision:
    """
    生成が失敗した時の判定の結果。もう一度やるか、その時どのモデルでやるか。


    CheckResultと分けているのは、「許可/拒否」では表せないため。
    再試行は許可の有無ではなく、同じ条件でやり直すのか、
    条件を変えてやり直すのかという選択になる。


    枠を使い切ったモデルは、待っても同じエラーが返る。
    その場合に選べる手はモデルを変えることだけなので、
    判定する側がモデル名を返せる必要がある。
    """


    retry: bool
    # 再試行に使うモデル名。空文字なら元のモデルのまま。
    model: str = ""




def _event_name(depth: int = 1) -> str:
    """
    呼び出し元のメソッド名を、イベント名として返す。


    イベント名として、そのメソッド自身の名前をそのまま使うために用いる。


        def execute_start(self, callback):
            self._register_here(callback)   # → "execute_start" として登録される


    メソッド名とイベント名の文字列を両方書くと、同じ情報が2箇所に存在する。
    しかもイベント名は登録側（on）と発火側（notify / check）の両方に現れるため、
    実際には3箇所へ散る。片方を直し忘れると、登録したイベントと
    発火するイベントの名前がずれ、コールバックが呼ばれなくなる。
    エラーにはならず「なぜか通知が来ない」という形で現れる。
    checkの場合はさらに悪く、判定が一度も呼ばれず全部許可されてしまう。


    メソッド名から導出すれば、名前は定義の1箇所にしか存在しなくなる。


    depth: 何段さかのぼるか。
        1 … この関数を呼んだ関数の名前（イベント名を持つメソッドが直接呼ぶ場合）
        2 … さらに1つ上（_register_hereのような中継を1段はさむ場合）


    フレームを数える実装のため、呼び出しの形を変えるとdepthも変わる。
    直接呼ぶ場合と中継を挟む場合で値が違うので、
    呼び出し側がどちらなのかを明示する。
    """
    frame = inspect.currentframe()
    # currentframe()はこの関数自身のフレーム。そこからdepth回さかのぼる。
    for _ in range(depth):
        if frame is None:
            return ""
        frame = frame.f_back
    return frame.f_code.co_name if frame else ""




class _OnRegistry:
    """
    イベントの登録専用の名前空間。interceptor.on.xxx() の形で使う。


    イベント名を文字列で受け取らずメソッドとして定義するのは、
    補完が効き、タイポが実行前に検出されるようにするため。
    文字列キーを受け取る作りだと interceptor.on("tool_strat", cb) が
    静かに何もしない（登録はされるが、誰もそのイベントを発火しない）。


    各メソッドの中身は同じで、自分のメソッド名をイベント名として登録するだけ。
    """


    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor


    def _register_here(self, callback: Callable) -> None:
        """呼び出し元のメソッド名をイベント名として登録する。"""
        # depth=2: この関数 → 呼び出し元のメソッド（execute_startなど）
        self._interceptor._register(_event_name(depth=2), callback)


    # ---- 実行（tool / agent 共通） ----
    def before_execute(self, callback: Callable[..., bool | str]) -> None:
        """
        実行してよいかを判定するコールバックを登録する。


        戻り値の意味:
            True   … 許可する
            文字列 … 拒否する。その文字列が理由として実行側へ返る
            それ以外（False / None / 戻り値なし / 例外）… 理由なしで拒否する


        観測だけを行う場合も、必ず True を返すこと。
        登録された全てのコールバックが True を返した時だけ実行が許可される。
        """
        self._register_here(callback)


    def generation_failed(self, callback: Callable[..., bool | str]) -> None:
        """
        生成が例外で失敗した時に、再試行するかどうかを判定する。


        コールバックは
            agent: str, phase, model: str, attempt: int, error: Exception
        をキーワード引数で受け取る（attemptは1始まり。今回が何回目か）。


        戻り値の意味:
            True   … 同じモデルでもう一度生成する
            文字列 … そのモデル名でもう一度生成する
            それ以外（False / None / 例外）… 諦める。例外はそのまま外へ出る


        文字列がモデル名を意味するのはこのイベントだけ。before_executeでは
        文字列は「拒否の理由」になる。同じ名前空間で意味が違うのは、
        判定の性質が違うため（実行してよいかと、次にどうするか）。


        枠を使い切ったモデルは、待っても同じエラーが返る。その場合に
        選べる手はモデルを変えることだけなので、モデル名を返せるようにしてある。


        待ってから再試行したい場合は、コールバックの中で待つ。
        このコールバックだけは複数スレッドから同時に呼ばれる場合がある
        （待つ処理を直列化すると、無関係な実行まで止まってしまうため）。
        「503なら3秒待って2回まで」は運用の方針であり、
        対話UIなら即座に諦めた方がよく、バッチなら長く待った方がよい。
        フレームワークがその方針を持たない。


        before_execute と違い、登録が無い場合は「再試行しない」になる。
        判定が無いことを許可と解釈すると、誰も登録していない状態で
        失敗が永久に再試行されることになるため。
        """
        self._register_here(callback)


    def execute_start(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    def execute_end(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    def execute_blocked(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    # ---- 応答（respondの境界） ----
    def respond_start(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    def respond_end(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    # ---- 生成 ----
    def generated(self, callback: Callable[..., None]) -> None:
        """
        LLMの生成1回ごとに、かかった時間とトークン数を構造のまま受け取る。


        合計だけを見ても、どのAgentのどのphaseが重いのかは分からない。
        遅い原因が生成回数か1回の重さかも、内訳がないと切り分けられない。


        引数の型（GenerationEvent）はagentモジュール側が定義する。
        Interceptorが具体的な型を知ると依存が逆流するため、ここでは受け取るだけ。
        """
        self._register_here(callback)


    # ---- 記憶 ----
    def memory_updated(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)


    def memory_diff(self, callback: Callable[..., None]) -> None:
        """
        記憶の差分を1回の生成ごとに構造のまま受け取る。検証・評価のための経路。


        memory_updatedが「人間が読む1行」なのに対し、こちらは
        「この入力に対して、この差分を出し、こう適用された」を機械で扱える形で渡す。
        文字列を組み立ててから解析し直す必要がないようにする。


        引数の型（MemoryDiffEvent）はagentモジュール側が定義する。
        Interceptorが具体的な型を知ると依存が逆流するため、ここでは受け取るだけ。
        """
        self._register_here(callback)


    # ---- 異常 ----
    def error(self, callback: Callable[[str], None]) -> None:
        self._register_here(callback)




class _CheckDispatcher:
    """
    拒否権を持つ判定の名前空間。interceptor.check.xxx() の形で使う。


    _OnRegistryと同じく、イベント名はメソッド名から導出する。
    こちらは登録側（on）と対になる発火側なので、名前がずれると
    「登録したのに一度も判定が呼ばれない」＝全部許可されてしまう。
    """


    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor


    def before_execute(self, *, name: str, kwargs: dict) -> CheckResult:
        """
        tool / agent の実行前判定。ToolとAgentで同じイベントを使う。
        Invokableとして両者を同一に扱う以上、観測側でも区別する理由がない。
        """
        # depth=1: このメソッド（before_execute）自身の名前
        return self._interceptor._check(_event_name(), name=name, kwargs=kwargs)


    def generation_failed(
        self, *, agent: str, phase, model: str, attempt: int, error: Exception
    ) -> RetryDecision:
        """
        生成が失敗した時の再試行判定。


        登録が無ければ再試行しない。before_execute とは既定が逆になる。
        どちらも安全側だが、安全側の向きが違う——実行の審査は
        「止める人がいなければ実行する」が正しく、再試行は
        「やると言う人がいなければやらない」が正しい。
        """
        return self._interceptor._decide_retry(
            _event_name(),
            agent=agent,
            phase=phase,
            model=model,
            attempt=attempt,
            error=error,
        )




class _NotifyDispatcher:
    """
    純粋な通知の名前空間。interceptor.notify.xxx() の形で使う。


    各メソッドは自分のメソッド名をイベント名として発火するだけ。
    戻り値は見ない（通知に拒否権は無い）。
    """


    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor


    def _notify_here(self, *args) -> None:
        """呼び出し元のメソッド名をイベント名として発火する。"""
        # depth=2: この関数 → 呼び出し元のメソッド（execute_startなど）
        self._interceptor._notify(_event_name(depth=2), *args)


    def execute_start(self, message: str) -> None:
        self._notify_here(message)


    def execute_end(self, message: str) -> None:
        self._notify_here(message)


    def execute_blocked(self, message: str) -> None:
        self._notify_here(message)


    def respond_start(self, message: str) -> None:
        self._notify_here(message)


    def respond_end(self, message: str) -> None:
        self._notify_here(message)


    def generated(self, event) -> None:
        # 文言ではなく数値なので、構造のまま渡す。整形は受け取る側が決める。
        self._notify_here(event)


    def memory_updated(self, message: str) -> None:
        self._notify_here(message)


    def memory_diff(self, event) -> None:
        """記憶の差分を構造のまま流す。登録がなければ何も起きない。"""
        self._notify_here(event)


    def error(self, message: str) -> None:
        """
        処理は継続するが記録すべき異常。


        LLMへ返すエラー文（expose_error_detailsで隠される場合がある）とは別に、
        観測者へは実際に起きたことをそのまま伝える。
        隠す相手はLLMであって、システムを運用する人間ではない。
        """
        self._notify_here(message)




class Interceptor:
    """
    Tool/Agentが発生させるイベントの登録・発火を一元管理する。


    使い方はすべて「種類（on/check/notify）」→「イベント名」の2段階：
        interceptor.on.execute_start(callback)
        interceptor.check.before_execute(name=..., kwargs=...)
        interceptor.notify.execute_start(message)


    文言はInterceptor自身は一切持たない。Tool/Agentが自分の言葉で
    組み立てたメッセージを、notify/checkに渡すだけ。


    【並列実行との関係】
    next_parallelで複数の呼び出しが同時に走る場合、notify/checkは複数の
    スレッドから呼ばれる。そのためコールバックの呼び出しは1件ずつに直列化
    してある（同時には走らない）。
    登録する側は、自分のコールバックが複数スレッドから同時に呼ばれることを
    考えなくてよい。ただし呼ばれる順序は実行の終了順になるため、
    「前の通知の内容」に依存する書き方はできない。
    """


    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}
        # コールバックの直列化用。コールバックの中からさらに通知が出る場合が
        # あるため、同一スレッドで再取得できるRLockを使う。
        self._lock = threading.RLock()
        self.on = _OnRegistry(self)
        self.check = _CheckDispatcher(self)
        self.notify = _NotifyDispatcher(self)


    @property
    def has_listeners(self) -> bool:
        """コールバックが1つでも登録されているか。Networkが配線時に参照する。"""
        return any(self._listeners.values())


    # ---- 内部の汎用実装（on/check/notifyの名前空間クラスからのみ呼ばれる） ----
    def _register(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)


    def _notify(self, event: str, *args, **kwargs) -> None:
        """
        通知を発火する。


        コールバック内の例外は握りつぶす。通知は観測のための仕組みであり、
        観測側の不具合（ログ出力の失敗、UI描画の失敗など）が
        観測対象の実行を止めてよい理由がない。
        ただし黙って消すと原因が追えないため、ログには残す。
        """
        with self._lock:
            for callback in self._listeners.get(event, []):
                try:
                    callback(*args, **kwargs)
                except Exception:
                    logger.exception("通知コールバックが例外を送出しました: event=%s", event)


    def _decide_retry(self, event: str, *args, **kwargs) -> RetryDecision:
        """
        失敗した後にどうするかの判定。_checkとは既定も戻り値の解釈も違う。


        _checkの既定（登録が無ければ許可）は「止める人がいなければ実行する」
        という意味で、実行前の審査には正しい。
        一方、再試行は「やると言う人がいなければやらない」が正しい。
        登録が無い状態でやると答えると、誰も止められないまま繰り返される。


        登録順に尋ね、Trueか文字列を返したものがあればそこで決まる。
        全てがそれ以外を返した場合、および例外を送出した場合は再試行しない。


        ここだけはコールバックを直列化しない。待つ処理はコールバックの中に
        書かれるため、直列化すると並列で走っている他の実行が、
        自分と無関係な待ち時間の分まで止まることになる。
        """
        for callback in self._listeners.get(event, []):
            try:
                verdict = callback(*args, **kwargs)
            except Exception:
                logger.exception("判定コールバックが例外を送出しました: event=%s", event)
                # 判定できなかったものを再試行として扱うと、
                # コールバックの不具合が無限の再試行になる。
                return RetryDecision(retry=False)
            if verdict is True:
                return RetryDecision(retry=True)
            if isinstance(verdict, str) and verdict:
                return RetryDecision(retry=True, model=verdict)
        return RetryDecision(retry=False)


    def _check(self, event: str, *args, **kwargs) -> CheckResult:
        """
        拒否権のある判定。


        安全側に倒す：明示的に True を返したものだけ許可する。
        None・書き忘れ・その他の値はすべて「拒否」として扱う。


        例外も拒否として扱う。判定できなかったものを許可すると、
        権限確認の失敗が「許可」として通ってしまう。


        文字列を返した場合も拒否だが、その文字列を理由として持ち帰る。
        「True かどうか」で許可を判定しているため、理由を返せるように
        しても許可の条件は変わらない（文字列は True ではない）。
        """
        with self._lock:
            for callback in self._listeners.get(event, []):
                try:
                    verdict = callback(*args, **kwargs)
                except Exception:
                    logger.exception("判定コールバックが例外を送出しました: event=%s", event)
                    # 例外の内容は理由にしない。実装の不具合が外へ出るのを防ぐ。
                    return CheckResult(allowed=False)
                if verdict is True:
                    continue
                # ここへ来たものはすべて拒否。文字列なら理由として扱う。
                return CheckResult(
                    allowed=False, reason=verdict if isinstance(verdict, str) else ""
                )
            return CheckResult(allowed=True)





