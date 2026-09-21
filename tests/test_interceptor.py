"""
Interceptor の検証。

checkは拒否権を持つ「門」なので、既定値の向きが安全側かどうかが全て。
ここが逆を向くと、権限確認の失敗や予算の門が静かに開く。
"""

import threading

from statecraft import Interceptor
from statecraft.prompts import Phase


# ---- check.before_execute（既定は拒否） ----
def test_登録が無ければ許可する():
    verdict = Interceptor().check.before_execute(name="t", kwargs={})

    assert verdict.allowed


def test_Trueを返した時だけ許可する():
    i = Interceptor()
    i.on.before_execute(lambda *, name, kwargs: True)

    assert i.check.before_execute(name="t", kwargs={}).allowed


def test_Noneを返すと拒否になる():
    """returnを書き忘れた場合。許可と解釈すると権限確認の失敗が通る。"""
    i = Interceptor()
    i.on.before_execute(lambda *, name, kwargs: None)

    assert not i.check.before_execute(name="t", kwargs={}).allowed


def test_文字列は拒否の理由として持ち帰られる():
    i = Interceptor()
    i.on.before_execute(lambda *, name, kwargs: "館長の承認が要ります")

    verdict = i.check.before_execute(name="t", kwargs={})

    assert not verdict.allowed
    assert verdict.reason == "館長の承認が要ります"


def test_コールバックの例外は拒否になる():
    i = Interceptor()

    def broken(*, name, kwargs):
        raise RuntimeError("判定の実装ミス")

    i.on.before_execute(broken)
    verdict = i.check.before_execute(name="t", kwargs={})

    assert not verdict.allowed
    # 例外の内容を理由にしない（実装の不具合が外へ出る）
    assert verdict.reason == ""


def test_全員がTrueでなければ拒否になる():
    i = Interceptor()
    i.on.before_execute(lambda *, name, kwargs: True)
    i.on.before_execute(lambda *, name, kwargs: False)

    assert not i.check.before_execute(name="t", kwargs={}).allowed


def test_拒否が決まった後は後続を呼ばない():
    i = Interceptor()
    reached = []
    i.on.before_execute(lambda *, name, kwargs: False)
    i.on.before_execute(lambda *, name, kwargs: reached.append(1) or True)

    i.check.before_execute(name="t", kwargs={})

    assert reached == []


def test_判定には名前と引数が渡る():
    seen = {}
    i = Interceptor()

    def gate(*, name, kwargs):
        seen.update(name=name, kwargs=kwargs)
        return True

    i.on.before_execute(gate)
    i.check.before_execute(name="issue_reservation", kwargs={"title": "深夜特急"})

    assert seen["name"] == "issue_reservation"
    assert seen["kwargs"] == {"title": "深夜特急"}


# ---- check.before_generate（既定は生成する） ----
def test_登録が無ければ生成する():
    """未登録の利用側が動かなくなるのを避けるため、こちらの既定は通す。"""
    decision = Interceptor().check.before_generate(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, step_no=1
    )

    assert decision.proceed


def test_Falseを返すと生成を中止する():
    i = Interceptor()
    i.on.before_generate(lambda **kw: False)

    decision = i.check.before_generate(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, step_no=1
    )

    assert not decision.proceed


def test_文字列はモデル名の差し替えになる():
    i = Interceptor()
    i.on.before_generate(lambda **kw: "gemini-3.1-flash-lite")

    decision = i.check.before_generate(
        agent="a", phase=Phase.ANSWER, model="pro", attempt=1, step_no=1
    )

    assert decision.proceed
    assert decision.model == "gemini-3.1-flash-lite"


def test_生成前の判定で例外が出たら中止する():
    """「生成してよい」と解釈すると、予算の門が不具合で開いてしまう。"""
    i = Interceptor()

    def broken(**kw):
        raise RuntimeError("x")

    i.on.before_generate(broken)
    decision = i.check.before_generate(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, step_no=1
    )

    assert not decision.proceed


# ---- check.generation_failed（既定は再試行しない） ----
def test_登録が無ければ再試行しない():
    """何回試すか・どれだけ待つかは運用の方針なので、フレームワークは決めない。"""
    decision = Interceptor().check.generation_failed(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, error=RuntimeError("x")
    )

    assert not decision.retry


def test_Trueで同じモデルのまま再試行する():
    i = Interceptor()
    i.on.generation_failed(lambda **kw: True)

    decision = i.check.generation_failed(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, error=RuntimeError("x")
    )

    assert decision.retry
    assert decision.model == ""


def test_文字列は別モデルでの再試行になる():
    """枠を使い切ったモデルは待っても同じエラーが返るため。"""
    i = Interceptor()
    i.on.generation_failed(lambda **kw: "flash-lite")

    decision = i.check.generation_failed(
        agent="a", phase=Phase.ANSWER, model="pro", attempt=1, error=RuntimeError("x")
    )

    assert decision.retry
    assert decision.model == "flash-lite"


def test_再試行の判定で例外が出たら諦める():
    """再試行として扱うと、コールバックの不具合が無限の再試行になる。"""
    i = Interceptor()

    def broken(**kw):
        raise RuntimeError("x")

    i.on.generation_failed(broken)
    decision = i.check.generation_failed(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, error=RuntimeError("y")
    )

    assert not decision.retry


def test_再試行の判定には元の例外が渡る():
    seen = {}
    i = Interceptor()

    def decide(*, agent, phase, model, attempt, error):
        seen["error"] = error
        return False

    i.on.generation_failed(decide)
    original = RuntimeError("429 Too Many Requests")
    i.check.generation_failed(
        agent="a", phase=Phase.ANSWER, model="m", attempt=1, error=original
    )

    assert seen["error"] is original


# ---- notify（観測側の不具合で実行を止めない） ----
def test_通知は登録された全員へ届く():
    seen = []
    i = Interceptor()
    i.on.error(seen.append)
    i.on.error(seen.append)

    i.notify.error("何かが起きた")

    assert seen == ["何かが起きた", "何かが起きた"]


def test_通知コールバックの例外は握りつぶす():
    """観測側の不具合で観測対象の実行を止めない。"""
    reached = []
    i = Interceptor()

    def broken(message):
        raise RuntimeError("観測側のバグ")

    i.on.error(broken)
    i.on.error(reached.append)

    i.notify.error("x")  # 例外が外へ出ないこと

    # 前が落ちても後続は呼ばれる
    assert reached == ["x"]


def test_登録が無い通知は何もしない():
    Interceptor().notify.respond_start("x")  # 例外が出ないこと


def test_has_listenersは登録の有無を返す():
    """Networkが「個別に登録済みなら尊重する」判定に使う。"""
    i = Interceptor()
    assert not i.has_listeners

    i.on.error(lambda m: None)
    assert i.has_listeners


# ---- 直列化 ----
def test_同じ種別の通知は1件ずつに直列化される():
    """並列実行で複数スレッドから呼ばれるため。"""
    i = Interceptor()
    overlaps = []
    active = []
    lock = threading.Lock()

    def slow(message):
        with lock:
            active.append(1)
            if len(active) > 1:
                overlaps.append(1)
        threading.Event().wait(0.01)
        with lock:
            active.pop()

    i.on.error(slow)
    threads = [threading.Thread(target=i.notify.error, args=(f"m{n}",)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps == []


# ---- イベント名の対称性（import時に検証される） ----
def test_登録と発火のイベント名が揃っている():
    """
    ズレても実行時エラーにはならず「登録したのに呼ばれない」形で静かに壊れる。
    _verify_events()がimport時に落とすので、importできている時点で揃っている。
    """
    from statecraft.interceptor import _verify_events

    _verify_events()  # 例外が出ないこと
