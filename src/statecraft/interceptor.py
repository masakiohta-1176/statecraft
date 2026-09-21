"""
実行の観測と、実行前の拒否。

    interceptor.on.execute_start(callback)            # 利用側が登録する
    interceptor.notify.execute_start("...")           # フレームワークが通知する
    interceptor.check.before_execute(name=, kwargs=)  # フレームワークが判定を仰ぐ

notifyは起きたことを伝えるだけ。checkは拒否権を持つ。
イベントの一覧と、各コールバックの引数・戻り値はREADMEを参照。

イベント名は各メソッドが文字列で明示し、_verify_events()がimport時に
名前の一致と3面の対称性を確かめる（ズレると静かに呼ばれなくなるため）。
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # 実行時にimportしない。イベントの型はagent側が定義しており、
    # ここで本当にimportすると依存が逆流する（agent → interceptor）。
    from .agent import ExecuteEvent, GenerationEvent, MemoryDiffEvent
    from .prompts import Phase

logger = logging.getLogger(__name__)


# ---- 判定の結果 ----
@dataclass
class CheckResult:
    """
    実行前の判定の結果。

    理由を返せるのは、無いと止められた側が原因を探して追加の実行を始めるため。
    返すかどうかは拒否する側が選べる（審査基準を伝えると回避される）。
    """

    allowed: bool
    # 拒否した理由。空文字なら理由を伝えない拒否。LLMへそのまま渡る。
    reason: str = ""


@dataclass
class RetryDecision:
    """
    生成が失敗した時の判定の結果。

    枠を使い切ったモデルは待っても同じエラーが返るため、モデル名を返せる。
    """

    retry: bool
    # 再試行に使うモデル名。空文字なら元のモデルのまま。
    model: str = ""


@dataclass
class GenerateDecision:
    """生成の前の判定の結果。RetryDecisionと形は同じだが意味が違う。"""

    proceed: bool
    # 生成に使うモデル名。空文字なら元のモデルのまま。
    model: str = ""


# ---- コールバックの形 ----
# 判定（check）は引数がイベントごとに違う。Protocolにしておくと利用側が書いた
# 関数の形を型チェッカーが見られる（Callable[..., bool] では引数が消える）。
class BeforeExecute(Protocol):
    """Trueだけが許可。文字列は拒否の理由としてLLMへ渡る。"""

    def __call__(self, *, name: str, kwargs: dict) -> bool | str | None: ...


class BeforeGenerate(Protocol):
    """Trueでそのまま生成。文字列はそのモデル名で生成。それ以外は中止。"""

    def __call__(
        self, *, agent: str, phase: "Phase", model: str, attempt: int, step_no: int
    ) -> bool | str | None: ...


class GenerationFailed(Protocol):
    """Trueで同じモデルで再試行。文字列はそのモデル名で再試行。それ以外は諦める。"""

    def __call__(
        self, *, agent: str, phase: "Phase", model: str, attempt: int, error: Exception
    ) -> bool | str | None: ...


# ---- 登録・通知・判定の3つの名前空間 ----
# 文字列キーの窓口（on("tool_strat", cb)）にしないのは、タイポが静かに何も
# しない状態になるため。メソッドなら補完が効き、誤りが実行前に分かる。
class _OnRegistry:
    """イベントの登録。interceptor.on.xxx(callback) の形で使う。"""

    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor

    # ---- 実行（tool / agent 共通） ----
    def before_execute(self, callback: BeforeExecute) -> None:
        """
        実行してよいかを判定する。

        登録された全てがTrueを返した時だけ実行される。観測だけを行う場合も
        必ずTrueを返すこと（None・書き忘れ・例外はすべて拒否になる）。
        """
        self._interceptor._register("before_execute", callback)

    def generation_failed(self, callback: GenerationFailed) -> None:
        """
        生成が例外で失敗した時に、再試行するかどうかを判定する。

        文字列がモデル名を意味するのはこのイベントだけ。登録が無い場合は
        再試行しない。待つ処理を書けるよう、直列化されない。
        """
        self._interceptor._register("generation_failed", callback)

    def before_generate(self, callback: BeforeGenerate) -> None:
        """
        生成を行う前に、行うかどうかを判定する。登録が無ければ生成する。

        用途は走行中の状態を見て決めること（phase_overridesは構築時の静的な
        割り当てなので「10万トークン使ったから落とす」は書けない）。門を生成に
        置くのは、費用の大きいMEMORY_UPDATEがbefore_executeの外にあるため。
        generation_failedと同じく直列化されない。
        """
        self._interceptor._register("before_generate", callback)

    def execute_start(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("execute_start", callback)

    def execute_end(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("execute_end", callback)

    def execute_blocked(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("execute_blocked", callback)

    def executed(self, callback: Callable[["ExecuteEvent"], None]) -> None:
        """
        実行1回ごとに、引数と戻り値を構造のまま受け取る。

        呼び出した側から発火するため、ループを経由しない直接のexecute()
        呼び出しでは流れない。
        """
        self._interceptor._register("executed", callback)

    # ---- 応答（respondの境界） ----
    def respond_start(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("respond_start", callback)

    def respond_end(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("respond_end", callback)

    # ---- 生成 ----
    def generated(self, callback: Callable[["GenerationEvent"], None]) -> None:
        """LLMの生成1回ごとに、かかった時間とトークン数を受け取る。"""
        self._interceptor._register("generated", callback)

    # ---- 記憶 ----
    def memory_updated(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("memory_updated", callback)

    def memory_diff(self, callback: Callable[["MemoryDiffEvent"], None]) -> None:
        """記憶の差分を1回の生成ごとに受け取る。検証・評価のための経路。"""
        self._interceptor._register("memory_diff", callback)

    # ---- 異常 ----
    def error(self, callback: Callable[[str], None]) -> None:
        self._interceptor._register("error", callback)


class _CheckDispatcher:
    """拒否権を持つ判定。interceptor.check.xxx() の形でフレームワークが呼ぶ。"""

    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor

    def before_execute(self, *, name: str, kwargs: dict) -> CheckResult:
        """tool / agent の実行前判定。Invokableとして同一に扱うため共通。"""
        return self._interceptor._check("before_execute", name=name, kwargs=kwargs)

    def before_generate(
        self, *, agent: str, phase: "Phase", model: str, attempt: int, step_no: int
    ) -> GenerateDecision:
        """生成の前の判定。登録が無ければ生成する。"""
        return self._interceptor._decide_generate(
            "before_generate",
            agent=agent,
            phase=phase,
            model=model,
            attempt=attempt,
            step_no=step_no,
        )

    def generation_failed(
        self, *, agent: str, phase: "Phase", model: str, attempt: int, error: Exception
    ) -> RetryDecision:
        """生成が失敗した時の再試行判定。登録が無ければ再試行しない。"""
        return self._interceptor._decide_retry(
            "generation_failed",
            agent=agent,
            phase=phase,
            model=model,
            attempt=attempt,
            error=error,
        )


class _NotifyDispatcher:
    """通知。interceptor.notify.xxx() の形でフレームワークが呼ぶ。戻り値は見ない。"""

    def __init__(self, interceptor: "Interceptor"):
        self._interceptor = interceptor

    def execute_start(self, message: str) -> None:
        self._interceptor._notify("execute_start", message)

    def execute_end(self, message: str) -> None:
        self._interceptor._notify("execute_end", message)

    def execute_blocked(self, message: str) -> None:
        self._interceptor._notify("execute_blocked", message)

    def executed(self, event: "ExecuteEvent") -> None:
        self._interceptor._notify("executed", event)

    def respond_start(self, message: str) -> None:
        self._interceptor._notify("respond_start", message)

    def respond_end(self, message: str) -> None:
        self._interceptor._notify("respond_end", message)

    def generated(self, event: "GenerationEvent") -> None:
        self._interceptor._notify("generated", event)

    def memory_updated(self, message: str) -> None:
        self._interceptor._notify("memory_updated", message)

    def memory_diff(self, event: "MemoryDiffEvent") -> None:
        self._interceptor._notify("memory_diff", event)

    def error(self, message: str) -> None:
        """
        処理は継続するが記録すべき異常。

        LLMへ返すエラー文（expose_error_detailsで隠れる場合がある）とは別に、
        観測者へは実際に起きたことを伝える。
        """
        self._interceptor._notify("error", message)


def _verify_events() -> None:
    """
    3つの名前空間の整合をimport時に確かめる。メソッドが自分の名前を渡して
    いるか、on と（check | notify）で名前が揃っているか、同じ名前がcheckと
    notifyの両方に無いか。

    どれもズレても実行時エラーにはならず「登録したのに呼ばれない」という形で
    静かに壊れるため、ここで落とす。
    """
    faces: dict[str, set[str]] = {}
    for face, cls in (
        ("on", _OnRegistry),
        ("check", _CheckDispatcher),
        ("notify", _NotifyDispatcher),
    ):
        names = set()
        for name, attr in vars(cls).items():
            code = getattr(attr, "__code__", None)
            if name.startswith("_") or code is None:
                continue
            if name not in code.co_consts:
                raise ImportError(
                    f"{cls.__name__}.{name} がイベント名 '{name}' を渡していません。"
                    f"メソッド名と渡す文字列を一致させてください。"
                )
            names.add(name)
        faces[face] = names

    fired = faces["check"] | faces["notify"]
    if faces["on"] != fired:
        raise ImportError(
            f"イベント名が揃っていません。"
            f"登録だけあるもの: {sorted(faces['on'] - fired)} / "
            f"発火だけあるもの: {sorted(fired - faces['on'])}"
        )
    if both := faces["check"] & faces["notify"]:
        raise ImportError(f"checkとnotifyに同じイベント名があります: {sorted(both)}")


_verify_events()


class Interceptor:
    """
    イベントの登録・発火を一元管理する。文言は持たず、Tool/Agentが
    組み立てたものを配るだけ。

    並列実行では複数のスレッドから呼ばれる。同じ種別のコールバックは1件ずつに
    直列化してあるが、順序は実行の終了順なので「前の通知の内容」には依存できない。
    notifyとcheckは互いに直列化されず（別のロック）、generation_failedは
    どちらとも直列化されない。
    """

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}
        # 別のロックにするのは、checkが人の承認を待つ場合に無関係な実行の通知まで
        # 止めないため。コールバックの中からさらに通知が出るためRLock。
        self._notify_lock = threading.RLock()
        self._check_lock = threading.RLock()
        self.on = _OnRegistry(self)
        self.check = _CheckDispatcher(self)
        self.notify = _NotifyDispatcher(self)

    @property
    def has_listeners(self) -> bool:
        """コールバックが1つでも登録されているか。Networkが配線時に参照する。"""
        return any(self._listeners.values())

    # ---- 3つの名前空間からのみ呼ばれる ----
    def _register(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def _notify(self, event: str, *args) -> None:
        """
        通知を発火する。

        コールバック内の例外は握りつぶす（観測側の不具合で観測対象の実行を
        止めない）。黙って消すと原因が追えないため、ログには残す。
        """
        with self._notify_lock:
            for callback in self._listeners.get(event, []):
                try:
                    callback(*args)
                except Exception:
                    logger.exception("通知コールバックが例外を送出しました: event=%s", event)

    def _decide_retry(self, event: str, **kwargs) -> RetryDecision:
        """
        失敗した後にどうするかの判定。登録順に尋ね、Trueか文字列を返したものが
        あればそこで決まる。それ以外と例外、登録が無い場合は再試行しない。

        直列化しない（待つ処理がコールバックの中に書かれるため、直列化すると
        他の実行が無関係な待ち時間まで待たされる）。
        """
        for callback in self._listeners.get(event, []):
            try:
                verdict = callback(**kwargs)
            except Exception:
                logger.exception("判定コールバックが例外を送出しました: event=%s", event)
                # 再試行として扱うと、コールバックの不具合が無限の再試行になる。
                return RetryDecision(retry=False)
            if verdict is True:
                return RetryDecision(retry=True)
            if isinstance(verdict, str) and verdict:
                return RetryDecision(retry=True, model=verdict)
        return RetryDecision(retry=False)

    def _decide_generate(self, event: str, **kwargs) -> GenerateDecision:
        """
        生成の前の判定。_decide_retryと形は同じだが既定が逆で、登録が無ければ
        生成する（未登録の利用側が動かなくなるのを避ける）。ロックは取らない。
        """
        for callback in self._listeners.get(event, []):
            try:
                verdict = callback(**kwargs)
            except Exception:
                logger.exception("判定コールバックが例外を送出しました: event=%s", event)
                # 「生成してよい」と解釈すると、予算の門が不具合で開く。
                return GenerateDecision(proceed=False)
            if verdict is True:
                continue
            if isinstance(verdict, str) and verdict:
                return GenerateDecision(proceed=True, model=verdict)
            return GenerateDecision(proceed=False)
        return GenerateDecision(proceed=True)

    def _check(self, event: str, **kwargs) -> CheckResult:
        """
        拒否権のある判定。明示的にTrueを返したものだけ許可し、None・書き忘れ・
        例外はすべて拒否（許可と解釈すると権限確認の失敗が通る）。
        文字列も拒否で、その文字列を理由として持ち帰る。

        checkだけのロックを取るので、人の承認を待つあいだも通知は流れ続ける。
        """
        with self._check_lock:
            for callback in self._listeners.get(event, []):
                try:
                    verdict = callback(**kwargs)
                except Exception:
                    logger.exception("判定コールバックが例外を送出しました: event=%s", event)
                    # 例外の内容は理由にしない（実装の不具合が外へ出る）。
                    return CheckResult(allowed=False)
                if verdict is True:
                    continue
                # ここへ来たものはすべて拒否。文字列なら理由として扱う。
                return CheckResult(
                    allowed=False, reason=verdict if isinstance(verdict, str) else ""
                )
            return CheckResult(allowed=True)


