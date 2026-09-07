import inspect
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class _OnRegistry:
    """イベント登録専用、interceptor.on.~って書けるように。タイポ予防のため"""

    def __init__(self, interceptor: "Interceptor"):
        self.interceptor = interceptor

    """補完の為、全部記載"""

    def before_execute(self, callback: Callable[..., bool]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def execute_start(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def execute_end(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def execute_blocked(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    #  respond用、Frontエージェント起動時専用
    def respond_start(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def respond_end(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    # memory関連
    def memory_updated(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def memory_diff(self, callback: Callable[..., bool]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)

    def error(self, callback: Callable[[str], None]) -> None:
        self.interceptor._register(inspect.currentframe().f_code.co_name, callback)


class _CheckDispatcher:
    """拒否権を持つ"""

    def __init__(self, interceptor: "Interceptor"):
        self.interceptor = interceptor

    def before_execute(self, *, name: str, kwargs: dict) -> bool:
        return self.interceptor._check(
            inspect.currentframe().f_code.co_name, name=name, kwargs=kwargs
        )


class _NotifyDispatcher:
    """通知のみを行う。"""

    def __init__(self, interceptor: "Interceptor"):
        self.interceptor = interceptor

    def execute_start(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def execute_end(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def execute_blocked(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def respond_start(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def respond_end(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def memory_updated(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def memory_diff(self, message: str) -> None:
        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)

    def error(self, message: str) -> None:

        self.interceptor._notify(inspect.currentframe().f_code.co_name, message)


class Interceptor:
    """ToolとかAgentの挙動をイベントとして一元管理するやつ
    メソッドは(on,check,notify)→イベント名で。
    interceptor.on.execute_start(callback)
    みたいな。
    """

    def __init__(self):
        self._listeners: dict[
            str, list[callable]
        ] = {}  #  リスナー=メソッド、後ほど登場するメソッドに対して関数を紐づける
        self.on = _OnRegistry(self)
        self.check = _CheckDispatcher(self)
        self.notify = _NotifyDispatcher(self)

    @property
    def has_listeners(self) -> bool:
        """コールバックが登録されてるかをnetworkが参照する用"""
        return any(
            self._listeners.values()
        )  #  リスナー辞書の中身だけ取り出して、存在するかどうかを返す（any部分）

    def _register(self, event: str, callback: callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def _notify(self, event: str, *args, **kwargs) -> None:
        """通知発火系
        コールバック内の例外はいったん握りつぶす。
        通知部分のエラーで止まったらたまったもんじゃないため

        ただし、通知部分のエラーに関しログには出す
        """
        for callback in self._listeners.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception("通知コールバックが例外を吐きました:event=%s", event)

    def _check(self, event: str, *args, **kwargs) -> bool:
        """拒否権がある判定。
        明示的にTrueを返したものだけ許可
        None/書き忘れ/そのほかの値はすべて「拒否」
        もちろんエラーも拒否として扱う。
        """
        for callback in self._listeners.get(event, []):
            try:
                allowed = callback(*args, **kwargs)
            except Exception:
                logger.exception(
                    "判定用のコールバック関数がエラーになりました event=%s", event
                )
                return False
            if allowed is not True:
                return False
        return True
