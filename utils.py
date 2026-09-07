import functools
import time


class _SafeDict(dict):
    """辞書に存在しないキーを指定されたときに空文字を返すように"辞書クラス"を変更"""

    def __missing__(self, key):
        return ""


def format_message(template: str, kwargs: dict) -> str:  # ->は戻り値の型を明示
    """文字テンプレに対して、引数を入れる関数
    "{title}の所在地を確認中..."
    みたいな
    必須ではない引数や、LLMの生成揺れによる引数省略をエラーで落とさず空文字に変換するための処理
    """
    try:
        return template.format_map(_SafeDict(kwargs))
    except (
        KeyError,
        ValueError,
        AttributeError,
    ):  # 発生し得るエラーだけをキャッチ、エラー内容を扱いたいときは Exception as eとかでいいらしい
        return template


def with_timing(func):
    """関数の実行時間を計測して戻り値に入れるデコレータ
    戻り値はexecution_timeというプロパティを持つオブジェクトである必要がある。
    """

    @functools.wraps(
        func
    )  #  これ付けないと全部の関数が全部wrapperにすり替わっちゃうらしい
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        result.execution_time = round(time.perf_counter() - start, 3)
        return result

    return wrapper
