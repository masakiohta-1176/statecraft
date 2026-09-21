"""
どのモジュールからも使う小さな道具。

特定の関心事に属さないものだけを置く（記憶はmemory.py、LLMはllm.py）。
"""

import functools
import time


class _SafeDict(dict):
    """存在しないキーを空文字として扱う辞書。format_mapへ渡す。"""

    def __missing__(self, key):
        return ""


def format_message(template: str, kwargs: dict) -> str:
    """
    実行中メッセージのテンプレートへ引数を差し込む。

        "『{title}』の所蔵状況を照会しています..."

    引数はLLMが生成するのでキーが欠けうる。欠けたキーは空文字、formatが失敗
    した場合はテンプレートを返す（文言の生成で実行を止めない）。
    """
    try:
        return template.format_map(_SafeDict(kwargs))
    except Exception:  # noqa: BLE001
        return template


def with_timing(func):
    """
    関数の実行時間を計測し、戻り値の execution_time へ入れるデコレータ。

    戻り値は execution_time 属性を持つ必要がある（LLMResponse /
    EmbeddingResponse / ToolResult）。無いとAttributeError。
    wrapsが必要なのは、ツール名をfuncの__name__から取っているため。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        result.execution_time = round(time.perf_counter() - start, 3)
        return result

    return wrapper


