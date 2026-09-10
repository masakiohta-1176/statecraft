"""
どのモジュールからも使う小さな道具。


ここに置くのは「特定の関心事に属さないもの」だけにする。
記憶に関わるものはmemory.py、LLMに関わるものはllm.pyへ置く。
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


    引数はLLMが生成するため、必須のはずのキーが欠けることがある。
    KeyErrorで落とさず空文字にする。文言の生成は観測のための処理であり、
    そこでの不備が実行を止めてよい理由がない。


    format自体が失敗する場合（テンプレートに素の { が含まれる等）も
    テンプレートをそのまま返す。何も出ないより、埋まっていない文言でも
    出た方が「今動いている」ことは伝わる。
    """
    try:
        return template.format_map(_SafeDict(kwargs))
    except Exception:
        return template




def with_timing(func):
    """
    関数の実行時間を計測し、戻り値の execution_time へ入れるデコレータ。


        @with_timing
        def execute(self, ...) -> ToolResult:
            ...
        → 戻ってきたToolResultのexecution_timeに秒数が入っている


    計測を各メソッドの中に書くと、開始時刻を変数に置いて、
    return する全ての経路で差を計算することになる。経路が増えるたびに
    書き忘れる余地が生まれるので、デコレータとして外側から包む。


    戻り値は execution_time という属性を持つオブジェクトである必要がある
    （LLMResponse / EmbeddingResponse / ToolResult のいずれか）。
    属性が無いオブジェクトを返す関数に付けるとAttributeErrorになる。


    functools.wrapsを付けているのは、包んだ後も元の関数の名前
    （__name__）とdocstringを保つため。これが無いと、デコレートした
    関数の名前が全て "wrapper" になってしまう。
    ツール名はfuncの__name__から取っているため、実害が出る。
    """


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        result.execution_time = round(time.perf_counter() - start, 3)
        return result


    return wrapper





