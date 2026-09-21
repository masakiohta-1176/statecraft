"""
utils の検証。

小さいが、どちらも「利用者が書いたもの」を相手にする場所なので、
壊れ方が実行の停止になってはいけない。
"""

from dataclasses import dataclass

from statecraft.utils import format_message, with_timing


# ---- format_message ----
def test_引数を差し込む():
    assert format_message("『{title}』を照会中", {"title": "深夜特急"}) == "『深夜特急』を照会中"


def test_足りないキーは空文字になる():
    """引数はLLMが生成するのでキーが欠けうる。欠けても文言は出す。"""
    assert format_message("『{title}』を照会中", {}) == "『』を照会中"


def test_余分なキーは無視される():
    assert format_message("{a}", {"a": "1", "b": "2"}) == "1"


def test_書式が壊れていてもテンプレートを返す():
    """文言の生成で実行を止めない。"""
    assert format_message("{未閉じ", {}) == "{未閉じ"


def test_差し込む値が例外を投げてもテンプレートを返す():
    class Exploding:
        def __format__(self, spec):
            raise RuntimeError("壊れた値")

    assert format_message("{x}", {"x": Exploding()}) == "{x}"


def test_差し込み記法が無ければそのまま返す():
    assert format_message("ただの文言", {"a": "1"}) == "ただの文言"


# ---- with_timing ----
def test_実行時間が戻り値へ入る():
    @dataclass
    class Result:
        execution_time: float = 0.0

    @with_timing
    def run() -> Result:
        return Result()

    assert run().execution_time >= 0


def test_関数名が保たれる():
    """Tool.nameがfunc.__name__から取るため、wrapsが外れると名前が変わる。"""

    @with_timing
    def search_rules():
        return type("R", (), {"execution_time": 0.0})()

    assert search_rules.__name__ == "search_rules"


def test_引数はそのまま渡る():
    @dataclass
    class Result:
        value: str = ""
        execution_time: float = 0.0

    @with_timing
    def run(a, *, b) -> Result:
        return Result(value=f"{a}/{b}")

    assert run("x", b="y").value == "x/y"
