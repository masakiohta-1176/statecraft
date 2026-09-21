"""
Tool の検証。

ここの要点は2つ。
  ・戻り値の契約（value / blobs / memory）を1つに固定し、推測で埋めない
  ・利用者が書いた関数がどう壊れても、例外を外へ出さずToolResultで返す
"""

import pytest

from statecraft import Interceptor, Tool, ToolResult
from statecraft.tools import parse_tool_return


# ---- 戻り値の契約 ----
def test_valueだけの戻り値を受け取る():
    result = parse_tool_return("t", {"value": "ok"})

    assert result.success
    assert result.value == "ok"


def test_辞書でない戻り値は失敗になる():
    result = parse_tool_return("t", "ただの文字列")

    assert not result.success
    assert "辞書ではありません" in result.error
    # 推測でvalueを埋めないこと
    assert result.value is None


def test_valueが無ければ失敗になる():
    result = parse_tool_return("t", {"blobs": []})

    assert not result.success
    assert "value" in result.error


def test_綴り違いのキーは失敗になる():
    """blob（sなし）のような綴り違いを黙って捨てると、添付が消えた理由が分からない。"""
    result = parse_tool_return("t", {"value": "ok", "blob": []})

    assert not result.success
    assert "blob" in result.error


def test_valueがNoneでも成功として扱う():
    """Noneは正当な戻り値。errorの有無だけが成否を決める。"""
    result = parse_tool_return("t", {"value": None})

    assert result.success


def test_blobsが配列でなければ失敗になる():
    result = parse_tool_return("t", {"value": "ok", "blobs": "画像"})

    assert not result.success


def test_memoryが辞書でなければ失敗になる():
    result = parse_tool_return("t", {"value": "ok", "memory": ["facts"]})

    assert not result.success


def test_memoryの値が文字列1つでも配列として扱う():
    """1件だけ書く時に [] を付け忘れるのはよくあるので吸収する。"""
    result = parse_tool_return("t", {"value": "ok", "memory": {"vars": "RSV-1"}})

    assert result.success
    assert result.memory == {"vars": ["RSV-1"]}


def test_memoryの値が配列でなければ失敗になる():
    result = parse_tool_return("t", {"value": "ok", "memory": {"vars": 123}})

    assert not result.success
    assert "vars" in result.error


# ---- ToolResult ----
def test_successはerrorの有無から導出される():
    """boolを別に持つと success=True かつ error=失敗 という矛盾が作れてしまう。"""
    assert ToolResult(value="x").success is True
    assert ToolResult(error="失敗").success is False
    # errorが空文字なら「理由を伝えない失敗」であって成功ではない
    assert ToolResult(error="").success is False


# ---- 実行 ----
def test_例外は外へ出ずToolResultになる():
    def broken() -> dict:
        raise RuntimeError("接続先 db://secret が見つからない")

    result = Tool(func=broken, summary="壊れる").execute(kwargs={})

    assert not result.success
    # 既定では実際の例外メッセージを見せない（接続先やパスが混ざりうる）
    assert "db://secret" not in result.error


def test_expose_error_detailsで例外の内容を見せられる():
    def broken() -> dict:
        raise RuntimeError("詳しい理由")

    tool = Tool(func=broken, summary="壊れる", expose_error_details=True)
    result = tool.execute(kwargs={})

    assert "詳しい理由" in result.error


def test_disabledなら実行されない():
    calls = []

    def spy() -> dict:
        calls.append(1)
        return {"value": "ok"}

    result = Tool(func=spy, summary="x", disabled=True).execute(kwargs={})

    assert not result.success
    assert calls == []


def test_before_executeが拒否すると実行されない():
    calls = []

    def spy(target: str) -> dict:
        calls.append(target)
        return {"value": "ok"}

    interceptor = Interceptor()
    interceptor.on.before_execute(lambda *, name, kwargs: "権限がありません")

    result = Tool(func=spy, summary="x").execute(
        kwargs={"target": "a"}, interceptor=interceptor
    )

    assert calls == []
    # 理由がそのままLLMへ返ること
    assert result.error == "権限がありません"


def test_実行時間が記録される():
    def quick() -> dict:
        return {"value": "ok"}

    result = Tool(func=quick, summary="x").execute(kwargs={})

    assert result.execution_time >= 0


def test_通知の失敗は実行を止めない():
    """describe_executionが壊れていても、tool本体は動くこと。"""

    def ok() -> dict:
        return {"value": "本体は動いた"}

    def broken_describe(kwargs: dict) -> str:
        raise ValueError("文言の組み立てに失敗")

    errors = []
    interceptor = Interceptor()
    interceptor.on.error(errors.append)

    tool = Tool(func=ok, summary="x", describe_execution=broken_describe)
    result = tool.execute(kwargs={}, interceptor=interceptor)

    assert result.value == "本体は動いた"
    # 黙って消さず、観測側へは伝えること
    assert any("文言" in e or "メッセージ" in e for e in errors)


# ---- 名前 ----
def test_nameは関数名から取る():
    def search_rules(topic: str) -> dict:
        return {"value": topic}

    assert Tool(func=search_rules, summary="x").name == "search_rules"


def test_nameは__name__を持たない呼び出し可能オブジェクトでも取れる():
    class Callable_:
        def __call__(self) -> dict:
            return {"value": "ok"}

    assert Tool(func=Callable_(), summary="x").name == "Callable_"


# ---- 型ヒントからのスキーマ生成 ----
def test_基本型がJSON_Schemaへ変換される():
    def f(a: str, b: int, c: float, d: bool) -> dict:
        return {"value": ""}

    props = Tool(func=f, summary="x").get_json_schema()["properties"]

    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert props["c"]["type"] == "number"
    assert props["d"]["type"] == "boolean"


def test_既定値のない引数だけがrequiredになる():
    def f(need: str, opt: str = "x") -> dict:
        return {"value": ""}

    schema = Tool(func=f, summary="x").get_json_schema()

    assert schema["required"] == ["need"]


def test_配列はitemsまで生成される():
    """itemsが無いと、LLMが配列を文字列として返してくる。"""

    def f(names: list[str]) -> dict:
        return {"value": ""}

    props = Tool(func=f, summary="x").get_json_schema()["properties"]

    assert props["names"]["type"] == "array"
    assert props["names"]["items"]["type"] == "string"


def test_要素型が不明な配列はstringで補う():
    def f(items: list) -> dict:
        return {"value": ""}

    props = Tool(func=f, summary="x").get_json_schema()["properties"]

    assert props["items"]["items"] == {"type": "string"}


@pytest.mark.parametrize("annotation", ["str | None", "typing.Optional[str]"])
def test_Optionalは中身の型で判定される(annotation):
    """書き方によってget_originの戻り値が変わるため、両方を試す。"""
    import typing  # execへ渡す名前空間で使う

    src = f"def f(a: {annotation}) -> dict: return {{'value': ''}}"
    namespace: dict = {"typing": typing}
    exec(src, namespace)  # noqa: S102

    props = Tool(func=namespace["f"], summary="x").get_json_schema()["properties"]

    assert props["a"]["type"] == "string"


def test_Optionalな配列も要素型を保つ():
    def f(names: list[int] | None = None) -> dict:
        return {"value": ""}

    props = Tool(func=f, summary="x").get_json_schema()["properties"]

    assert props["names"]["type"] == "array"
    assert props["names"]["items"]["type"] == "integer"


def test_注釈の無い引数はstringになる():
    def f(a) -> dict:
        return {"value": ""}

    props = Tool(func=f, summary="x").get_json_schema()["properties"]

    assert props["a"]["type"] == "string"


def test_可変長引数はスキーマに含めない():
    """*args / **kwargs はJSON Schemaのプロパティとして表現できない。"""

    def f(a: str, *args, **kwargs) -> dict:
        return {"value": ""}

    schema = Tool(func=f, summary="x").get_json_schema()

    assert list(schema["properties"]) == ["a"]


def test_param_descriptionsがスキーマへ入る():
    def f(topic: str) -> dict:
        return {"value": ""}

    tool = Tool(func=f, summary="x", param_descriptions={"topic": "調べたい題目"})
    props = tool.get_json_schema()["properties"]

    assert props["topic"]["description"] == "調べたい題目"


# ---- 提示のされ方 ----
def test_カタログ行はusageを既定で含めない():
    def f() -> dict:
        return {"value": ""}

    tool = Tool(func=f, summary="概要", usage="詳しい使い方")

    assert tool.to_catalog_line() == "- f: 概要"
    assert "詳しい使い方" in tool.to_catalog_line(include_usage=True)


def test_declarationはsummaryとusageを結合する():
    def f() -> dict:
        return {"value": ""}

    d = Tool(func=f, summary="概要", usage="使い方").to_declaration()

    assert d["name"] == "f"
    assert "概要" in d["description"]
    assert "使い方" in d["description"]
    assert d["parameters"]["type"] == "object"
