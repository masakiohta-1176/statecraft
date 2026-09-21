"""
テスト全体で使う道具。

方針は1つだけ——**LLMのAPIを絶対に呼ばない**。
生成の中身はFakeLLMへ台本として渡し、テストが検証するのは
「その応答を受けてフレームワークがどう振る舞ったか」に限る。
"""

import json

import pytest

from statecraft import BaseLLM, Tool
from statecraft.llm import FunctionCall, LLMResponse


class FakeLLM(BaseLLM):
    """
    台本どおりに応答する偽のLLM。

    responsesへ入れた順に1つずつ返す。要素が例外なら送出する
    （再試行の判定を試すため）。呼び出し可能なら kwargs を渡して呼ぶ
    （「そのphaseに何が渡ってきたか」で応答を変えたい場合に使う）。

    渡された引数は全てcallsへ残す。プロンプトやtoolsやresponse_schemaが
    正しく組み立てられたかは、ここを見て検証する。
    """

    def __init__(self, responses=None):
        self.scripted = list(responses or [])
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError(
                f"FakeLLMの台本が尽きました（{len(self.calls)}回目の生成）。"
                f"想定より多く生成が走っています。"
            )
        nxt = self.scripted.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt):
            return nxt(kwargs)
        return nxt

class RuleLLM(BaseLLM):
    """
    役割ごとに応答を決める偽LLM。台本の「数」を当てにしない。

    打ち切りや停滞の検証では、記憶の差し戻しなどで生成回数が変わるため、
    FakeLLMの台本だと数がずれて検証したい所へ届かない。こちらは渡された
    引数の形からphaseを見分けて応答する。

        toolsがある           → 呼び出しフェーズ
        response_schemaがある → 記憶を更新するフェーズ
        どちらも無い          → 回答フェーズ

    output_schemaを持つAgentでは回答フェーズにもresponse_schemaが渡るため、
    この見分けは使えない（そのケースはFakeLLMの台本で書く）。
    """

    def __init__(self, *, call=None, memory=None, answer: str = "回答"):
        self.call = call
        self.memory = memory if memory is not None else []
        self.answer = answer
        self.calls: list[dict] = []
        self.call_count = 0

    def generate(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)

        if kwargs.get("tools"):
            self.call_count += 1
            return self.call(self.call_count) if callable(self.call) else self.call

        if kwargs.get("response_schema"):
            rows = self.memory(self.call_count) if callable(self.memory) else self.memory
            return diff_response(rows)

        return text_response(self.answer)


def text_response(text: str, **kw) -> LLMResponse:
    """テキストだけを返す応答。"""
    return LLMResponse(text=text, **kw)


def call_response(name: str, **args) -> LLMResponse:
    """tool/agentの呼び出しを1件要求する応答。"""
    return LLMResponse(text="", function_calls=[FunctionCall(name=name, args=args)])


def calls_response(*pairs) -> LLMResponse:
    """複数の呼び出しを要求する応答。pairsは (名前, 引数dict) の並び。"""
    return LLMResponse(
        text="", function_calls=[FunctionCall(name=n, args=a) for n, a in pairs]
    )


def diff_response(rows: list) -> LLMResponse:
    """memory差分（JSON配列）を返す応答。"""
    return LLMResponse(text=json.dumps(rows, ensure_ascii=False))


def task_row(task_id: str, text: str, status: str, targets=None) -> dict:
    """tasks用の差分1行。"""
    row = {"field": "tasks", "id": task_id, "text": text, "status": status}
    if targets is not None:
        row["target_names"] = targets
    return row


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def echo_tool():
    """引数をそのまま返すtool。実行されたかどうかの確認に使う。"""

    def echo(message: str) -> dict:
        return {"value": f"echo: {message}"}

    return Tool(func=echo, summary="受け取った文字列をそのまま返す")


@pytest.fixture
def recording_tool():
    """呼ばれた引数を記録するtool。calls属性で確認する。"""
    calls: list[dict] = []

    def search_rules(topic: str) -> dict:
        calls.append({"topic": topic})
        return {"value": f"{topic}の規程です"}

    tool = Tool(func=search_rules, summary="規程を調べる")
    tool.calls = calls  # type: ignore[attr-defined]
    return tool
