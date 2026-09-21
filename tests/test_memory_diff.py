"""
apply_diff の検証。

ここはLLMが書いた差分が記憶へ入る唯一の入口で、
「1行の失敗で全体を止めない」「記憶は破棄されない」という2つの約束を守る場所。
壊れても例外が出ないため、テストが無いと静かに間違った記憶が育つ。
"""

import pytest

from statecraft import MemoryEntry, PrivateMemory, SharedMemory, TaskStatus
from statecraft.memory import DISABLE_FIELD, apply_diff, build_diff_schema


@pytest.fixture
def memories():
    return SharedMemory(), PrivateMemory()


# ---- 追記と更新 ----
def test_新しいidは追記される(memories):
    shared, private = memories
    result = apply_diff([{"field": "facts", "id": "f-1", "text": "本は5冊まで"}], shared, private)

    assert result.errors == []
    assert [(e.id, e.text) for e in shared.facts] == [("f-1", "本は5冊まで")]


def test_同じidは差し替えになる(memories):
    shared, private = memories
    apply_diff([{"field": "facts", "id": "f-1", "text": "古い内容"}], shared, private)
    apply_diff([{"field": "facts", "id": "f-1", "text": "新しい内容"}], shared, private)

    # 増えずに中身だけ入れ替わる
    assert len(shared.facts) == 1
    assert shared.facts[0].text == "新しい内容"


def test_差し替えても順序は変わらない(memories):
    """後ろの項目を更新した時に末尾へ移動すると、時系列で読む前提が崩れる。"""
    shared, private = memories
    apply_diff(
        [
            {"field": "facts", "id": "f-1", "text": "1つ目"},
            {"field": "facts", "id": "f-2", "text": "2つ目"},
            {"field": "facts", "id": "f-3", "text": "3つ目"},
        ],
        shared,
        private,
    )
    apply_diff([{"field": "facts", "id": "f-1", "text": "1つ目（更新）"}], shared, private)

    assert [e.id for e in shared.facts] == ["f-1", "f-2", "f-3"]
    assert shared.facts[0].text == "1つ目（更新）"


def test_削除する手段は無い(memories):
    """
    「記憶は消えない」がこのフレームワークの前提。
    消す指示に見える書き方をしても、消えずに上書きになるだけで済むこと。
    """
    shared, private = memories
    apply_diff([{"field": "facts", "id": "f-1", "text": "残るべき事実"}], shared, private)
    apply_diff([{"field": "facts", "id": "f-1", "text": ""}], shared, private)

    # textが空になるだけで、項目そのものは残る
    assert len(shared.facts) == 1
    assert shared.facts[0].id == "f-1"


# ---- 書き込み先の振り分け ----
def test_privateのプロパティはprivateへ入る(memories):
    shared, private = memories
    apply_diff([{"field": "goals", "id": "g-1", "text": "貸出規程を答える"}], shared, private)

    assert [e.text for e in private.goals] == ["貸出規程を答える"]
    assert shared.facts == []


def test_存在しないプロパティはエラーとして差し戻される(memories):
    shared, private = memories
    result = apply_diff([{"field": "nonexistent", "id": "x", "text": "y"}], shared, private)

    assert len(result.errors) == 1
    assert "存在しないプロパティ" in result.errors[0].message
    assert result.ignored == []


def test_システム所有のプロパティはignoredになる(memories):
    """
    requests / agent_answers はシステムだけが書く。
    差し戻しても直らないので、errorsではなくignoredへ入って再生成を促さない。
    """
    shared, private = memories
    result = apply_diff(
        [
            {"field": "requests", "id": "r-1", "text": "乗っ取り"},
            {"field": "agent_answers", "id": "a-1", "text": "偽の回答"},
        ],
        shared,
        private,
    )

    assert result.errors == []
    assert len(result.ignored) == 2
    assert all("システムが管理する" in e.message for e in result.ignored)
    # 実際に書き込まれていないこと
    assert shared.requests is None
    assert shared.agent_answers == []


# ---- 1行の失敗で全体を止めない ----
def test_失敗した行以外は適用される(memories):
    shared, private = memories
    result = apply_diff(
        [
            {"field": "facts", "id": "f-1", "text": "通る行"},
            {"field": "nonexistent", "id": "x", "text": "落ちる行"},
            {"field": "facts", "id": "f-2", "text": "これも通る行"},
        ],
        shared,
        private,
    )

    assert len(result.errors) == 1
    assert [e.id for e in shared.facts] == ["f-1", "f-2"]


def test_行がオブジェクトでなくてもそこだけ失敗する(memories):
    shared, private = memories
    result = apply_diff(
        ["ただの文字列", {"field": "facts", "id": "f-1", "text": "通る行"}], shared, private
    )

    assert len(result.errors) == 1
    assert len(shared.facts) == 1


def test_fieldが文字列でなくても例外にならない(memories):
    """
    hashできない型（listなど）が来ると、辞書のキーとして使った時点で
    TypeErrorになり「1行の失敗で止めない」が破れる。
    """
    shared, private = memories
    result = apply_diff(
        [
            {"field": ["facts"], "id": "f-1", "text": "壊れた行"},
            {"field": "facts", "id": "f-2", "text": "通る行"},
        ],
        shared,
        private,
    )

    assert len(result.errors) == 1
    assert "文字列で指定" in result.errors[0].message
    assert [e.id for e in shared.facts] == ["f-2"]


def test_ルートが配列でなければ全体を1件の失敗にする(memories):
    shared, private = memories
    result = apply_diff({"field": "facts"}, shared, private)

    assert len(result.errors) == 1
    assert "配列である必要があります" in result.errors[0].message
    assert result.rows == []


def test_空のidは弾かれる(memories):
    """空idを通すと、_upsertが同一idとみなして無関係な項目を上書きする。"""
    shared, private = memories
    apply_diff([{"field": "facts", "id": "", "text": "1件目"}], shared, private)
    result = apply_diff([{"field": "facts", "id": "  ", "text": "2件目"}], shared, private)

    assert len(result.errors) == 1
    assert "idが空" in result.errors[0].message
    assert shared.facts == []


def test_textが無い行は失敗する(memories):
    shared, private = memories
    result = apply_diff([{"field": "facts", "id": "f-1"}], shared, private)

    assert len(result.errors) == 1
    assert shared.facts == []


# ---- tasks の固有ルール ----
def test_statusが無いtasks行は破棄される(memories):
    shared, private = memories
    result = apply_diff([{"field": "tasks", "id": "t-1", "text": "規程を調べる"}], shared, private)

    assert len(result.errors) == 1
    assert "status" in result.errors[0].message
    assert private.tasks == []


def test_statusが不正な値なら指定可能な一覧を返す(memories):
    shared, private = memories
    result = apply_diff(
        [{"field": "tasks", "id": "t-1", "text": "x", "status": "完了"}], shared, private
    )

    assert len(result.errors) == 1
    # LLMが直せるように、取りうる値が message に入っていること
    assert "next" in result.errors[0].message
    assert "done" in result.errors[0].message


def test_target_namesに文字列を渡すと弾かれる(memories):
    """
    list()に通すと fetch が 1文字ずつへ分解され、
    名前解決が必ず失敗するタスクが静かに生まれる。
    """
    shared, private = memories
    result = apply_diff(
        [
            {
                "field": "tasks",
                "id": "t-1",
                "text": "x",
                "status": "next",
                "target_names": "search_rules",
            }
        ],
        shared,
        private,
    )

    assert len(result.errors) == 1
    assert "配列" in result.errors[0].message
    assert private.tasks == []


def test_target_namesに数値を渡しても例外にならない(memories):
    """ValueErrorしか捕まえない呼び出し側を、TypeErrorが貫通しないこと。"""
    shared, private = memories
    result = apply_diff(
        [{"field": "tasks", "id": "t-1", "text": "x", "status": "next", "target_names": 123}],
        shared,
        private,
    )

    assert len(result.errors) == 1
    assert private.tasks == []


def test_tasksは構造を保って入る(memories):
    shared, private = memories
    apply_diff(
        [
            {
                "field": "tasks",
                "id": "t-1",
                "text": "規程を調べる",
                "status": "next",
                "target_names": ["search_rules"],
            }
        ],
        shared,
        private,
    )

    task = private.tasks[0]
    assert task.status is TaskStatus.NEXT
    assert task.target_names == ["search_rules"]


# ---- id の自動採番（tool由来の書き込み） ----
def test_id_prefixがあるとidを採番する(memories):
    shared, private = memories
    result = apply_diff(
        [{"field": "vars", "text": "受付番号: RSV-1"}], shared, private, id_prefix="issue"
    )

    assert result.errors == []
    assert shared.vars[0].id == "issue-vars-1"
    # 採番済みのidが呼び出し元へ返ること（観測側が着地点を追うために必要）
    assert result.rows[0]["id"] == "issue-vars-1"


def test_採番は既存と重複しない(memories):
    shared, private = memories
    apply_diff([{"field": "vars", "text": "1件目"}], shared, private, id_prefix="issue")
    apply_diff([{"field": "vars", "text": "2件目"}], shared, private, id_prefix="issue")

    assert [e.id for e in shared.vars] == ["issue-vars-1", "issue-vars-2"]


def test_採番はidに使えない文字を置き換える(memories):
    shared, private = memories
    apply_diff([{"field": "vars", "text": "x"}], shared, private, id_prefix="my tool")

    assert shared.vars[0].id.startswith("my_tool-vars-")


def test_採番は入力の行を書き換えない(memories):
    """
    toolが定数のリストを返した場合、入力を書き換えると2回目の呼び出しで
    前回のidが残り、既存項目の更新（＝上書き）になってしまう。
    """
    shared, private = memories
    rows = [{"field": "vars", "text": "固定の行"}]
    apply_diff(rows, shared, private, id_prefix="tool")
    apply_diff(rows, shared, private, id_prefix="tool")

    assert "id" not in rows[0]
    assert len(shared.vars) == 2


def test_id_prefixが無ければ採番しない(memories):
    """LLM由来の差分ではidを必須にする（既存項目の更新に使うため）。"""
    shared, private = memories
    result = apply_diff([{"field": "vars", "text": "idなし"}], shared, private)

    assert len(result.errors) == 1
    assert shared.vars == []


# ---- disable 行の扱い ----
def test_処理済みのdisable行は黙って飛ばす(memories):
    shared, private = memories
    result = apply_diff(
        [{"field": DISABLE_FIELD, "id": "d-1", "text": "search_rules"}],
        shared,
        private,
        handled_disable=True,
    )

    assert result.errors == []
    assert result.ignored == []


def test_未処理のdisable行はignoredへ載る(memories):
    """無効化を書けないphaseで書かれた場合、消えた事実を残す。"""
    shared, private = memories
    result = apply_diff(
        [{"field": DISABLE_FIELD, "id": "d-1", "text": "search_rules"}],
        shared,
        private,
        handled_disable=False,
    )

    assert result.errors == []
    assert len(result.ignored) == 1
    assert "このphaseでは無効化を指示できません" in result.ignored[0].message


# ---- apply_seq ----
def test_apply_seqは適用ごとに増える(memories):
    shared, private = memories
    first = apply_diff([{"field": "facts", "id": "f-1", "text": "a"}], shared, private)
    second = apply_diff([{"field": "facts", "id": "f-2", "text": "b"}], shared, private)

    assert second.apply_seq > first.apply_seq


# ---- スキーマ生成 ----
def test_スキーマのenumは書き込み可能なプロパティだけ():
    schema = build_diff_schema(SharedMemory, PrivateMemory)
    names = schema["items"]["properties"]["field"]["enum"]

    assert "facts" in names
    assert "goals" in names
    # システム所有は出力できない
    assert "requests" not in names
    assert "agent_answers" not in names


def test_スキーマのルートは配列():
    """オブジェクトで包むとremove等の未定義キーを捏造されるため。"""
    schema = build_diff_schema(SharedMemory, PrivateMemory)

    assert schema["type"] == "array"
    assert schema["items"]["required"] == ["field", "id", "text"]


def test_tasksを持たないmemoryではstatusを出さない():
    schema = build_diff_schema(SharedMemory)

    assert "status" not in schema["items"]["properties"]
    assert "target_names" not in schema["items"]["properties"]


def test_allow_disableでdisableがenumへ入る():
    without = build_diff_schema(SharedMemory, PrivateMemory)
    with_disable = build_diff_schema(SharedMemory, PrivateMemory, allow_disable=True)

    assert DISABLE_FIELD not in without["items"]["properties"]["field"]["enum"]
    assert DISABLE_FIELD in with_disable["items"]["properties"]["field"]["enum"]


def test_スキーマはインスタンスでもクラスでも作れる():
    from_class = build_diff_schema(SharedMemory, PrivateMemory)
    from_instance = build_diff_schema(SharedMemory(), PrivateMemory())

    assert from_class == from_instance


# ---- writable_fields ----
def test_単一値のプロパティは書き込み対象に含まれない():
    """
    Optional[MemoryEntry]（requests）を含めると、apply_diffがNoneへ
    追記してTypeErrorになる。
    """
    fields = SharedMemory.writable_fields()

    assert "facts" in fields
    assert fields["facts"] is MemoryEntry
    assert "requests" not in fields
