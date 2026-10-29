"""Tests that run without a GPU or an API key, using the mock backend.

These cover the parts that broke in the previous phase: SQL extraction from
thinking output, the loop refusing an unsupported answer, PRAGMA not being
graded as an answer, and memory surviving a resume.

    PYTHONPATH=src python -m pytest tests -q
"""

import os
import shutil
import tempfile

import pytest

from agsql import dataset
from agsql.backends import get_backend
from agsql.grading import extract_sql, grade
from agsql.memory import Memory
from agsql.runner import run_arm
from agsql.setups import get_setup
from agsql.setups.loop import run_loop

TABLES = ["Album", "Artist", "Track", "Customer", "Invoice"]


def gen_from(script):
    it = iter(script)
    last = [script[-1]]

    def gen(_prompt):
        try:
            t = next(it)
        except StopIteration:
            t = last[0]
        return t, 10, 5
    return gen


# ---- SQL extraction ----

@pytest.mark.parametrize("text,want", [
    ("SELECT COUNT(*) FROM Track", "SELECT COUNT(*) FROM Track"),
    ("SELECT COUNT(*) FROM Artist;", "SELECT COUNT(*) FROM Artist"),
    ("<think>hidden</think>SELECT 1", "SELECT 1"),
    # thinking on: Qwen emits a closing tag with no opening tag
    ("reasoning\n</think>\n\nPRAGMA table_info(Track)", "PRAGMA table_info(Track)"),
    ("why\n</think>\n\nSELECT COUNT(*) FROM Track\n\nLet me run this.",
     "SELECT COUNT(*) FROM Track"),
    # the loop's own transcript format must not leak into the SQL
    ("QUERY: SELECT 1\nRESULT: [(1,)]", "SELECT 1"),
    ("```sql\nSELECT MAX(Milliseconds) / 60000.0 FROM Track\n```",
     "SELECT MAX(Milliseconds) / 60000.0 FROM Track"),
    ("I cannot answer that.", ""),
])
def test_extract_sql(text, want):
    assert extract_sql(text) == want


# ---- grading ----

def test_grade_exact_and_loose():
    gold = "SELECT COUNT(*) FROM Artist"
    assert grade(gold, gold)[:2] == (True, True)
    # extra column: right answer, wrong shape
    correct, loose, _ = grade(
        "SELECT COUNT(*) AS n, 'x' AS label FROM Artist", gold)
    assert (correct, loose) == (False, True)
    # wrong answer is neither
    assert grade("SELECT COUNT(*) FROM Track", gold)[:2] == (False, False)


def test_grade_unanswerable():
    assert grade("", None)[0] is True
    assert grade("", None, "CANNOT ANSWER")[0] is True
    assert grade("SELECT 1", None)[0] is False


def test_column_order_does_not_matter():
    assert grade("SELECT LastName, FirstName FROM Employee WHERE EmployeeId = 1",
                 "SELECT FirstName, LastName FROM Employee WHERE EmployeeId = 1")[0]


# ---- the loop ----

def test_loop_rejects_answer_with_no_query():
    """Answering from memory on the first step would measure hallucination."""
    r = run_loop(gen_from(["ANSWER: 42"]), "How many songs?", TABLES, max_steps=3)
    assert r["queries"] == 0 and r["sql"] == ""


def test_loop_accepts_bare_sql_then_answers():
    r = run_loop(gen_from(["PRAGMA table_info(Track)",
                           "SELECT COUNT(*) FROM Track",
                           "ANSWER: 3503"]),
                 "How many songs?", TABLES)
    assert r["queries"] == 2 and r["raw"] == "3503"
    assert r["sql"] == "SELECT COUNT(*) FROM Track"


def test_loop_does_not_grade_a_pragma_as_the_answer():
    """A schema lookup on the final step must not be scored against the gold."""
    r = run_loop(gen_from(["SELECT COUNT(*) FROM Artist",
                           "PRAGMA table_info(Track)",
                           "ANSWER: 275"]),
                 "How many musicians?", TABLES)
    assert r["sql"] == "SELECT COUNT(*) FROM Artist"


def test_loop_ignores_echoed_placeholder():
    r = run_loop(gen_from(["SELECT COUNT(*) FROM Track",
                           "ANSWER: <the answer>",
                           "ANSWER: 3503"]),
                 "How many songs?", TABLES)
    assert r["raw"] == "3503"


def test_loop_counts_sql_errors_and_recovers():
    r = run_loop(gen_from(["SELECT COUNT(*) FROM Track WHERE TrackName IS NOT NULL",
                           "PRAGMA table_info(Track)",
                           "SELECT COUNT(*) FROM Track",
                           "ANSWER: 3503"]),
                 "How many songs?", TABLES)
    assert r["errors"] == 1 and r["sql"] == "SELECT COUNT(*) FROM Track"


def test_repeats_show_in_distinct_count():
    r = run_loop(gen_from(["SELECT COUNT(*) FROM Track"] * 6),
                 "How many songs?", TABLES, max_steps=6)
    assert r["queries"] == 6 and r["queries_distinct"] == 1


# ---- memory ----

def test_memory_is_truthy_when_empty():
    """__len__ without __bool__ silently disables memory on the first question."""
    assert bool(Memory(path=None)) is True


def test_memory_accumulates_persists_and_restores():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "m.json")
        mem = Memory(path=path)
        run_loop(gen_from(["SELECT COUNT(*) FROM Track", "ANSWER: 3503"]),
                 "q", TABLES, memory=mem)
        assert len(mem) == 1
        assert len(Memory(path=path)) == 1      # a resumed arm restores its notes
        assert "Notes from earlier questions" in mem.prefix()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_memory_disabled_stays_empty():
    mem = Memory(path=None, enabled=False)
    run_loop(gen_from(["SELECT COUNT(*) FROM Track", "ANSWER: 3503"]),
             "q", TABLES, memory=mem)
    assert len(mem) == 0 and mem.prefix() == ""


def test_memory_does_not_record_failed_queries():
    """Memory holds verified usage only, or it propagates the model's mistakes."""
    mem = Memory(path=None)
    run_loop(gen_from(["SELECT COUNT(*) FROM Track WHERE Nope = 1", "ANSWER: 0"]),
             "q", TABLES, max_steps=2, memory=mem)
    assert len(mem) == 0


# ---- dataset and runner ----

def test_dataset_loads_and_buckets():
    qs = dataset.load("chinook_v1")
    assert len(qs) == 43
    assert dataset.bucket_counts(qs) == {"vocabulary": 16, "logic": 18, "judgment": 9}
    assert sum(1 for q in qs if not q.answerable) == 4


def test_subset_is_deterministic():
    qs = dataset.load("chinook_v1")
    a = [q.id for q in dataset.subset(qs, per_bucket=2)]
    b = [q.id for q in dataset.subset(qs, per_bucket=2)]
    assert a == b and len(a) == 6


def test_runner_checkpoints_and_resumes():
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, "arm.json")
        qs = dataset.subset(dataset.load("chinook_v1"), limit=4)
        calls = {"n": 0}

        def flaky(_q):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom")
            return {"sql": "SELECT COUNT(*) FROM Track"}

        rows = run_arm("flaky", flaky, qs, out=out, verbose=False)
        assert len(rows) == 4
        # the crashed question is recorded as a crash, not silently dropped
        assert any("crash" in r["detail"] for r in rows)

        rows2 = run_arm("flaky", flaky, qs, out=out, verbose=False)
        assert len(rows2) == 4            # resumed, nothing duplicated
        assert len({r["id"] for r in rows2}) == 4
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_templates_arm_scores_as_expected():
    """The no-model baseline is perfect on what it anticipated, zero elsewhere."""
    qs = dataset.load("chinook_v1")
    rows = run_arm("templates", get_setup("templates"), qs, verbose=False)
    acc = sum(r["correct"] for r in rows) / len(rows)
    assert 0.4 < acc < 0.6


def test_mock_backend_runs_a_whole_arm():
    qs = dataset.subset(dataset.load("chinook_v1"), limit=5)
    arm = get_setup("loop", backend=get_backend("mock"))
    rows = run_arm("mock", arm, qs, verbose=False)
    assert len(rows) == 5 and all(r["model_calls"] >= 1 for r in rows)
