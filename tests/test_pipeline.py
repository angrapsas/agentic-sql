"""The component pipeline, without a GPU: code stages, and a model that is
perfect at every stage, which must give a perfect pipeline."""

import re

import pytest

from agsql import dataset, grading
from agsql.setups import get_setup
from agsql.setups.pipeline import Schema, code_linker, code_router


@pytest.fixture(scope="module")
def questions():
    return dataset.load("chinook_v2")


class Oracle:
    def __init__(self, questions):
        self.by_q = {q.question: q for q in questions}
        self.tables = Schema().tables

    def generate(self, prompt):
        q = self.by_q[re.search(r"Question: (.*)", prompt).group(1)]
        if "Answer only YES or NO" in prompt:
            return ("YES" if q.answerable else "NO"), 1, 1
        if "comma-separated" in prompt:
            return ", ".join(t for t in self.tables if re.search(r"\b%s\b" % t, q.gold_sql)), 1, 1
        if "This query failed" in prompt:
            return q.gold_sql, 1, 1
        return (q.gold_sql if q.answerable else "CANNOT ANSWER"), 1, 1


def score(arm, questions):
    return sum(grading.grade(arm(q.question)["sql"], q.gold_sql if q.answerable else None)[0]
               for q in questions)


def test_perfect_model_gives_perfect_pipeline(questions):
    arm = get_setup("pipeline", backend=Oracle(questions), router="model",
                    linker="model", generator="model", repair="model")
    assert score(arm, questions) == len(questions)


def test_all_code_matches_templates(questions):
    pipe = get_setup("pipeline")
    templ = get_setup("templates")
    assert score(pipe, questions) == score(templ, questions)


def test_code_linker_connects_tables():
    # Artist -> Track has no direct key: the linker must add Album to join them
    tables = code_linker(Schema(), "Which genres does the artist AC/DC have tracks in?")["tables"]
    assert {"Artist", "Album", "Track", "Genre"} <= set(tables)


def test_code_linker_cannot_know_names():
    # the brittleness being measured: nothing in the words says AC/DC is an artist
    tables = code_linker(Schema(), "Which genres does AC/DC have tracks in?")["tables"]
    assert "Artist" not in tables


def test_code_router_refuses_out_of_schema():
    assert not code_router(None, "What is Nancy Edwards's salary?")["answerable"]
    assert code_router(None, "How many tracks are there?")["answerable"]


def test_model_stage_needs_backend():
    with pytest.raises(SystemExit):
        get_setup("pipeline", router="model")
