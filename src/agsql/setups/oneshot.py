"""One shot: table names in the prompt, no tools, no retries.

The model cannot look anything up, so whatever it gets right came out of its
weights. That is what makes this the clean test of whether fine-tuning installed
schema knowledge, as opposed to whether the model can discover it through tools.
"""

import sqlite3
import time

from . import register
from ..dataset import DB_PATH
from ..grading import extract_sql

PROMPT = ("Tables: {tables}\n"
          "Question: {question}\n"
          "Write one sqlite query that answers it. Output only the SQL. "
          "If the schema cannot answer it, reply exactly: CANNOT ANSWER")


def table_names(db=None):
    con = sqlite3.connect(db or DB_PATH)
    try:
        return [r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")]
    finally:
        con.close()


@register("oneshot")
def make(backend, memory=None, db=None, **kw):
    tables = ", ".join(table_names(db))

    def arm(question):
        prompt = PROMPT.format(tables=tables, question=question)
        if memory is not None:
            prompt = memory.prefix() + prompt
        t0 = time.perf_counter()
        text, tin, tout = backend.generate(prompt)
        secs = round(time.perf_counter() - t0, 2)
        sql = extract_sql(text)
        if memory is not None and sql:
            # only record it if it actually runs
            con = sqlite3.connect(db or DB_PATH)
            try:
                con.execute(sql).fetchall()
                memory.add_worked(sql)
            except Exception:
                pass
            finally:
                con.close()
        return {"sql": sql, "raw": text, "secs": secs,
                "tokens_in": tin, "tokens_out": tout,
                "model_calls": 1, "queries": 1 if sql else 0,
                "queries_distinct": 1 if sql else 0, "errors": 0}

    return arm
