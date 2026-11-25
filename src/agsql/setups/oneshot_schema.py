"""One shot with the full schema: every table AND its columns in the prompt.

The plain oneshot setup lists table names only, so a model that has seen
Chinook in pre-training can fill in the columns from memory and one that has
not cannot. Here every model is shown the columns, so memory of the schema
gives no advantage and one-shot measures reasoning without tools. On the
renamed schema this is the fair one-shot condition.
"""

import sqlite3
import time

from . import register
from ..dataset import DB_PATH
from ..grading import extract_sql

PROMPT = ("Schema:\n{schema}\n"
          "Question: {question}\n"
          "Write one sqlite query that answers it. Output only the SQL. "
          "If the schema cannot answer it, reply exactly: CANNOT ANSWER")


def schema_text(db=None):
    """One line per table: Name(col1, col2, ...), in database order."""
    con = sqlite3.connect(db or DB_PATH)
    try:
        tables = [r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")]
        lines = []
        for t in tables:
            cols = [c[1] for c in con.execute('PRAGMA table_info("%s")' % t)]
            lines.append("%s(%s)" % (t, ", ".join(cols)))
        return "\n".join(lines)
    finally:
        con.close()


@register("oneshot_schema")
def make(backend, memory=None, db=None, **kw):
    schema = schema_text(db)

    def arm(question):
        prompt = PROMPT.format(schema=schema, question=question)
        t0 = time.perf_counter()
        text, tin, tout = backend.generate(prompt)
        secs = round(time.perf_counter() - t0, 2)
        sql = extract_sql(text)
        return {"sql": sql, "raw": text, "secs": secs,
                "tokens_in": tin, "tokens_out": tout,
                "model_calls": 1, "queries": 1 if sql else 0,
                "queries_distinct": 1 if sql else 0, "errors": 0}

    return arm
