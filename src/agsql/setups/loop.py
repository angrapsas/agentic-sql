"""A tool loop: the model writes a query, we run it, it sees the rows, repeat.

Model-agnostic on purpose. The same loop runs the tuned and untuned models, and
the Ollama and Anthropic backends, so the only thing that differs between those
arms is the thing under test.

Three behaviours here were found the hard way and are load-bearing:

1. Bare SQL counts as a query. The model was trained and prompted to emit SQL,
   so it often ignores a QUERY: prefix. Demanding the prefix made it look as
   though the model never attempted the task.
2. An ANSWER is rejected unless a query actually ran. Otherwise the model answers
   from memory on the first step and the loop measures hallucination instead of
   tool use.
3. A PRAGMA is schema discovery, not an answer. Grading the last query outright
   scores a schema dump against the gold rows whenever the model looks something
   up on its final step.
"""

import re
import sqlite3
import time

from . import register
from ..dataset import DB_PATH
from ..grading import extract_sql

PROMPT = """You answer questions about a sqlite database.

Tables: {tables}

You do not know the column names. Look them up before writing a query:
PRAGMA table_info(<table>) returns a table's columns.

Write one sqlite query per reply and nothing else. I will run it and show you
the rows. Do not guess the answer from memory: you must run a query and see the
rows first. Only once a query has returned the answer, reply with
ANSWER: <the answer>. Do not repeat a query you have already run.
"""

PLACEHOLDER = re.compile(r"[<\[]?the answer[>\]]?[.\s]*", re.I)


def run_loop(generate, question, tables, max_steps=8, db=None, memory=None):
    con = sqlite3.connect(db or DB_PATH)
    transcript = PROMPT.format(tables=", ".join(tables))
    if memory is not None:
        transcript += "\n" + memory.prefix()
    transcript += f"\nQuestion: {question}\n"

    ran, errors, calls, tin, tout = [], 0, 0, 0, 0
    answer, last_text = "", ""
    t0 = time.perf_counter()
    try:
        for _ in range(max_steps):
            out = generate(transcript)
            text, a, b = out if isinstance(out, tuple) else (out, 0, 0)
            last_text, calls, tin, tout = text, calls + 1, tin + a, tout + b

            m = re.search(r"ANSWER:\s*(.+)", text)
            if m and PLACEHOLDER.fullmatch(m.group(1).strip()):
                m = None          # it echoed the prompt's placeholder
            if m:
                if any(q for q in ran):
                    answer = m.group(1).strip()
                    break
                transcript += ("\nYou have not run a query yet, so you cannot know "
                               "that. Run a query. Start with "
                               "PRAGMA table_info(<table>) to see a table's columns.\n")
                continue

            sql = extract_sql(text)
            if not sql:
                transcript += "\nReply with one sqlite query, or ANSWER: <the answer>.\n"
                continue

            ran.append(sql)
            try:
                rows = con.execute(sql).fetchall()
                transcript += f"QUERY: {sql}\nRESULT: {str(rows[:20])}\n"
                if memory is not None and not re.match(r"\s*PRAGMA\b", sql, re.I):
                    memory.add_worked(sql)
            except Exception as e:
                errors += 1
                transcript += f"QUERY: {sql}\nERROR: {e}\n"
    finally:
        con.close()

    answering = [q for q in ran if not re.match(r"\s*PRAGMA\b", q, re.I)]
    return {"sql": (answering or ran or [""])[-1], "raw": answer,
            "last_text": last_text, "queries_list": ran,
            "secs": round(time.perf_counter() - t0, 2),
            "model_calls": calls, "tokens_in": tin, "tokens_out": tout,
            "queries": len(ran), "queries_distinct": len({q.strip() for q in ran}),
            "errors": errors}


@register("loop")
def make(backend, memory=None, db=None, max_steps=8, **kw):
    from .oneshot import table_names
    tables = table_names(db)

    def arm(question):
        if hasattr(backend, "reset"):
            backend.reset()
        return run_loop(backend.generate, question, tables,
                        max_steps=max_steps, db=db, memory=memory)

    return arm
