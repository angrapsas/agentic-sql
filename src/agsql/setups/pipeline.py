"""The component pipeline: router -> schema linker -> SQL generator -> repair.

Each stage has a fixed contract and two interchangeable versions, "code" and
"model", chosen per arm in the grid:

    setup: pipeline
    setup_args: {router: model, linker: code, generator: model, repair: code}

Contracts (what every version of a stage takes and returns):

    router(question)                  -> {"answerable": bool}
    linker(question)                  -> {"tables": [table, ...]}   connected by FKs
    generator(question, tables)       -> {"sql": str}               "" means no answer
    repair(question, tables, sql)     -> {"sql": str}               "" if it cannot be made to run

A refusal is an empty SQL string, which the grader scores as a refusal.

Every answer carries a trace of what each stage decided, so each component can
be scored on its own (router against the gold answerable flag, linker against
the tables the gold SQL uses), not only the pipeline end to end.

The coded stages hard-code Chinook's names on purpose: that is what hand-written
code does, and the renamed schema then measures what a schema change costs it.
"""

import re
import sqlite3
import time
from collections import deque

from . import register
from .templates import COMPILED as TEMPLATE_RULES
from ..dataset import DB_PATH
from ..grading import extract_sql

STAGES = ("router", "linker", "generator", "repair")


# ------------------------------------------------------------------ schema

class Schema:
    """Tables, columns and foreign keys of whatever database we point at."""

    def __init__(self, db=None):
        self.db = db or DB_PATH
        con = sqlite3.connect(self.db)
        try:
            self.tables = [r[0] for r in con.execute(
                "select name from sqlite_master where type='table'")]
            self.columns = {t: [c[1] for c in con.execute('PRAGMA table_info("%s")' % t)]
                            for t in self.tables}
            self.edges = {t: set() for t in self.tables}
            for t in self.tables:
                for fk in con.execute('PRAGMA foreign_key_list("%s")' % t):
                    other = fk[2]
                    if other in self.edges:
                        self.edges[t].add(other)
                        self.edges[other].add(t)
        finally:
            con.close()

    def text(self, tables=None):
        tables = tables or self.tables
        return "\n".join("%s(%s)" % (t, ", ".join(self.columns[t])) for t in tables)

    def connect(self, tables):
        """Add the tables needed to join the given ones (shortest FK paths)."""
        tables = [t for t in tables if t in self.edges]
        if len(tables) < 2:
            return tables
        keep = {tables[0]}
        for target in tables[1:]:
            if target in keep:
                continue
            prev, seen, q = {}, set(keep), deque(keep)
            while q:
                cur = q.popleft()
                if cur == target:
                    break
                for n in self.edges[cur]:
                    if n not in seen:
                        seen.add(n)
                        prev[n] = cur
                        q.append(n)
            node = target
            while node in prev:
                keep.add(node)
                node = prev[node]
            keep.add(target)
        return [t for t in self.tables if t in keep]

    def run(self, sql):
        con = sqlite3.connect(self.db)
        try:
            con.execute(sql).fetchall()
            return None
        except Exception as e:
            return str(e)
        finally:
            con.close()


# ------------------------------------------------------------- code stages

# Data the schema does not hold. A developer writing a router by hand lists the
# things users ask about that the database cannot answer.
OUT_OF_SCHEMA = re.compile(
    r"\b(rating|rated|review|salary|salaries|commission|follower|label|royalt|lyric|"
    r"explicit|discount|shipping|payment method|musical key|key is|remaster|"
    r"subscription|renew|skipped|skip|visit|how old|age of|date of birth of the customer|"
    r"birthday of|birth date of the customer|form\b|formed|release year|released in|"
    r"stock|streams?|plays?\b|played|listeners?)", re.I)

# words people use -> Chinook tables. Deliberately ordinary: what a developer
# would write, not tuned to the question set.
LEXICON = [
    (r"\b(tracks?|songs?|tunes?|composers?|composed|wrote|songwriter|duration|"
     r"milliseconds|minutes|bytes|file size|unit price|cost|priced?)\b", "Track"),
    (r"\b(albums?|records?)\b", "Album"),
    (r"\b(artists?|bands?|musicians?|performers?)\b", "Artist"),
    (r"\b(genres?|styles?|jazz|rock|metal|blues|pop|latin|classical|opera)\b", "Genre"),
    (r"\b(media types?|formats?|video|audio|mpeg|aac)\b", "MediaType"),
    (r"\b(playlists?)\b", "Playlist"),
    (r"\b(customers?|shoppers?|clients?|buyers?)\b", "Customer"),
    (r"\b(employees?|staff|workers?|reps?|support|managers?|manages|reports? to|hired)\b",
     "Employee"),
    (r"\b(invoices?|orders?|sales?|revenue|spent|spend|billed|billing|purchases?|"
     r"bought|sold|total)\b", "Invoice"),
    (r"\b(items?|invoice lines?|quantity|copies)\b", "InvoiceLine"),
]
LEXICON = [(re.compile(p, re.I), t) for p, t in LEXICON]


def code_router(schema, question):
    return {"answerable": not OUT_OF_SCHEMA.search(question)}


def code_linker(schema, question):
    found = [t for pat, t in LEXICON if pat.search(question) and t in schema.columns]
    # a playlist question is about tracks in it; sales of tracks need the lines
    if "Playlist" in found and "Track" in found:
        found.append("PlaylistTrack")
    if "Invoice" in found and "Track" in found:
        found.append("InvoiceLine")
    return {"tables": schema.connect(list(dict.fromkeys(found)))}


def code_generator(schema, question, tables):
    q = question.strip()
    for pat, tmpl in TEMPLATE_RULES:
        m = pat.search(q)
        if m:
            return {"sql": tmpl.format(*m.groups())}
    return {"sql": ""}


def code_repair(schema, question, tables, sql):
    # validate only: never pass on a query that does not run
    error = schema.run(sql) if sql else None
    if error:
        return {"sql": "", "error": error, "repaired": False}
    return {"sql": sql}


# ------------------------------------------------------------ model stages

ROUTER_PROMPT = (
    "Database schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Can this question be answered with a SQL query over exactly this schema? "
    "Answer only YES or NO.")

LINKER_PROMPT = (
    "Database schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Which tables are needed to answer it, including any needed only for joins? "
    "Reply with the table names, comma-separated, and nothing else.")

GENERATOR_PROMPT = (
    "Tables:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Write one sqlite query that answers it. Output only the SQL. "
    "If these tables cannot answer it, reply exactly: CANNOT ANSWER")

REPAIR_PROMPT = (
    "Tables:\n{schema}\n\n"
    "Question: {question}\n\n"
    "This query failed:\n{sql}\n\nError: {error}\n\n"
    "Write a corrected sqlite query. Output only the SQL.")


class Calls:
    """Adds up model cost across stages."""

    def __init__(self):
        self.n = self.tin = self.tout = 0
        self.secs = 0.0

    def __call__(self, backend, prompt):
        t0 = time.perf_counter()
        text, tin, tout = backend.generate(prompt)
        self.secs += time.perf_counter() - t0
        self.n += 1
        self.tin += tin or 0
        self.tout += tout or 0
        return text


def model_router(schema, question, backend, calls):
    text = calls(backend, ROUTER_PROMPT.format(schema=schema.text(), question=question))
    word = re.sub(r"<think>.*?</think>", " ", text, flags=re.S).strip().upper()
    return {"answerable": not word.startswith("NO"), "said": text.strip()[:40]}


def model_linker(schema, question, backend, calls):
    text = calls(backend, LINKER_PROMPT.format(schema=schema.text(), question=question))
    lower = {t.lower(): t for t in schema.tables}
    named = [lower[w.lower()] for w in re.findall(r"[A-Za-z_]+", text) if w.lower() in lower]
    return {"tables": schema.connect(list(dict.fromkeys(named)))}


def model_generator(schema, question, tables, backend, calls):
    text = calls(backend, GENERATOR_PROMPT.format(
        schema=schema.text(tables or None), question=question))
    if re.search(r"CANNOT ANSWER", text, re.I) and not re.search(r"\bSELECT\b", text, re.I):
        return {"sql": ""}
    return {"sql": extract_sql(text)}


def model_repair(schema, question, tables, sql, backend, calls):
    if not sql:
        return {"sql": ""}
    error = schema.run(sql)
    if not error:
        return {"sql": sql}
    text = calls(backend, REPAIR_PROMPT.format(
        schema=schema.text(tables or None), question=question, sql=sql, error=error))
    fixed = extract_sql(text)
    if fixed and not schema.run(fixed):
        return {"sql": fixed, "repaired": True}
    return {"sql": "", "error": error, "repaired": False}


# ---------------------------------------------------------------- pipeline

@register("pipeline")
def make(backend=None, memory=None, db=None, router="code", linker="code",
         generator="code", repair="code", **kw):
    choice = {"router": router, "linker": linker, "generator": generator, "repair": repair}
    for stage, v in choice.items():
        if v not in ("code", "model"):
            raise SystemExit(f"pipeline {stage} must be code or model, got {v!r}")
        if v == "model" and backend is None:
            raise SystemExit(f"pipeline {stage}=model needs a backend")
    schema = Schema(db)

    def arm(question):
        calls = Calls()
        trace = {"config": dict(choice)}
        t0 = time.perf_counter()

        r = (model_router(schema, question, backend, calls) if router == "model"
             else code_router(schema, question))
        trace["router"] = r
        sql = ""
        if r["answerable"]:
            ln = (model_linker(schema, question, backend, calls) if linker == "model"
                  else code_linker(schema, question))
            trace["linker"] = ln
            g = (model_generator(schema, question, ln["tables"], backend, calls)
                 if generator == "model" else code_generator(schema, question, ln["tables"]))
            trace["generator"] = {"sql": g["sql"][:300]}
            rp = (model_repair(schema, question, ln["tables"], g["sql"], backend, calls)
                  if repair == "model" else code_repair(schema, question, ln["tables"], g["sql"]))
            trace["repair"] = {k: (v[:200] if isinstance(v, str) else v) for k, v in rp.items()}
            sql = rp["sql"]

        return {"sql": sql, "raw": "" if sql else "CANNOT ANSWER",
                "secs": round(time.perf_counter() - t0, 2),
                "tokens_in": calls.tin, "tokens_out": calls.tout,
                "model_calls": calls.n, "queries": 1 if sql else 0,
                "queries_distinct": 1 if sql else 0, "errors": 0, "trace": trace}

    return arm
