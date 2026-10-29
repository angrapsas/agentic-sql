"""Grading: run the model's SQL and the gold SQL, compare result sets.

No model grades another model. Every branch here was tested against the real
database in the phase 3 harness; behaviour is unchanged.
"""

import re
import sqlite3

from .dataset import DB_PATH


def _norm(v):
    """Compare 4.0 and 4, and "USA " and "USA", as equal. Round floats so
    35.4998711 and 35.4999 are not treated as different answers."""
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(round(float(v), 4))
    return str(v).strip()


def _rows(sql, db=None):
    con = sqlite3.connect(db or DB_PATH)
    try:
        return con.execute(sql).fetchall(), None
    except Exception as e:
        return None, str(e)
    finally:
        con.close()


REFUSAL = re.compile(
    r"\b(cannot|can't|cant|no such column|not available|unanswerable|"
    r"does not exist|doesn't exist|not in the schema|no column)\b", re.I)


def grade(sql, gold, raw_text="", db=None):
    """Returns (correct, loose, detail).

    For an unanswerable question (gold is None) the correct behaviour is to
    refuse, so producing any SQL is wrong.
    """
    if gold is None:
        refused = not sql.strip() or bool(REFUSAL.search(raw_text or sql))
        return refused, refused, "refused" if refused else "answered an unanswerable question"

    if not sql.strip():
        return False, False, "no sql produced"

    got, err = _rows(sql, db)
    if err:
        return False, False, f"sql error: {err}"

    want, gerr = _rows(gold, db)
    if gerr:
        return False, False, f"GOLD BROKEN: {gerr}"

    # Rows as a multiset, each row a sorted tuple of normalised cells. Sorting
    # the cells means column order does not matter, which is what we want when
    # the question is "who spent the most" and the model returns surname first.
    key = lambda rs: sorted(tuple(sorted(_norm(c) for c in r)) for r in rs)
    if key(got) == key(want):
        return True, True, "exact row match"

    # Loose: every gold value is present, but the shape differs, e.g. the model
    # returned the artist name plus the track count where the gold returned the
    # name alone. Right answer, wrong shape. A numerically near miss is NOT
    # loose: 88 against 88.1159 is simply wrong.
    flat_got = {_norm(c) for r in got for c in r}
    flat_want = {_norm(c) for r in want for c in r}
    if flat_want and flat_want <= flat_got:
        return False, True, "values present but shape differs (loose match)"

    return False, False, f"wrong rows: got {got[:3]} want {want[:3]}"


FENCE = "```"


def extract_sql(text):
    """Pull the SQL out of a model's reply.

    With thinking on, Qwen emits a closing </think> with no opening tag, so
    stripping matched <think>...</think> pairs is not enough: keep only what
    follows the LAST </think>. Then cut the query at a blank line, since the
    model often follows it with prose such as "Let me run this query."
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    # the loop writes its transcript as QUERY:/RESULT:, and models copy that
    # format back into their SQL if we let them
    text = re.sub(r"^\s*(?:RESULT|ERROR):.*$", " ", text, flags=re.M)

    for block in re.findall(FENCE + r"(?:sql)?\s*(.*?)" + FENCE, text, flags=re.S):
        if re.search(r"\bSELECT\b|\bPRAGMA\b", block, re.I):
            return tidy_sql(block)

    m = re.search(r"\b(?:SELECT|PRAGMA)\b[\s\S]*?(?:;|\n\s*\n|$)", text, flags=re.I)
    return tidy_sql(m.group(0)) if m else ""


def tidy_sql(sql):
    """Trim the junk that rides along with a generated query."""
    sql = sql.strip().strip("`").strip()
    lines = [l for l in sql.splitlines() if not re.match(
        r"\s*(?:let me|i will|now i|actually|this (?:query|will))", l, re.I)]
    sql = "\n".join(lines).strip()
    return sql.rstrip(";").strip().strip("`").strip()
