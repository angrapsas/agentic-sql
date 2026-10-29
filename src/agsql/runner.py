"""Running one arm over a question set, with checkpointing.

An arm is a (backend, setup, thinking, memory, dataset) combination. It is just a
function from a question to a dict with a `sql` key, plus whatever metrics it can
report. Everything else here is bookkeeping.
"""

import json
import os
import time

from .grading import grade


def run_arm(name, fn, questions, out=None, resume=True, verbose=True,
            on_question=None):
    """fn(question_text) -> dict with 'sql' and optional metric keys.

    Writes `out` after every question and skips questions already recorded in
    it, so a crashed run costs one question rather than the whole arm. Rerunning
    the same command resumes. Pass resume=False when the configuration has
    changed, or one file ends up holding results from two different setups.
    """
    results = []
    if out and resume and os.path.exists(out):
        try:
            results = json.load(open(out))
            if results:
                print(f"resuming {name}: {len(results)} of {len(questions)} already done")
        except (json.JSONDecodeError, OSError):
            results = []
    done = {r["id"] for r in results}

    for i, q in enumerate(questions, 1):
        if q.id in done:
            continue
        t0 = time.perf_counter()
        try:
            r = fn(q.question) or {}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            r = {"sql": "", "crash": f"{type(e).__name__}: {e}"}
        secs = r.get("secs", round(time.perf_counter() - t0, 2))

        sql = (r.get("sql") or "").strip()
        correct, loose, detail = grade(sql, q.gold_sql, r.get("raw", ""))
        if r.get("crash"):
            detail = f"crash: {r['crash']}"

        row = {
            "arm": name, "id": q.id, "question": q.question,
            "bucket": q.bucket, "category_id": q.category_id,
            "correct": correct, "loose": loose, "detail": detail,
            "sql": sql,
            "raw": (r.get("raw") or "")[:300],
            "last_text": (r.get("last_text") or "")[:300],
            "queries_list": r.get("queries_list", []),
            "secs": secs,
            "tokens_in": r.get("tokens_in", 0),
            "tokens_out": r.get("tokens_out", 0),
            "model_calls": r.get("model_calls", 1),
            "queries": r.get("queries", 0),
            "queries_distinct": r.get("queries_distinct", r.get("queries", 0)),
            "errors": r.get("errors", 0),
        }
        results.append(row)

        if verbose:
            flag = "OK " if correct else ("~  " if loose else "XX ")
            print(f"{i:3d}/{len(questions)} [{q.bucket[:4]}] {flag} {secs:7.1f}s  "
                  f"{detail[:58]:<60} {q.question[:46]}", flush=True)
        if on_question:
            on_question(row)
        if out:
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(results, f, indent=1)
            os.replace(tmp, out)   # atomic, so a kill mid-write cannot corrupt it

    return results


def load_results(path):
    with open(path) as f:
        return json.load(f)
