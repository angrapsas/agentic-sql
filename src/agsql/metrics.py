"""Aggregating per-question records into the two report tables."""

from collections import defaultdict

BUCKET_ORDER = ["vocabulary", "logic", "judgment"]


def _flatten(all_results):
    if all_results and isinstance(all_results[0], list):
        return [r for rs in all_results for r in rs]
    return list(all_results)


def _agg(rows):
    n = len(rows)
    if not n:
        return dict(n=0, acc=0.0, loose=0.0, secs=0.0, tok_in=0, tok_out=0,
                    calls=0.0, queries=0.0, uniq_q=0.0, looped=0, errors=0)
    uq = lambda r: r.get("queries_distinct", r.get("queries", 0))
    return {
        "n": n,
        "acc": round(100.0 * sum(r["correct"] for r in rows) / n, 1),
        "loose": round(100.0 * sum(r.get("loose", r["correct"]) for r in rows) / n, 1),
        "secs": round(sum(r["secs"] for r in rows) / n, 1),
        "tok_in": round(sum(r["tokens_in"] for r in rows) / n),
        "tok_out": round(sum(r["tokens_out"] for r in rows) / n),
        "calls": round(sum(r["model_calls"] for r in rows) / n, 1),
        "queries": round(sum(r["queries"] for r in rows) / n, 1),
        "uniq_q": round(sum(uq(r) for r in rows) / n, 1),
        "looped": sum(r["queries"] > uq(r) for r in rows),
        "errors": sum(r["errors"] for r in rows),
    }


def arm_table(all_results, out=None):
    """One row per arm. Cost columns are per-question averages; `errors` and
    `looped` are totals across the arm."""
    rows = _flatten(all_results)
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    hdr = (f"{'arm':<28} {'n':>3} {'acc%':>6} {'loose%':>7} {'s/q':>7} {'tok_in':>7} "
           f"{'tok_out':>8} {'calls':>6} {'queries':>8} {'uniq_q':>7} {'looped':>7} "
           f"{'sqlerr':>7}")
    lines = [hdr, "-" * len(hdr)]
    for arm, rs in by.items():
        a = _agg(rs)
        lines.append(
            f"{arm:<28} {a['n']:>3} {a['acc']:>6} {a['loose']:>7} {a['secs']:>7} "
            f"{a['tok_in']:>7} {a['tok_out']:>8} {a['calls']:>6} {a['queries']:>8} "
            f"{a['uniq_q']:>7} {a['looped']:>7} {a['errors']:>7}")
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
    return text


def bucket_table(all_results, out=None):
    """Accuracy per bucket, one column per arm. Shows n, because a bucket of 9
    and a bucket of 18 do not deserve the same confidence."""
    rows = _flatten(all_results)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    cell = defaultdict(list)
    for r in rows:
        cell[(r["bucket"], r["arm"])].append(r["correct"])

    w = max([len(a) for a in arms] + [7]) + 2
    lines = [f"{'bucket':<14} " + "".join(f"{a:>{w}}" for a in arms)]
    lines.append("-" * (14 + w * len(arms)))
    buckets = BUCKET_ORDER + sorted({r["bucket"] for r in rows} - set(BUCKET_ORDER))
    for b in buckets:
        line = f"{b:<14} "
        for a in arms:
            v = cell.get((b, a))
            line += f"{(f'{sum(v)}/{len(v)}') if v else '-':>{w}}"
        lines.append(line)
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
    return text


def failures(results, bucket=None, show_said=True):
    """Why each wrong answer was wrong. For loop arms the query counts matter as
    much as the SQL: 'no sql produced' with queries=0 means the model never
    wrote a query, which is a different failure from writing a broken one."""
    for r in results:
        if r["correct"] or (bucket is not None and r["bucket"] != bucket):
            continue
        q = r.get("queries", 0)
        uq = r.get("queries_distinct", q)
        print(f"[{r['bucket']}] {r['question']}")
        print(f"     why      {r['detail']}")
        print(f"     cost     {r.get('model_calls', 0)} calls, {q} queries "
              f"({uq} distinct), {r.get('errors', 0)} sql errors")
        print(f"     last sql {r['sql'][:200] or '(none)'}")
        said = r.get("raw") or r.get("last_text") or ""
        if show_said and said:
            print(f"     said     {said[:200]}")
        print()
