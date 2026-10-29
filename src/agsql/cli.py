"""Command line entry point.

    python -m agsql.cli arms                      # what is defined
    python -m agsql.cli run --arm qwen2b-loop     # one arm from the grid
    python -m agsql.cli run --arm all --smoke     # 1 question per arm, proves plumbing
    python -m agsql.cli run --arm all             # the whole grid
    python -m agsql.cli report                    # tables from results/

An arm is named once, in configs/grid.yaml, and never spelled out on the command
line for a real run. That way the chart is reproducible from the repo rather than
from someone's shell history.
"""

import argparse
import glob
import json
import os
import sys
import time

from . import dataset
from .backends import get_backend
from .memory import Memory
from .metrics import arm_table, bucket_table, failures
from .runner import run_arm
from .setups import get_setup

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GRID = os.path.join(ROOT, "configs", "grid.yaml")
RESULTS = os.environ.get("AGSQL_RESULTS", os.path.join(ROOT, "results"))


def load_grid(path=GRID):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    arms = cfg.get("arms") or {}
    defaults = cfg.get("defaults") or {}
    for name, a in arms.items():
        merged = dict(defaults)
        merged.update(a)
        arms[name] = merged
    return arms, cfg


def build_arm(spec, results_dir):
    """Turn a grid entry into a callable plus its memory store."""
    backend_kw = dict(spec.get("backend_args") or {})
    if "thinking" in spec:
        backend_kw["thinking"] = bool(spec["thinking"])

    backend = None
    if spec["backend"] != "none":
        backend = get_backend(spec["backend"], **backend_kw)

    mem = None
    if spec.get("memory"):
        mem = Memory(path=os.path.join(results_dir, spec["_name"] + ".memory.json"))

    setup_kw = dict(spec.get("setup_args") or {})
    arm = get_setup(spec["setup"], backend=backend, memory=mem, **setup_kw)
    return arm, mem, backend


def cmd_run(args):
    arms, cfg = load_grid(args.grid)
    if args.arm == "all":
        names = [n for n in arms if not arms[n].get("skip")]
    else:
        names = [a.strip() for a in args.arm.split(",")]
        for n in names:
            if n not in arms:
                raise SystemExit(f"unknown arm {n!r}. defined: {sorted(arms)}")

    os.makedirs(args.results, exist_ok=True)
    questions_all = dataset.load(args.dataset or cfg.get("dataset", "chinook_v1"))

    print(f"{len(names)} arm(s): {', '.join(names)}")
    if args.smoke:
        print("SMOKE MODE: 1 question per arm, results not kept\n")

    t_start = time.perf_counter()
    for i, name in enumerate(names, 1):
        spec = dict(arms[name])
        spec["_name"] = name

        qs = questions_all
        if spec.get("limit"):
            qs = dataset.subset(qs, limit=spec["limit"])
        if spec.get("per_bucket"):
            qs = dataset.subset(qs, per_bucket=spec["per_bucket"])
        if args.smoke:
            qs = qs[:1]

        out = None if args.smoke else os.path.join(args.results, f"{name}.json")
        label = spec.get("label", name)

        print(f"\n[{i}/{len(names)}] {label}  ({len(qs)} questions)")
        print(f"    backend={spec['backend']} setup={spec['setup']} "
              f"thinking={bool(spec.get('thinking'))} memory={bool(spec.get('memory'))}")
        try:
            arm, mem, backend = build_arm(spec, args.results)
        except SystemExit as e:
            print(f"    SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"    SKIPPED: could not build backend: {type(e).__name__}: {e}")
            continue

        t0 = time.perf_counter()
        run_arm(label, arm, qs, out=out, resume=not args.no_resume,
                verbose=not args.quiet)
        print(f"    done in {(time.perf_counter()-t0)/60:.1f} min"
              + (f", memory holds {len(mem)} notes" if mem else ""))

        del backend   # free the GPU before the next arm loads a model

    print(f"\ntotal {(time.perf_counter()-t_start)/60:.1f} min")
    if not args.smoke:
        print()
        cmd_report(args)


def cmd_report(args):
    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    files = [f for f in files if not f.endswith(".memory.json")]
    if not files:
        raise SystemExit(f"no results in {args.results}")
    all_res = [json.load(open(f)) for f in files]
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    arm_table(all_res, out=os.path.join(ROOT, "report", "arms.txt"))
    print()
    bucket_table(all_res, out=os.path.join(ROOT, "report", "buckets.txt"))
    print(f"\nwrote report/arms.txt and report/buckets.txt from {len(files)} arm(s)")
    if getattr(args, "failures", False):
        for rs in all_res:
            print(f"\n=== {rs[0]['arm']}")
            failures(rs)


def cmd_arms(args):
    arms, cfg = load_grid(args.grid)
    print(f"dataset: {cfg.get('dataset', 'chinook_v1')}\n")
    for name, a in arms.items():
        flags = []
        if a.get("thinking"):
            flags.append("thinking")
        if a.get("memory"):
            flags.append("memory")
        if a.get("skip"):
            flags.append("SKIP")
        print(f"  {name:<26} {a['backend']:<10} {a['setup']:<10} "
              f"{' '.join(flags)}")
        if a.get("note"):
            print(f"      {a['note']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="agsql")
    p.add_argument("--grid", default=GRID)
    p.add_argument("--results", default=RESULTS)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one arm, several, or all")
    r.add_argument("--arm", required=True,
                   help="arm name, comma-separated names, or 'all'")
    r.add_argument("--dataset", default=None)
    r.add_argument("--smoke", action="store_true",
                   help="1 question per arm, nothing written. Run this first.")
    r.add_argument("--no-resume", action="store_true",
                   help="start arms over instead of continuing a checkpoint")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="build the tables from results/")
    rp.add_argument("--failures", action="store_true")
    rp.set_defaults(func=cmd_report)

    a = sub.add_parser("arms", help="list the arms defined in the grid")
    a.set_defaults(func=cmd_arms)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
