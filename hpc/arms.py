"""Print the arms that run on a GPU node, comma-separated (or one per line with -n).

Set ARMS=a,b,c to choose arms; otherwise every local arm in the grid. Haiku arms
are never included: they are API calls, need no GPU, and compute nodes may have
no internet. Run those from a laptop or the front-end.
"""
import os
import sys
import yaml

g = yaml.safe_load(open("configs/grid.yaml"))
local = {n for n, a in g["arms"].items()
         if a.get("backend") in ("hf", "none") and not a.get("skip")}
if os.environ.get("ARMS"):
    arms = [a.strip() for a in os.environ["ARMS"].split(",") if a.strip()]
    bad = [a for a in arms if a not in local]
    if bad:
        sys.exit(f"not local arms in grid.yaml: {bad}")
else:
    arms = [n for n in g["arms"] if n in local]
print("\n".join(arms) if "-n" in sys.argv else ",".join(arms))
