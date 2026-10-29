# agentic-sql

Comparing ways of building one software component.

The component is a query layer over a SQL database: a natural-language question
goes in, an answer comes out. The question under test is whether decision logic
that is normally written as code can instead be made at runtime by a small
language model, and how small that model can be.

Built on the Agentic Software framing (Cao, arXiv 2606.05608v2). Traditional
software is `S = (C, D, E)` with static decision logic `D`. An agent is
`A = (M, T, 𝓜, Π)`. So `D` can live in four places, and this repo measures the
difference between them:

| Where D lives | Arm |
|---|---|
| In code `C` | hand-written patterns to SQL templates |
| In the weights `M` | the model answering in one shot, no tools |
| Discovered through tools `T` | the model in a loop that can query the database |
| In persistent memory `𝓜` | the loop, carrying verified queries across questions |

## Quick start

```bash
pip install -r requirements.txt
python data/verify_dataset.py          # builds Chinook.db, checks every gold query
export PYTHONPATH=src

python -m agsql.cli arms               # what is defined
python -m agsql.cli run --arm all --smoke   # 1 question per arm. always do this first
python -m agsql.cli run --arm all      # the grid
python -m agsql.cli report             # the tables
```

`--smoke` is not optional in spirit. It runs one question through every arm in a
couple of minutes and catches the broken-backend and broken-protocol failures
that otherwise surface an hour into a run.

## Layout

```
data/        the question set, frozen. See data/DATASET.md
configs/     grid.yaml: one entry per cell of the chart
src/agsql/
  dataset.py   loading questions and deterministic subsets
  grading.py   execute both queries, compare rows. No model grades a model
  metrics.py   aggregation and the two report tables
  memory.py    persistent memory across questions
  runner.py    run one arm, checkpointing after every question
  cli.py       entry point
  backends/    how a model is served: transformers, Ollama, Anthropic, mock
  setups/      what scaffold it runs in: templates, oneshot, loop
results/     one json per arm, per-question records. Committed
report/       generated tables
```

### Backends and setups are separate on purpose

A **backend** is how a model is served. A **setup** is the scaffold it runs in.
Keeping them apart is the main structural lesson from the previous phase, where
the fine-tuned model could only be served through transformers while the agent
ran on Ollama, and the two ended up compared as though the only difference were
the fine-tune. Changing the serving path should never silently change the
scaffold.

## Adding an arm

Add an entry to `configs/grid.yaml`. Nothing else. Arms are named in the config
and never spelled out on the command line for a real run, so the chart is
reproducible from the repo rather than from someone's shell history.

```yaml
  qwen8b-loop:
    label: "8b loop"
    backend: hf
    setup: loop
    thinking: false
    backend_args:
      model_id: Qwen/Qwen3.5-8B
```

## Metrics

Per question: correct, loose, wall time, input and output tokens, model calls,
queries, distinct queries, SQL errors, plus the full query trajectory.

- **correct** is an exact row match. **loose** also counts a right answer in the
  wrong shape, e.g. returning `(Iron Maiden, 213)` where the gold returns the
  name alone. A numerically near miss is not loose.
- **queries** counts every query run; **uniq_q** counts distinct ones. The gap is
  the model repeating itself, which is a termination failure rather than the cost
  of discovering a schema, and worth reporting separately.
- Cost columns are per-question averages over 43 observations and are
  considerably more stable than accuracy. At n=43 a single accuracy figure
  carries roughly ±15 points of 95% CI, so differences smaller than that are not
  conclusive on their own. Because every arm runs the identical questions,
  comparisons between arms are paired and tighter than that figure suggests.

## Running on Colab

`notebooks/colab_runner.ipynb` clones this repo, installs, and calls the CLI.
Results are written to Drive so a disconnect does not lose them. Arms are ordered
cheap first in the grid, so a run that dies partway still leaves most of the chart,
and rerunning the same command resumes from the per-question checkpoint.

## Status

The dataset and harness are in place and tested. The 2b adapter has not been
trained yet, so the tuned arms will skip until `sql_adapter_2b` exists. The
Anthropic arms skip unless `ANTHROPIC_API_KEY` and `AGSQL_ANTHROPIC_MODEL` are
set. Prior results and their caveats are written up in the project docs rather
than here.

One finding from the previous phase is unexplained and worth knowing before
reading any numbers: a LangGraph agent on Ollama, untuned, substantially
outperformed the hand-written loop on the same weights. Turning thinking on in
the loop closed about a third of that gap at five times the cost, so reasoning is
not the explanation. The leading suspect is the tool-calling interface, since
structured tool calls cannot be corrupted by surrounding prose while a text
protocol can, and several failures were exactly that. A structured-tool-call
setup is the next thing to add here.
