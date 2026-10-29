# agentic-sql

A test harness for comparing ways of answering natural-language questions over a
SQL database.

The task stays the same throughout: a question goes in, the system produces a SQL
query, the query runs against the database, and the resulting rows are compared
against a hand-written correct answer. What changes is how the system is built.
Each configuration differs from the previous one in a single way, so the
comparison isolates that one change.

| Configuration | What it is |
|---|---|
| templates | Hand-written regex patterns mapped to SQL. No model involved. |
| oneshot | The model sees the table names and writes one query. No tools, no retries. |
| loop | The same model can run queries, read the rows, and try again. |
| loop + adapter | The same loop, with a fine-tuned LoRA adapter on the model. |
| loop + memory | The same loop, carrying queries that worked across questions. |

Each of these can run with thinking on or off, and against either a local model
or an API model. Those options make up the rest of the grid.

For every question the harness records whether the answer was correct, how long
it took, input and output tokens, how many times the model was called, how many
queries it ran, how many of those were distinct, and how many SQL errors it hit.

## Quick start

```bash
pip install -r requirements.txt
python data/verify_dataset.py          # builds Chinook.db, checks every gold query
export PYTHONPATH=src

python -m agsql.cli arms               # list the configurations
python -m agsql.cli run --arm all --smoke   # one question per configuration
python -m agsql.cli run --arm all      # the full grid
python -m agsql.cli report             # the tables
```

Start with `--smoke`. It runs one question through every configuration in about
two minutes, which is usually enough to catch a misconfigured backend or a
prompt the model ignores. Those problems are much cheaper to find there than an
hour into a full run.

## Layout

```
data/        the question set, frozen. See data/DATASET.md
configs/     grid.yaml: one entry per configuration being measured
src/agsql/
  dataset.py   loading questions and deterministic subsets
  grading.py   execute both queries, compare rows. No model grades a model
  metrics.py   aggregation and the two report tables
  memory.py    carrying verified queries across questions
  runner.py    run one configuration, checkpointing after every question
  cli.py       entry point
  backends/    how a model is served: transformers, Ollama, Anthropic, mock
  setups/      what scaffold it runs in: templates, oneshot, loop
results/     one json per configuration, with per-question records. Committed
report/      generated tables
```

### Why backends and setups are separate

A backend is how a model is served. A setup is the scaffold it runs inside.

They are separate modules because of a problem in earlier work. The fine-tuned
model could only be served through transformers, while the agent ran on Ollama,
so two configurations differed in both the scaffold and the serving path even
though they were compared as though only the fine-tune had changed. Splitting the
two means you cannot change one by accident while meaning to change the other.

## Adding a configuration

Add an entry to `configs/grid.yaml`:

```yaml
  qwen8b-loop:
    label: "8b loop"
    backend: hf
    setup: loop
    thinking: false
    backend_args:
      model_id: Qwen/Qwen3.5-8B
```

Configurations live in the config file rather than in command-line flags, so the
grid can be reproduced from the repo instead of from someone's shell history.

## Reading the numbers

`correct` means the returned rows match the expected rows exactly. `loose` also
counts a right answer in the wrong shape, such as returning `(Iron Maiden, 213)`
where the expected answer is just the name. A numerically close answer does not
count as loose.

`queries` counts every query the model ran, and `uniq_q` counts the distinct
ones. When those two numbers diverge, the model was repeating itself, which is a
failure to stop rather than the cost of exploring an unfamiliar schema. They are
reported separately for that reason.

The cost columns are per-question averages and are more stable than accuracy. At
43 questions, a single accuracy figure carries roughly ±15 points of 95%
confidence interval, so smaller differences are not conclusive by themselves.
Every configuration runs the same questions, which makes comparisons between them
paired and somewhat tighter than that figure suggests.

## Running on Colab

`notebooks/colab_runner.ipynb` clones the repo, installs it, and calls the CLI.
Results are written to Drive so that a disconnect does not lose them. The grid
runs cheapest configurations first, so a run that dies partway through still
leaves most of the chart, and rerunning the same command picks up from the last
completed question.

## Training the adapter

```bash
python scripts/make_training_data.py       # 164 pairs, verified against the database
python scripts/train_adapter.py --out /content/drive/MyDrive/sql_checker/sql_adapter_2b
```

The generator fills templates with real values from the database, checks that
every query runs and returns rows, and drops anything that overlaps the
evaluation set. The training prompt is imported from the oneshot setup rather
than written out again, so the adapter is trained on the same format it is later
measured on.

One known gap: the prompt asks the model to reply `CANNOT ANSWER` when the schema
cannot answer a question, but none of the training pairs demonstrate that, since
every pair has a working query. The adapter is therefore expected to do worse
than the base model on the unanswerable questions, and that difference should not
be read as fine-tuning having hurt.

## Status

The dataset and harness are working, with 26 tests that run without a GPU. The Anthropic configurations skip unless `ANTHROPIC_API_KEY` and
`AGSQL_ANTHROPIC_MODEL` are set.

One result from earlier work has not been explained and is worth knowing before
reading any numbers here. A LangGraph agent running on Ollama, with no
fine-tuning, scored considerably higher than the hand-written loop on the same
model weights. Turning thinking on in the loop closed about a third of that gap
while costing five times as much, so reasoning does not account for it. The most
likely remaining explanation is the tool-calling interface: structured tool calls
cannot be corrupted by surrounding prose, while a text protocol can, and several
of the failures were exactly that. Adding a structured-tool-call setup is the
next step.
