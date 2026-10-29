# chinook_v1

43 natural-language questions over the Chinook sample database, each paired with a
hand-written SQL query that produces the correct answer. Built to compare different
ways of building one software component: a query layer that takes a question and
returns an answer.

## Files

| File | What it is |
|---|---|
| `chinook_v1.csv` | The dataset, readable in any spreadsheet |
| `chinook_v1.json` | Same content for code to load, plus the category and bucket maps |
| `verify_dataset.py` | Rebuilds the database from source and checks every gold query |

## Database

Chinook 1.4.5, from `lerocha/chinook-database`, file `Chinook_Sqlite.sql`. 11 tables,
3503 tracks, 275 artists, 59 customers, 412 invoices dated 2021-01-01 to 2025-12-22.

The dataset does not ship the database. `verify_dataset.py` downloads the SQL and
builds `Chinook.db`, so everyone runs against a byte-identical database.

## Columns

- `id` — stable identifier. Never reused, never renumbered.
- `question` — the natural-language question, as a user would type it.
- `gold_sql` — a query that returns the correct answer. Empty for unanswerable questions.
- `answerable` — false for the 4 questions the schema cannot answer.
- `category_id`, `category_name` — one of 14 fine-grained categories, for drilling down.
- `bucket` — one of 3 coarse buckets, for reporting.
- `gold_row_count` — how many rows the gold query returns. Flags the tie cases.
- `notes` — why the gold is what it is, where that is not obvious.

## Buckets

Reporting happens at bucket level because 3 or 4 questions per fine-grained category
cannot support a percentage. The buckets are chosen to match the hypothesis under
test: schema knowledge and tool access should help where the difficulty is knowing
what things are called, and not where the difficulty is reasoning.

**vocabulary** (16) — the model needs to know the schema's names. Single-table lookup,
filtered count, aggregate, multi-hop join, vocabulary mismatch.

**logic** (18) — the difficulty is constructing the right query. Superlative, self-join,
anti-join, temporal, nested comparison, ratio.

**judgment** (9) — the difficulty is knowing what the question even asks. Ambiguous
entity, ties, unanswerable.

## Grading

Execute the model's SQL, execute the gold SQL, compare result sets. No model grades
another model.

- **Correct** — exact match. Rows compared as a multiset; cells within a row are sorted
  before comparison, so column order does not matter, and floats are rounded to 4 decimal
  places.
- **Loose** — every gold value appears in the returned rows, but the shape differs.
  Returning `(Iron Maiden, 213)` where the gold returns `(Iron Maiden)` is loose, not
  correct. Numerically near misses are not loose: `88` against `88.1159` is simply wrong.
- **Unanswerable questions** are correct when the system declines. Producing any SQL for
  them is incorrect. Note that a template-matching baseline scores these correctly for
  free, since it emits nothing for anything it does not recognise, so this bucket should
  be reported with that asterisk.

## Known soft spots

Documented rather than removed, because they are the questions that distinguish systems.

- Two playlists are named "Music" (ids 1 and 8). The gold counts both. A system that
  picks one is arguably more right than the gold.
- Two tracks are named "Black Dog" with different composers. The gold returns both.
- 213 tracks tie at the maximum unit price, and 58 customers tie at the maximum invoice
  count. The gold returns the full tie set, so a system returning one row is wrong. This
  is a specification choice, not a fact about the data.
- "Length of the longest song in minutes" requires real division. Integer division gives
  88 rather than 88.1159. Graded wrong, but it is a near miss worth reporting separately.
- Questions use the data's own spellings, including accents where they occur in names.

## Rules of use

1. **Frozen.** Questions and gold queries are never edited in place. Changes ship as
   `chinook_v2`, so results from different weeks stay comparable.
2. **Same set across arms.** Every configuration under comparison runs the identical
   question set. Different question sets across arms makes the comparison meaningless.
3. **Not for training.** These are held out. The fine-tuning data is 118 separately
   generated question/SQL pairs with no overlap with this set.

## Statistical note

At n=43 the 95% confidence interval on a single accuracy figure is roughly ±15
percentage points, so differences smaller than that are not conclusive on their own.
Because every configuration runs the identical questions, comparisons between
configurations are paired and therefore tighter than that figure suggests, and should
be analysed as paired rather than as two independent proportions.

The per-question cost measures (tokens, model calls, queries, distinct queries, SQL
errors) are averages over 43 observations and are considerably more stable than
accuracy. Where a result rests on cost rather than accuracy, it rests on firmer ground.
