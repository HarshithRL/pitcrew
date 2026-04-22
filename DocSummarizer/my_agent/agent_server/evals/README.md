# Evals

Three-stage evaluation harness for the document summarizer: parser, per-chunk, end-to-end.

## Layout

```
evals/
├── predict_fns.py          # 3 predict_fn variants (parser / chunk / doc)
├── dataset.py              # build eval rows from client trios
├── run_eval.py             # entry point
├── scorers/
│   ├── parser_scorers.py   # A1-A9 — deterministic
│   ├── chunk_scorers.py    # B1-B7 — LLM judges + code
│   └── doc_scorers.py      # C1-C9 — LLM judges + code
└── README.md
```

## Data layout

Drop one directory per document under `eval_data/`:

```
eval_data/
├── doc_001/
│   ├── source.pdf          # or source.md / source.txt
│   ├── parsed.md           # client reference parsed markdown (optional)
│   └── summary.md          # client reference summary (optional)
├── doc_002/
│   └── source.pdf
└── ...
```

Rows still run without `parsed.md` / `summary.md` — only the scorers that
depend on those ground-truth files become no-ops.

## Run

From `agent_server/`:

```bash
python -m evals.run_eval --stage all    --data-root ./eval_data
python -m evals.run_eval --stage parser --data-root ./eval_data
python -m evals.run_eval --stage chunk  --data-root ./eval_data
python -m evals.run_eval --stage doc    --data-root ./eval_data
```

Requires the same Databricks auth env vars as the agent (`DATABRICKS_HOST`
+ `DATABRICKS_TOKEN`, or Apps-issued `CLIENT_ID` / `CLIENT_SECRET`) plus
`MLFLOW_EXPERIMENT_NAME` (defaults to `/Shared/document-summarizer-eval`).

## Run on Databricks (recommended)

Sync the code + upload data once, then open the notebook.

```bash
# 1. Sync agent_server so the notebook + packages land in the workspace
databricks sync ./agent_server /Users/harshith.r@diggibyte.com/agent_server \
  --profile dbc-6afb4d73-0485

# 2. Upload per-doc folders to the shared volume
for dir in ./eval_data/*/; do
  name=$(basename "$dir")
  databricks fs cp -r "$dir" \
    "dbfs:/Volumes/pitcrew_databricks_dev/default/eval_data/$name" \
    --profile dbc-6afb4d73-0485
done
```

Open `/Users/harshith.r@diggibyte.com/agent_server/evals/run_on_databricks`
and click **Run all**. Default `DATA_ROOT` points at
`/Volumes/pitcrew_databricks_dev/default/eval_data`.

## Scorer catalog

| ID | Scorer | Stage | Reads from expectations |
|----|--------|-------|-------------------------|
| A1 | extraction_completeness | parser | expected_min_chars |
| A2 | section_count_match | parser | expected_section_count |
| A3 | heading_path_f1 | parser | expected_paths |
| A4 | hierarchy_depth | parser | expected_depth |
| A5 | chunk_boundary_integrity | parser | — |
| A6 | no_oversize_chunk | parser | max_chunk_chars |
| A7 | no_sibling_collision | parser | — |
| A8 | chars_tolerance_vs_reference | parser | reference_markdown |
| A9 | headings_match_reference | parser | reference_markdown |
| B1 | grounded (Guidelines) | chunk | — |
| B2 | on_topic (Guidelines) | chunk | — |
| B3 | format_compliance | chunk | — |
| B4 | length_bounds | chunk | min_bullets / max_bullets / max_chars |
| B5 | number_preservation | chunk | — |
| B6 | no_prompt_leakage | chunk | — |
| B7 | Correctness | chunk | expected_response |
| C1 | hierarchy_preservation | doc | reference_source_markdown / expected_paths |
| C2 | coverage (Guidelines) | doc | — |
| C3 | no_sibling_merge | doc | reference_source_markdown |
| C4 | Correctness | doc | expected_response / expected_facts |
| C6 | coherent (Guidelines) | doc | — |
| C7 | hierarchy_matches_reference (Guidelines) | doc | — |
| C8 | Safety | doc | — |
| C9 | no_pii (Guidelines) | doc | — |

## Wiring more data

Best eval set is real production traces labeled by SMEs:

1. `mlflow.search_traces()` to pull prod rows.
2. Curate into a UC Delta table (`main.eval.doc_golden_v1`) matching the
   schema in `agent-eval-skill/references/eval-dataset-patterns.md`.
3. Point `dataset.build_doc_rows` at a loader that reads from the table
   instead of `eval_data/`.
