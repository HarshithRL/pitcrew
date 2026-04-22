# Databricks notebook source
# MAGIC %md
# MAGIC # Document-summarizer evaluation
# MAGIC
# MAGIC Runs parser / per-chunk / end-to-end evals against the client trios
# MAGIC stored in a UC Volume. Traces + per-row feedback land in the MLflow
# MAGIC experiment configured below.

# COMMAND ----------
# MAGIC %pip install -r /Workspace/Users/harshith.r@diggibyte.com/.bundle/summarizer-agent/default/files/agent_server/requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Make the agent's `core` + `evals` packages importable inside the notebook.
import sys
AGENT_SERVER = "/Workspace/Users/harshith.r@diggibyte.com/.bundle/summarizer-agent/default/files/agent_server"
if AGENT_SERVER not in sys.path:
    sys.path.insert(0, AGENT_SERVER)

# COMMAND ----------
# Configuration — adjust if the volume / experiment names differ.
DATA_ROOT = "/Volumes/pitcrew_databricks_dev/default/eval_data"
EXPERIMENT = "/Users/harshith.r@diggibyte.com/document-summarizer"  # same experiment as the agent
STAGE = "all"  # one of: parser | chunk | doc | all

# COMMAND ----------
import mlflow
import mlflow.langchain

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT)
mlflow.langchain.autolog()

# COMMAND ----------
from evals import dataset
from evals.predict_fns import chunk_predict, doc_predict, parser_predict
from evals.scorers import chunk_scorers, doc_scorers, parser_scorers
from evals.simple_eval import run_eval

summary = {}
if STAGE in ("parser", "all"):
    summary["parser"] = run_eval(
        "parser", dataset.build_parser_rows(DATA_ROOT),
        parser_predict, parser_scorers.ALL,
    )
if STAGE in ("chunk", "all"):
    summary["chunk"] = run_eval(
        "chunk", dataset.build_chunk_rows(DATA_ROOT),
        chunk_predict, chunk_scorers.ALL,
    )
if STAGE in ("doc", "all"):
    summary["doc"] = run_eval(
        "doc", dataset.build_doc_rows(DATA_ROOT),
        doc_predict, doc_scorers.ALL,
    )

# COMMAND ----------
import json
print(json.dumps(summary, indent=2, default=str))
