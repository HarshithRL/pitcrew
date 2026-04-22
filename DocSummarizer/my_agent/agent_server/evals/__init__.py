"""MLflow 3 + Mosaic AI evaluation harness for the document-summarizer agent.

Three stages, each with its own predict_fn + scorer set:

    A. Parser            evals.scorers.parser_scorers   (deterministic)
    B. Per-chunk summary evals.scorers.chunk_scorers    (LLM judge + code)
    C. End-to-end doc    evals.scorers.doc_scorers      (LLM judge + code)

Entry point: ``python -m evals.run_eval --stage parser|chunk|doc|all``.
"""
