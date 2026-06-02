"""Palimp LongMemEval Benchmark Adapter.

Evaluates Palimp's retrieval quality against the LongMemEval dataset.
Measures retrieval recall at session level: can Palimp surface the evidence sessions
needed to answer each QA pair?

Usage:
    python benchmarks/longmemeval_adapter.py \
        --dataset _refs/longmemeval-bench/data/longmemeval_s_cleaned.json \
        --output results/longmemeval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from palimp.embeddings import DeterministicEmbedder
from palimp.retriever import RecallEngine
from palimp.storage import SQLiteStore


from palimp.embeddings import BaseEmbedder

def _ingest_sessions(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    ns: str,
    sessions: list[list[dict]],
    session_ids: list[str],
) -> None:
    """Ingest all history sessions into Palimp at the session granularity."""
    for idx, (session, sess_id) in enumerate(zip(sessions, session_ids)):
        # Format the entire session as a single text block
        content_parts = []
        for turn in session:
            role = turn.get("role", "unknown").upper()
            text = turn.get("content", "")
            if text.strip():
                content_parts.append(f"[{role}]: {text}")

        content = "\n".join(content_parts)
        if not content.strip():
            continue

        # Ingest as a memory episode
        episode_id = store.insert_episode(
            ns=ns,
            content=content,
            source_type="memory",
            source_ref=sess_id,
        )
        store.insert_memory(ns=ns, content=content, source_ref=sess_id)

        # Generate and store embedding
        vec = embedder.embed(content)
        from palimp.retriever import _vector_to_blob
        blob = _vector_to_blob(vec)
        model_name = "deterministic-sha256" if isinstance(embedder, DeterministicEmbedder) else "http-embedder"
        store.insert_embedding(
            ns=ns,
            owner_type="episode",
            owner_id=episode_id,
            model=model_name,
            dimension=embedder.dimension,
            vector_blob=blob,
        )


def _evaluate_retrieval(
    engine: RecallEngine,
    ns: str,
    question: str,
    correct_session_ids: list[str],
    modes: list[str],
    top_ks: list[int],
) -> dict:
    """Evaluate retrieval recall for a single question.

    Returns scores for each mode and each K in top_ks.
    """
    correct_set = set(correct_session_ids)
    results_by_mode = {}

    for mode in modes:
        # We retrieve up to max(top_ks)
        max_k = max(top_ks)
        output = engine.recall(ns, question, mode=mode, limit=max_k)
        
        # Extract session IDs from the retrieved episodes
        retrieved_sess_ids = []
        for result in output.results:
            source_ref = None
            if result.provenance:
                source_ref = result.provenance[0].get("source_ref")
            if source_ref:
                retrieved_sess_ids.append(source_ref)

        mode_scores = {}
        for k in top_ks:
            retrieved_k = retrieved_sess_ids[:k]
            retrieved_set = set(retrieved_k)

            # Compute recall_any
            recall_any = 1.0 if (correct_set & retrieved_set) else 0.0

            # Compute recall_all
            if correct_set:
                recall_all = 1.0 if correct_set.issubset(retrieved_set) else 0.0
                recall_avg = len(correct_set & retrieved_set) / len(correct_set)
            else:
                recall_all = 1.0
                recall_avg = 1.0

            mode_scores[f"recall_any@{k}"] = round(recall_any, 4)
            mode_scores[f"recall_all@{k}"] = round(recall_all, 4)
            mode_scores[f"recall_avg@{k}"] = round(recall_avg, 4)

        results_by_mode[mode] = {
            "mode": mode,
            "retrieved_session_ids": retrieved_sess_ids,
            "scores": mode_scores,
        }

    return results_by_mode


def main():
    parser = argparse.ArgumentParser(description="Palimp LongMemEval Benchmark Adapter")
    parser.add_argument("--dataset", required=True, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--output", required=True, help="Output results JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples to run (0 for all)")
    parser.add_argument("--modes", nargs="+", default=["fast", "hybrid", "thinking"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10], help="Top-K recall limits to evaluate")
    parser.add_argument("--embedder-type", default="deterministic", choices=["deterministic", "http"])
    parser.add_argument("--embedder-endpoint", default="http://localhost:11434/v1/embeddings")
    parser.add_argument("--embedder-model", default="nomic-embed-text")
    parser.add_argument("--embedder-dim", type=int, default=768)
    args = parser.parse_args()

    print(f"Loading LongMemEval dataset from {args.dataset}...")
    with open(args.dataset) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} questions")
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limiting execution to first {args.limit} samples")

    all_results = []
    overall_start = time.time()

    # Track overall statistics
    for idx, sample in enumerate(samples):
        q_id = sample["question_id"]
        q_type = sample.get("question_type", "unknown")
        question = sample["question"]
        correct_ids = sample.get("answer_session_ids", [])
        sessions = sample["haystack_sessions"]
        session_ids = sample["haystack_session_ids"]

        print(f"[{idx+1}/{len(samples)}] Sample {q_id} ({q_type}) - {len(sessions)} sessions...")

        # Create a fresh DB for each sample
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="palimp_lme_")
        os.close(fd)
        os.remove(db_path)

        try:
            store = SQLiteStore(db_path)
            if args.embedder_type == "http":
                from palimp.embeddings import HttpEmbedder
                embedder = HttpEmbedder(
                    endpoint=args.embedder_endpoint,
                    api_key="",
                    model=args.embedder_model,
                    dim=args.embedder_dim,
                )
            else:
                embedder = DeterministicEmbedder(dim=384)
            engine = RecallEngine(store=store, embedder=embedder)
            ns = f"lme_{q_id}"

            # Ingest sessions
            t0 = time.time()
            _ingest_sessions(store, embedder, ns, sessions, session_ids)
            ingest_time = time.time() - t0

            # Evaluate retrieval
            t0 = time.time()
            mode_results = _evaluate_retrieval(
                engine=engine,
                ns=ns,
                question=question,
                correct_session_ids=correct_ids,
                modes=args.modes,
                top_ks=args.top_k,
            )
            eval_time = time.time() - t0

            all_results.append({
                "question_id": q_id,
                "question_type": q_type,
                "num_sessions": len(sessions),
                "num_evidence": len(correct_ids),
                "ingest_time_s": round(ingest_time, 3),
                "eval_time_s": round(eval_time, 3),
                "results_by_mode": mode_results,
            })

        finally:
            del store
            try:
                os.remove(db_path)
            except OSError:
                pass

    total_time = time.time() - overall_start

    # Aggregate overall stats
    overall_summary = {}
    for mode in args.modes:
        mode_aggregates = {}
        for k in args.top_k:
            mode_aggregates[f"recall_any@{k}"] = []
            mode_aggregates[f"recall_all@{k}"] = []
            mode_aggregates[f"recall_avg@{k}"] = []

        type_aggregates = {}  # type -> { metric: [scores] }

        for result in all_results:
            q_type = result["question_type"]
            if q_type not in type_aggregates:
                type_aggregates[q_type] = {}
                for k in args.top_k:
                    type_aggregates[q_type][f"recall_any@{k}"] = []
                    type_aggregates[q_type][f"recall_all@{k}"] = []
                    type_aggregates[q_type][f"recall_avg@{k}"] = []

            mode_res = result["results_by_mode"].get(mode, {})
            scores = mode_res.get("scores", {})

            for k in args.top_k:
                for metric in (f"recall_any@{k}", f"recall_all@{k}", f"recall_avg@{k}"):
                    if metric in scores:
                        val = scores[metric]
                        mode_aggregates[metric].append(val)
                        type_aggregates[q_type][metric].append(val)

        # Average them
        mode_summary = {
            "overall": {
                metric: round(sum(vals) / len(vals), 4) if vals else 0.0
                for metric, vals in mode_aggregates.items()
            },
            "by_type": {}
        }
        for q_type, metrics in type_aggregates.items():
            mode_summary["by_type"][q_type] = {
                metric: round(sum(vals) / len(vals), 4) if vals else 0.0
                for metric, vals in metrics.items()
            }

        overall_summary[mode] = mode_summary

    output_data = {
        "benchmark": "LongMemEval-S",
        "system": "Palimp v0.3.0",
        "embedder": args.embedder_model if args.embedder_type == "http" else "deterministic-sha256",
        "top_ks": args.top_k,
        "total_time_s": round(total_time, 2),
        "overall_summary": overall_summary,
        "per_question": all_results,
    }

    # Strip retrieved_session_ids from summary to save space
    summary_data = json.loads(json.dumps(output_data))
    for item in summary_data["per_question"]:
        for mode_key in item.get("results_by_mode", {}):
            item["results_by_mode"][mode_key].pop("retrieved_session_ids", None)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_data, f, indent=2)

    full_path = args.output.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    embedder_label = args.embedder_model if args.embedder_type == "http" else "deterministic-sha256"
    print(f"LongMemEval-S Results (Palimp v0.3.0, {embedder_label} embeddings)")
    print(f"{'='*60}")
    for mode, summary in overall_summary.items():
        print(f"\n  {mode.upper()} mode:")
        for metric, val in sorted(summary["overall"].items()):
            print(f"    {metric}: {val:.4f}")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
