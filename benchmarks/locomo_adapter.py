"""Palimp LoCoMo Benchmark Adapter.

Evaluates Palimp's retrieval quality against the LoCoMo-10 dataset.
Measures retrieval recall: can Palimp surface the evidence dialogues needed
to answer each QA pair?

Usage:
    python benchmarks/locomo_adapter.py \
        --dataset _refs/locomo-bench/data/locomo10.json \
        --output results/locomo_results.json
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


def _ingest_conversation(
    store: SQLiteStore,
    embedder: DeterministicEmbedder,
    ns: str,
    conversation: dict,
) -> dict[str, str]:
    """Ingest all conversation sessions into Palimp.

    Returns a mapping from dia_id (e.g. "D1:3") to episode_id.
    """
    dia_to_episode: dict[str, str] = {}

    # Find all session keys
    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")

    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_key = f"{session_key}_date_time"
        session_date = conversation.get(date_key, "")
        turns = conversation[session_key]

        if not isinstance(turns, list):
            continue

        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")

            if not text.strip():
                continue

            content = f"[{speaker}] {text}"
            source_ref = f"session:{session_num}:{dia_id}"

            episode_id = store.insert_episode(ns, content, "memory")
            store.insert_memory(ns, content, source_ref=source_ref)

            # Embed the episode
            vec = embedder.embed(content)
            from palimp.retriever import _vector_to_blob
            blob = _vector_to_blob(vec)
            store.insert_embedding(
                ns=ns,
                owner_type="episode",
                owner_id=episode_id,
                model="deterministic-sha256",
                dimension=embedder._dim,
                vector_blob=blob,
            )

            if dia_id:
                dia_to_episode[dia_id] = episode_id

    return dia_to_episode


def _evaluate_retrieval(
    engine: RecallEngine,
    ns: str,
    qa_items: list[dict],
    dia_to_episode: dict[str, str],
    top_k: int = 10,
    modes: list[str] | None = None,
) -> dict:
    """Evaluate retrieval recall for each QA item.

    For each question, we check if the retrieved episodes contain
    the evidence dialogue IDs needed to answer it.
    """
    if modes is None:
        modes = ["fast", "hybrid", "thinking"]

    results_by_mode = {}

    for mode in modes:
        per_question = []
        category_scores: dict[int, list[float]] = {}

        for qa in qa_items:
            question = qa["question"]
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            evidence_ids = qa.get("evidence", [])
            category = qa.get("category", 0)

            # Map evidence dia_ids to episode_ids
            evidence_episodes = set()
            for eid in evidence_ids:
                if eid in dia_to_episode:
                    evidence_episodes.add(dia_to_episode[eid])

            # Run recall
            output = engine.recall(ns, question, mode=mode, limit=top_k)
            retrieved_ids = {r.id for r in output.results}

            # Compute retrieval recall
            if evidence_episodes:
                hits = evidence_episodes & retrieved_ids
                recall = len(hits) / len(evidence_episodes)
            else:
                recall = 1.0  # No evidence needed

            per_question.append({
                "question": question,
                "answer": str(answer),
                "category": category,
                "evidence_ids": evidence_ids,
                "recall": round(recall, 4),
                "num_retrieved": len(output.results),
                "num_evidence": len(evidence_episodes),
                "hits": len(hits) if evidence_episodes else 0,
            })

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(recall)

        # Aggregate
        all_recalls = [q["recall"] for q in per_question]
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0

        category_avg = {}
        category_names = {
            1: "multi-hop",
            2: "single-hop",
            3: "temporal",
            4: "open-domain",
            5: "adversarial",
        }
        for cat, scores in sorted(category_scores.items()):
            cat_name = category_names.get(cat, f"cat-{cat}")
            category_avg[cat_name] = round(sum(scores) / len(scores), 4)

        results_by_mode[mode] = {
            "mode": mode,
            "top_k": top_k,
            "avg_recall": round(avg_recall, 4),
            "num_questions": len(per_question),
            "category_recall": category_avg,
            "per_question": per_question,
        }

    return results_by_mode


def main():
    parser = argparse.ArgumentParser(description="Palimp LoCoMo Benchmark Adapter")
    parser.add_argument("--dataset", required=True, help="Path to locomo10.json")
    parser.add_argument("--output", required=True, help="Output results JSON path")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K retrieval limit")
    parser.add_argument("--modes", nargs="+", default=["fast", "hybrid", "thinking"])
    args = parser.parse_args()

    print(f"Loading LoCoMo dataset from {args.dataset}...")
    with open(args.dataset) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} conversations")

    all_results = []
    overall_start = time.time()

    for idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", f"sample_{idx}")
        conversation = sample["conversation"]
        qa_items = sample["qa"]

        print(f"\n--- Sample {sample_id} ({len(qa_items)} questions) ---")

        # Create a fresh DB for each conversation
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="palimp_locomo_")
        os.close(fd)
        os.remove(db_path)

        try:
            store = SQLiteStore(db_path)
            embedder = DeterministicEmbedder(dim=384)
            engine = RecallEngine(store=store, embedder=embedder)
            ns = f"locomo_{sample_id}"

            # Ingest conversation
            t0 = time.time()
            dia_to_episode = _ingest_conversation(store, embedder, ns, conversation)
            ingest_time = time.time() - t0
            print(f"  Ingested {len(dia_to_episode)} dialogue turns in {ingest_time:.2f}s")

            # Evaluate retrieval
            t0 = time.time()
            mode_results = _evaluate_retrieval(
                engine, ns, qa_items, dia_to_episode,
                top_k=args.top_k, modes=args.modes,
            )
            eval_time = time.time() - t0

            for mode, result in mode_results.items():
                print(f"  {mode}: avg_recall={result['avg_recall']:.4f}")
                for cat, score in result["category_recall"].items():
                    print(f"    {cat}: {score:.4f}")

            all_results.append({
                "sample_id": sample_id,
                "num_turns": len(dia_to_episode),
                "num_questions": len(qa_items),
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

    # Compute overall averages across all samples
    overall_summary = {}
    for mode in args.modes:
        all_recalls = []
        category_all: dict[str, list[float]] = {}
        for sample_result in all_results:
            mode_result = sample_result["results_by_mode"].get(mode, {})
            if "avg_recall" in mode_result:
                all_recalls.append(mode_result["avg_recall"])
            for cat, score in mode_result.get("category_recall", {}).items():
                if cat not in category_all:
                    category_all[cat] = []
                category_all[cat].append(score)

        overall_summary[mode] = {
            "avg_recall": round(sum(all_recalls) / len(all_recalls), 4) if all_recalls else 0.0,
            "category_recall": {
                cat: round(sum(scores) / len(scores), 4)
                for cat, scores in sorted(category_all.items())
            },
        }

    output_data = {
        "benchmark": "LoCoMo-10",
        "system": "Palimp v0.3.0",
        "embedder": "deterministic-sha256",
        "top_k": args.top_k,
        "total_time_s": round(total_time, 2),
        "overall_summary": overall_summary,
        "per_sample": all_results,
    }

    # Strip per_question details for summary (they're huge)
    summary_data = json.loads(json.dumps(output_data))
    for sample in summary_data["per_sample"]:
        for mode_key in sample.get("results_by_mode", {}):
            sample["results_by_mode"][mode_key].pop("per_question", None)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Also save full details
    full_path = args.output.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"LoCoMo-10 Results (Palimp v0.3.0, deterministic embeddings)")
    print(f"{'='*60}")
    for mode, summary in overall_summary.items():
        print(f"\n  {mode.upper()} mode (top-{args.top_k}):")
        print(f"    Overall Recall: {summary['avg_recall']:.4f}")
        for cat, score in summary["category_recall"].items():
            print(f"    {cat}: {score:.4f}")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
