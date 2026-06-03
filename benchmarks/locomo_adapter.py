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
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from palimp.embeddings import DeterministicEmbedder, HttpEmbedder, BaseEmbedder
from palimp.extractor import HttpExtractor, RuleBasedExtractor, BaseExtractor
from palimp.ingest import ingest_memory
from palimp.parallel_extractor import AsyncHttpExtractor, CachedExtractor, HybridExtractor
from palimp.extraction_cache import ExtractionCache
from palimp.ingest_batch import ingest_batch_sync, BatchItem
from palimp.retriever import RecallEngine
from palimp.storage import SQLiteStore
from palimp.config import get_config


def _deadline(seconds: float) -> float | None:
    return time.monotonic() + seconds if seconds and seconds > 0 else None


def _check_deadline(deadline: float | None, label: str) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError(f"sample deadline exceeded during {label}")


def _heartbeat(state: list[float], heartbeat_seconds: float, message: str) -> None:
    if heartbeat_seconds <= 0:
        return
    now = time.monotonic()
    if now - state[0] >= heartbeat_seconds:
        print(f"[watchdog] {message}", flush=True)
        state[0] = now


@contextmanager
def _timebox(deadline: float | None, label: str):
    """Interrupt a single blocking benchmark operation once the sample deadline expires."""
    if deadline is None:
        yield
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"sample deadline exceeded before {label}")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"sample deadline exceeded during {label}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _extract_speaker_from_question(question: str, conversation: dict) -> str | None:
    """Extract person/speaker name from a question.

    Prioritizes matching the exact speakers defined in the conversation metadata.
    """
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    q_lower = question.lower()
    if speaker_a and speaker_a.lower() in q_lower:
        return speaker_a
    if speaker_b and speaker_b.lower() in q_lower:
        return speaker_b

    # Fallback to regex-based capital word matching
    import re
    stopwords = {
        "What", "Would", "Could", "Does", "Did", "How", "Why", "When", "Where",
        "Which", "Who", "Whose", "Will", "Can", "Should", "Has", "Have", "Had",
        "Is", "Are", "Was", "Were", "Do", "Been", "Being", "The", "Answer",
        "Yes", "No", "Likely", "Based", "According", "Since", "Because",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    }
    words = question.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", word)
        if len(clean_word) < 2 or clean_word in stopwords:
            continue
        if "'s" in word or "\u2019s" in word:
            continue
        if len(clean_word) > 1 and clean_word[0].isupper() and clean_word[1:].islower():
            if i == 0:
                continue
            return clean_word

    possessives = re.findall(r"\b([A-Z][a-z]+)['\u2019]s\b", question)
    if possessives:
        name = possessives[0]
        if name not in stopwords:
            return name
    return None


def _ingest_conversation(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    extractor: BaseExtractor | None,
    ns: str,
    conversation: dict,
    deadline: float | None = None,
    heartbeat_state: list[float] | None = None,
    heartbeat_seconds: float = 15.0,
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

    for session_key in session_keys:
        _check_deadline(deadline, f"ingest {session_key}")
        session_num = session_key.split("_")[1]
        date_key = f"{session_key}_date_time"
        session_date = conversation.get(date_key, "")
        turns = conversation[session_key]

        if not isinstance(turns, list):
            continue

        # Check if we should decompose date
        date_words = ""
        if session_date:
            from dateutil import parser as date_parser
            try:
                dt = date_parser.parse(session_date)
                month_name = dt.strftime("%B").lower()
                month_abbr = dt.strftime("%b").lower()
                date_words = f"date: {dt.year} {month_name} {month_abbr} {dt.day}"
            except Exception:
                pass

        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")

            if not text.strip():
                continue

            content = f"[{speaker}] {text}"
            if date_words:
                content += f" ({date_words})"
            source_ref = f"session:{session_num}:{dia_id}"

            # Ingest memory using the public ingestion pipeline
            with _timebox(deadline, f"ingest {source_ref}"):
                res = ingest_memory(
                    store=store,
                    embedder=embedder,
                    extractor=extractor,
                    ns=ns,
                    content=content,
                    source_ref=source_ref,
                    extract=True if extractor else False,
                    category="memory",
                )
            episode_id = res["episode_id"]

            if dia_id:
                dia_to_episode[dia_id] = episode_id
            if heartbeat_state is not None:
                _heartbeat(
                    heartbeat_state,
                    heartbeat_seconds,
                    f"LoCoMo ingest progress: {len(dia_to_episode)} dialogue turns",
                )

    return dia_to_episode


def _evaluate_retrieval(
    engine: RecallEngine,
    ns: str,
    qa_items: list[dict],
    dia_to_episode: dict[str, str],
    conversation: dict,
    top_k: int = 10,
    modes: list[str] | None = None,
    deadline: float | None = None,
    heartbeat_state: list[float] | None = None,
    heartbeat_seconds: float = 15.0,
) -> dict:
    """Evaluate retrieval recall for each QA item.

    For each question, we check if the retrieved episodes contain
    the evidence dialogue IDs needed to answer it.
    """
    if modes is None:
        modes = ["fast", "hybrid", "thinking"]

    results_by_mode = {}

    for mode in modes:
        _check_deadline(deadline, f"evaluate mode {mode}")
        per_question = []
        category_scores: dict[int, list[float]] = {}

        for qa in qa_items:
            _check_deadline(deadline, f"evaluate question in mode {mode}")
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
            with _timebox(deadline, f"recall mode={mode} question={len(per_question) + 1}"):
                output = engine.recall(ns, question, mode=mode, limit=top_k)
            results = output.results
            if heartbeat_state is not None:
                _heartbeat(
                    heartbeat_state,
                    heartbeat_seconds,
                    f"LoCoMo evaluated mode={mode}, questions={len(per_question) + 1}",
                )
            retrieved_ids = {r.id for r in results}

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
                "num_retrieved": len(results),
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
    parser.add_argument("--limit", type=int, default=0, help="Limit number of conversations to run (0 for all)")
    parser.add_argument("--qa-limit", type=int, default=0, help="Limit QA items per conversation (0 for all)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K retrieval limit")
    parser.add_argument("--modes", nargs="+", default=["fast", "hybrid", "thinking"])
    
    # Embedder options
    parser.add_argument("--embedder-type", default="deterministic", choices=["deterministic", "http"])
    parser.add_argument("--embedder-endpoint", default="http://localhost:11434/v1/embeddings")
    parser.add_argument("--embedder-model", default="nomic-embed-text")
    parser.add_argument("--embedder-dim", type=int, default=768)
    parser.add_argument("--embedder-timeout", type=float, default=10.0, help="HTTP embedder request timeout in seconds")
    
    # Extractor options
    parser.add_argument("--extractor-type", default="none", choices=["none", "rule", "http", "hybrid"])
    parser.add_argument("--extractor-endpoint", default="https://api.mimo.org/v1/chat/completions")
    parser.add_argument("--extractor-model", default="mimo-v2.5-pro")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent extraction calls")
    parser.add_argument("--use-cache", action="store_true", help="Use extraction cache")
    parser.add_argument("--extractor-timeout", type=float, default=30.0, help="HTTP extractor request timeout in seconds")

    # Watchdog options
    parser.add_argument("--max-sample-seconds", type=float, default=0.0, help="Abort after one conversation exceeds this many seconds (0 disables)")
    parser.add_argument("--max-total-seconds", type=float, default=0.0, help="Stop after total run exceeds this many seconds (0 disables)")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0, help="Print watchdog heartbeat after this many seconds of work")

    args = parser.parse_args()

    print(f"Loading LoCoMo dataset from {args.dataset}...")
    with open(args.dataset) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} conversations")
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limiting execution to first {args.limit} conversations")

    all_results = []
    overall_start = time.time()
    run_status = "ok"

    for idx, sample in enumerate(samples):
        if args.max_total_seconds > 0 and time.time() - overall_start > args.max_total_seconds:
            run_status = "total_timeout"
            print(f"[watchdog] total deadline hit before sample {idx + 1}; writing partial results")
            break

        sample_id = sample.get("sample_id", f"sample_{idx}")
        conversation = sample["conversation"]
        qa_items = sample["qa"]
        if args.qa_limit > 0:
            qa_items = qa_items[:args.qa_limit]

        print(f"\n--- Sample {sample_id} ({len(qa_items)} questions) ---")

        # Create a fresh DB for each conversation
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="palimp_locomo_")
        os.close(fd)
        os.remove(db_path)

        try:
            store = SQLiteStore(db_path)
            sample_deadline = _deadline(args.max_sample_seconds)
            heartbeat_state = [time.monotonic()]
            
            # Setup Embedder
            if args.embedder_type == "http":
                embedder = HttpEmbedder(
                    endpoint=args.embedder_endpoint,
                    api_key="",
                    model=args.embedder_model,
                    dim=args.embedder_dim,
                    timeout=args.embedder_timeout,
                )
            else:
                embedder = DeterministicEmbedder(dim=args.embedder_dim)
                
            # Setup Extractor
            if args.extractor_type == "http":
                extractor_key = os.environ.get("MIMO_API_KEY", os.environ.get("PALIMP_EXTRACTOR_API_KEY", ""))
                extractor = AsyncHttpExtractor(
                    endpoint=args.extractor_endpoint,
                    api_key=extractor_key,
                    model=args.extractor_model,
                    timeout=args.extractor_timeout,
                    max_concurrent=args.max_concurrent,
                )
                if args.use_cache:
                    extractor = CachedExtractor(extractor)
            elif args.extractor_type == "hybrid":
                extractor_key = os.environ.get("MIMO_API_KEY", os.environ.get("PALIMP_EXTRACTOR_API_KEY", ""))
                http_ext = AsyncHttpExtractor(
                    endpoint=args.extractor_endpoint,
                    api_key=extractor_key,
                    model=args.extractor_model,
                    timeout=args.extractor_timeout,
                    max_concurrent=args.max_concurrent,
                )
                extractor = HybridExtractor(http_ext)
                if args.use_cache:
                    extractor = CachedExtractor(extractor)
            elif args.extractor_type == "rule":
                extractor = RuleBasedExtractor()
            else:
                extractor = None

            engine = RecallEngine(store=store, embedder=embedder)
            ns = f"locomo_{sample_id}"

            # Ingest conversation
            t0 = time.time()
            dia_to_episode = _ingest_conversation(
                store,
                embedder,
                extractor,
                ns,
                conversation,
                deadline=sample_deadline,
                heartbeat_state=heartbeat_state,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            ingest_time = time.time() - t0
            print(f"  Ingested {len(dia_to_episode)} dialogue turns in {ingest_time:.2f}s")

            # Evaluate retrieval
            t0 = time.time()
            mode_results = _evaluate_retrieval(
                engine, ns, qa_items, dia_to_episode, conversation,
                top_k=args.top_k, modes=args.modes,
                deadline=sample_deadline,
                heartbeat_state=heartbeat_state,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            eval_time = time.time() - t0

            for mode, result in mode_results.items():
                print(f"  {mode}: avg_recall={result['avg_recall']:.4f}")
                for cat, score in result["category_recall"].items():
                    print(f"    {cat}: {score:.4f}")

            all_results.append({
                "status": "ok",
                "sample_id": sample_id,
                "num_turns": len(dia_to_episode),
                "num_questions": len(qa_items),
                "ingest_time_s": round(ingest_time, 3),
                "eval_time_s": round(eval_time, 3),
                "results_by_mode": mode_results,
            })
        except TimeoutError as exc:
            run_status = "sample_timeout"
            print(f"[watchdog] safety stop on sample {sample_id}: {exc}")
            all_results.append({
                "status": "timeout",
                "sample_id": sample_id,
                "num_questions": len(qa_items),
                "timeout_reason": str(exc),
                "results_by_mode": {},
            })
            break

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
        "scoring_profile": "fts5_primary",
        "weights": get_config().weights.to_dict(),
        "embedder": args.embedder_model if args.embedder_type == "http" else "deterministic-sha256",
        "extractor": args.extractor_type,
        "safety": {
            "run_status": run_status,
            "max_sample_seconds": args.max_sample_seconds,
            "max_total_seconds": args.max_total_seconds,
            "heartbeat_seconds": args.heartbeat_seconds,
            "embedder_timeout": args.embedder_timeout,
            "extractor_timeout": args.extractor_timeout,
            "qa_limit": args.qa_limit,
        },
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
    print(f"LoCoMo-10 Results (Palimp v0.3.0, {output_data['embedder']} embeddings)")
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
