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


def _decompose_lme_date(date_str: str) -> str:
    """Decompose an LME date string (e.g. '2023/04/10 (Mon) 17:50') into FTS-matchable words."""
    if not date_str:
        return ""
    import re
    # Strip day of week parenthetical e.g. '(Mon)' to make parsing clean
    clean_str = re.sub(r"\([A-Za-z]+\)", "", date_str).strip()
    try:
        from dateutil import parser as date_parser
        dt = date_parser.parse(clean_str)
        month_name = dt.strftime("%B").lower()  # 'april'
        month_abbr = dt.strftime("%b").lower()  # 'apr'
        day_of_week = dt.strftime("%A").lower()  # 'monday'
        day_of_week_abbr = dt.strftime("%a").lower()  # 'mon'
        return f"date: {dt.year} {month_name} {month_abbr} {dt.day} {day_of_week} {day_of_week_abbr}"
    except Exception:
        # Fallback to simple split logic if parsing fails
        parts = date_str.split()[0].split('/')
        if len(parts) == 3:
            year, month, day = parts
            months_map = {
                "01": "january jan", "02": "february feb", "03": "march mar",
                "04": "april apr", "05": "may may", "06": "june jun",
                "07": "july jul", "08": "august aug", "09": "september sep",
                "10": "october oct", "11": "november nov", "12": "december dec"
            }
            m_words = months_map.get(month, "")
            return f"date: {year} {m_words} {int(day)}"
        return ""


def _ingest_sessions(
    store: SQLiteStore,
    embedder: BaseEmbedder,
    extractor: BaseExtractor | None,
    ns: str,
    sessions: list[list[dict]],
    session_ids: list[str],
    dates: list[str],
    deadline: float | None = None,
    heartbeat_state: list[float] | None = None,
    heartbeat_seconds: float = 15.0,
) -> None:
    """Ingest all history sessions into Palimp at the session granularity."""
    for idx, (session, sess_id) in enumerate(zip(sessions, session_ids)):
        _check_deadline(deadline, f"ingest session {idx + 1}/{len(sessions)}")
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

        # Decompose corresponding date
        if idx < len(dates):
            date_words = _decompose_lme_date(dates[idx])
            if date_words:
                content += f"\nSession {date_words}"

        # Ingest memory using the public ingestion pipeline
        with _timebox(deadline, f"ingest session {idx + 1}/{len(sessions)}"):
            ingest_memory(
                store=store,
                embedder=embedder,
                extractor=extractor,
                ns=ns,
                content=content,
                source_ref=sess_id,
                extract=True if extractor else False,
                category="memory",
            )
        if heartbeat_state is not None:
            _heartbeat(
                heartbeat_state,
                heartbeat_seconds,
                f"LongMemEval ingest progress: {idx + 1}/{len(sessions)} sessions",
            )


def _evaluate_retrieval(
    engine: RecallEngine,
    ns: str,
    question: str,
    correct_session_ids: list[str],
    modes: list[str],
    top_ks: list[int],
    deadline: float | None = None,
    heartbeat_state: list[float] | None = None,
    heartbeat_seconds: float = 15.0,
) -> dict:
    """Evaluate retrieval recall for a single question.

    Returns scores for each mode and each K in top_ks.
    """
    correct_set = set(correct_session_ids)
    results_by_mode = {}

    for mode in modes:
        _check_deadline(deadline, f"evaluate mode {mode}")
        # We retrieve up to max(top_ks)
        max_k = max(top_ks)
        with _timebox(deadline, f"recall mode {mode}"):
            output = engine.recall(ns, question, mode=mode, limit=max_k)
        if heartbeat_state is not None:
            _heartbeat(
                heartbeat_state,
                heartbeat_seconds,
                f"LongMemEval evaluated mode={mode}",
            )
        
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
    parser.add_argument("--max-sample-seconds", type=float, default=0.0, help="Abort after one sample exceeds this many seconds (0 disables)")
    parser.add_argument("--max-total-seconds", type=float, default=0.0, help="Stop after total run exceeds this many seconds (0 disables)")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0, help="Print watchdog heartbeat after this many seconds of work")
    
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
    run_status = "ok"

    # Track overall statistics
    for idx, sample in enumerate(samples):
        if args.max_total_seconds > 0 and time.time() - overall_start > args.max_total_seconds:
            run_status = "total_timeout"
            print(f"[watchdog] total deadline hit before sample {idx + 1}; writing partial results")
            break

        q_id = sample["question_id"]
        q_type = sample.get("question_type", "unknown")
        question = sample["question"]
        correct_ids = sample.get("answer_session_ids", [])
        sessions = sample["haystack_sessions"]
        session_ids = sample["haystack_session_ids"]
        dates = sample.get("haystack_dates", [])

        print(f"[{idx+1}/{len(samples)}] Sample {q_id} ({q_type}) - {len(sessions)} sessions...")

        # Create a fresh DB for each sample
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="palimp_lme_")
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
            ns = f"lme_{q_id}"

            # Ingest sessions
            t0 = time.time()
            _ingest_sessions(
                store,
                embedder,
                extractor,
                ns,
                sessions,
                session_ids,
                dates,
                deadline=sample_deadline,
                heartbeat_state=heartbeat_state,
                heartbeat_seconds=args.heartbeat_seconds,
            )
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
                deadline=sample_deadline,
                heartbeat_state=heartbeat_state,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            eval_time = time.time() - t0

            all_results.append({
                "status": "ok",
                "question_id": q_id,
                "question_type": q_type,
                "num_sessions": len(sessions),
                "num_evidence": len(correct_ids),
                "ingest_time_s": round(ingest_time, 3),
                "eval_time_s": round(eval_time, 3),
                "results_by_mode": mode_results,
            })
        except TimeoutError as exc:
            run_status = "sample_timeout"
            print(f"[watchdog] safety stop on sample {q_id}: {exc}")
            all_results.append({
                "status": "timeout",
                "question_id": q_id,
                "question_type": q_type,
                "num_sessions": len(sessions),
                "num_evidence": len(correct_ids),
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
        },
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
