# Benchmarks

## Local Benchmark Command

Palimp includes a local benchmark command for measuring ingest and recall performance on your own machine:

```bash
palimp benchmark --namespace bench --items 1000 --queries 100
```

This measures:

- Ingest total time (ms) and throughput (items/sec)
- Recall latency at p50, p95, and max for `fast`, `hybrid`, and `thinking` modes
- Average result count per query
- Database size (bytes)
- Embedding count
- Tombstoned item exclusion sanity check
- Graph max hops
- Temporal filter enabled
- Alias dedup enabled
- Reranker enabled/disabled

## JSON Output

For structured output suitable for CI or automated tracking:

```bash
palimp benchmark --namespace bench --items 1000 --queries 100 --json
```

## Important Caveats

**Results are machine-specific and not comparable to hosted platforms.**

Palimp runs locally on your hardware. Benchmark numbers depend on your CPU, disk speed, available memory, and SQLite configuration. They are not comparable to:

- Hosted memory platforms (Mem0, HydraDB, Zep) that run on dedicated infrastructure
- Benchmarks published with GPU-accelerated embedding models
- Results from cloud VMs with different hardware profiles

Palimp reports local benchmarks and mini-evals. Standard LoCoMo/LongMemEval/BEAM parity requires running the same benchmark harness and should not be claimed from local synthetic tests.

Use `palimp benchmark` to:

- Track performance regressions across Palimp versions on the same machine
- Compare retrieval modes (fast vs. hybrid vs. thinking) on your hardware
- Validate that your SQLite configuration is performing as expected

Do not use `palimp benchmark` to:

- Claim parity with hosted platforms
- Compare against published LoCoMo or LongMemEval scores
- Make production deployment decisions

## Mini Evaluation

For a correctness-focused evaluation (not a performance benchmark):

```bash
palimp eval mini --namespace eval --json
```

This runs category-level pass/fail checks for single-hop recall, temporal facts, contradiction handling, namespace isolation, prompt injection safety, and deletion exclusion. See `palimp eval mini --help` for details.

**This is a local mini-eval, not LongMemEval or LoCoMo.** It validates that Palimp's retrieval logic works correctly on your machine, not that it achieves competitive scores on academic benchmarks.

## v0.3 Expanded Mini-Eval

The v0.3 mini-eval expanded from 9 to 15 categories:

**Existing (9):**
single-hop preference, static knowledge, temporal current, temporal historical, contradiction warning, namespace isolation, prompt injection safety, deletion exclusion, multi-hop placeholder

**Added in v0.3 (6):**
actual 2-hop bridge, actual 3-hop bridge (when enabled), alias dedup, category priority under budget, trigger keyword recall, runbook gotcha pack, no-answer abstention / low-confidence suppression

```bash
palimp eval mini --namespace eval --json
```

## v0.3 Performance Benchmark

Extended benchmark reports v0.3 configuration details:

```bash
palimp benchmark --items 10000 --queries 300 --json
```

Output includes: graph max hops, temporal filter status, alias dedup status, reranker status, and per-mode latency.

## Standard Benchmark Adapter

Palimp provides a retrieval-only adapter for standard memory benchmarks. This adapter supports LongMemEval retrieval-only evaluation as a first target, with LoCoMo retrieval-only as a secondary target.

> **Important:** Palimp reports local benchmarks and mini-evals. Standard LoCoMo/LongMemEval/BEAM parity requires running the same benchmark harness and should not be claimed from local synthetic tests.

The adapter lives in `benchmarks/palimp_adapter/` and uses `mem0ai/memory-benchmarks` shapes for ingest, search, and evaluate/predict-only operations.

## Sources

Benchmark and design context drawn from:

- Mem0 memory evaluation docs: https://docs.mem0.ai/core-concepts/memory-evaluation
- Mem0 token-efficient memory research/benchmarks: https://mem0.ai/research
- Mem0 temporal reasoning benchmark discussion: https://mem0.ai/blog/introducing-temporal-reasoning-in-mem0
- Graphiti/Zep temporal context graph framing: https://www.getzep.com/platform/graphiti/
- LongMemEval-V2 "experienced colleague" benchmark framing: https://arxiv.org/abs/2605.12493 and https://xiaowu0162.github.io/longmemeval-v2/
- MemX local-first memory design: https://arxiv.org/abs/2603.16171
- MemReranker reasoning-aware reranking context: https://arxiv.org/abs/2605.06132
