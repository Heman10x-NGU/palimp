# Benchmarks

## Local Benchmark Command

GraphCtx includes a local benchmark command for measuring ingest and recall performance on your own machine:

```bash
graphctx benchmark --namespace bench --items 1000 --queries 100
```

This measures:

- Ingest total time (ms) and throughput (items/sec)
- Recall latency at p50, p95, and max for `fast`, `hybrid`, and `thinking` modes
- Average result count per query
- Database size (bytes)
- Embedding count
- Tombstoned item exclusion sanity check

## JSON Output

For structured output suitable for CI or automated tracking:

```bash
graphctx benchmark --namespace bench --items 1000 --queries 100 --json
```

## Important Caveats

**Results are machine-specific and not comparable to hosted platforms.**

GraphCtx runs locally on your hardware. Benchmark numbers depend on your CPU, disk speed, available memory, and SQLite configuration. They are not comparable to:

- Hosted memory platforms (Mem0, HydraDB, Zep) that run on dedicated infrastructure
- Benchmarks published with GPU-accelerated embedding models
- Results from cloud VMs with different hardware profiles

Use `graphctx benchmark` to:

- Track performance regressions across GraphCtx versions on the same machine
- Compare retrieval modes (fast vs. hybrid vs. thinking) on your hardware
- Validate that your SQLite configuration is performing as expected

Do not use `graphctx benchmark` to:

- Claim parity with hosted platforms
- Compare against published LoCoMo or LongMemEval scores
- Make production deployment decisions

## Mini Evaluation

For a correctness-focused evaluation (not a performance benchmark):

```bash
graphctx eval mini --namespace eval --json
```

This runs category-level pass/fail checks for single-hop recall, temporal facts, contradiction handling, namespace isolation, prompt injection safety, and deletion exclusion. See `graphctx eval mini --help` for details.

**This is a local mini-eval, not LongMemEval or LoCoMo.** It validates that GraphCtx's retrieval logic works correctly on your machine, not that it achieves competitive scores on academic benchmarks.
