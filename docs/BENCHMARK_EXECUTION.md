# FactIndic Benchmark Execution & Quota-Aware Orchestration Guide

This document specifies the execution protocols, API call budgeting, cryptographic integrity verification, response caching architecture, and controlled run status for the FactIndic Six-Condition Benchmark Experiment.

---

## 1. Benchmark Dataset & Cryptographic Manifest

The full-scale benchmark dataset consists of **200 stratified records** sampled deterministically from the canonical XL-Sum test split:
- **100 English records** (`data/benchmark/benchmark_en.jsonl`)
- **100 Hindi records** (`data/benchmark/benchmark_hi.jsonl`)
- **Sampling seed**: `42`
- **Source split**: `test` (Commit `ab0484252a17cb12999710f4ccdd9966141a2fa3`)

### Manifest Fingerprints (`data/benchmark/benchmark_manifest.json`)

To preserve redistribution boundaries and prevent raw BBC text from being checked into version control, the benchmark dataset is cryptographically fingerprinted using SHA-256 checksums:

| Language | Records | Filename | SHA-256 Checksum |
|---|---|---|---|
| **English** | 100 | `benchmark_en.jsonl` | `bc8f5775d9ad1b5e983ae98f2329daac777cb1e4f6f3ef0a66b21730ffbe67fa` |
| **Hindi** | 100 | `benchmark_hi.jsonl` | `6eb026f0bf79a5a9b3b50ff5e42fed23d3e8bbda2be2cdc3d5daa99720735194` |

Cryptographic validation confirms:
- **Zero duplicate record IDs** across English and Hindi splits.
- **Zero empty source texts** or reference summaries.
- **Exact record count match** (100 English + 100 Hindi = 200 total).

---

## 2. API Quota Diagnosis & Budget Breakdown

### Quota Constraints
- **Trial Key Monthly Cap**: `1,000 API calls / month`.
- **Per-Minute Rate Limit**: `20 calls / minute` (`x-trial-endpoint-call-limit: 20`).
- **Production Tier**: `100+ calls / minute` without monthly volume caps.

### Mathematical Call Breakdown for 200 Records

Based on empirical distributions observed during the 20-summary controlled pilot (3.4 sentences/summary, ~2.088 claims/sentence, top-3 BM25 evidence retrieval):

| Stage | Pipeline Operation | Multiplier | Calls Required |
|---|---|---|---|
| **Generation** | Prompt A & Prompt B | 200 records x 2 prompts | **400 calls** |
| **C1 & C2** | Whole-Document NLI | 200 records x 2 summaries | **400 calls** |
| **C3 & C4** | Sentence-Level NLI | 400 summaries x 3.4 sentences | **1,360 calls** |
| **Decomposition** | Atomic Claim Extraction | 400 summaries x 3.4 sentences | **1,360 calls** |
| **C5 & C6** | Hybrid Claim NLI | 2,840 claims x 3 evidence passages | **8,520 calls** |
| **TOTAL** | **Full 6-Condition Experiment** | | **12,040 calls** |

### Execution Runtime Projections
- **Trial Key (20 RPM)**: $\frac{12,040}{20} = 602.0\text{ minutes} \approx \mathbf{10.03\text{ hours}}$.
- **Production Key (100 RPM)**: $\frac{12,040}{100} = 120.4\text{ minutes} \approx \mathbf{2.01\text{ hours}}$.

> [!CAUTION]
> **Full-Scale Execution Safeguard**: Executing the full 200-record benchmark requires **12,040 API calls**, which exceeds the Cohere trial key monthly allowance of 1,000 calls by **12x**. The preflight verification script (`scripts/preflight_check.py`) enforces this constraint and will refuse to start the full benchmark on a trial key to prevent mid-run quota exhaustion and corrupted state.

---

## 3. Persistent Response Caching Architecture

FactIndic implements a persistent, SQLite-backed response cache (`factindic/cache.py`) located at `data/cache/response_cache.db`:
- **WAL Mode (Write-Ahead Logging)**: Enables concurrent reads with serialized atomic writes.
- **Deterministic Keying**: Generates stable SHA-256 hashes (`compute_cache_key`) from normalized JSON payloads, ensuring key identity regardless of dictionary key insertion order.
- **Deduplication**: Automatically caches NLI premise-hypothesis pairs, extracted sentence claims, and generated summaries.
- **Resumption**: Re-running any pipeline stage reuses cached outputs, consuming 0 API calls for previously verified pairs.

---

## 4. Preflight Verification CLI

Always run preflight verification before triggering benchmark experiments:

```bash
# Check full 200-record benchmark readiness
python scripts/preflight_check.py

# Check controlled sample (10 records) readiness
python scripts/preflight_check.py --records 10

# Output machine-readable JSON report
python scripts/preflight_check.py --json
```

---

## 5. Controlled Run (10 Records) Status & Honest Provenance

The controlled experiment covers 10 source records (5 English, 5 Hindi) across conditions C1–C6 (60 evaluations total).

### Evaluation Status

| Source Record | Language | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `xlsum_en_uk-wales-33738390` | English | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS |
| `xlsum_en_uk-northern-ireland-54885209` | English | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS |
| `xlsum_en_world-europe-55173830` | English | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS | SUCCESS |
| `xlsum_en_uk-england-cambridgeshire-22952135` | English | SUCCESS | SUCCESS | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_en_world-latin-america-44779257` | English | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_hi_india-54887372` | Hindi | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_hi_institutional-55276383` | Hindi | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_hi_india-44673829` | Hindi | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_hi_world-51882940` | Hindi | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |
| `xlsum_hi_india-38294711` | Hindi | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED | QUOTA_BLOCKED |

- **Preserved Summaries**: All 20 generated summaries (10 Prompt A, 10 Prompt B) are preserved intact.
- **Preserved SUCCESS Evaluations**: 20 evaluations across Records 1–3 and Record 4 C1/C2 are preserved with empirical latencies and raw probabilities.
- **Quota Blocked**: 40 evaluations are documented honestly as `QUOTA_BLOCKED` due to Cohere's trial key monthly cap (1,000 API calls). No scores have been fabricated.

### Running Controlled Re-Verification in Dry-Run Mode

```bash
python scripts/reverify_controlled_run.py --dry-run
```
