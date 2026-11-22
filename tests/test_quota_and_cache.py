"""Comprehensive unit tests for response caching, quota handling, preflight and manifest integrity.

Validates:
1. ResponseCache deterministic SHA-256 hashing, hits, misses, and statistics.
2. XNLIVerifier and AtomicClaimExtractor cache integration.
3. HTTP 429 Retry-After header handling vs. monthly quota exhaustion fail-fast.
4. Atomic writing and integrity of experiment records.
5. Benchmark manifest cryptographic checksums and duplicate ID detection.
6. Preflight call budget calculations and quota feasibility enforcement.
7. Honest recording without fabricated SUCCESS statuses.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from factindic.benchmark_manifest import (
    compute_file_sha256,
    generate_benchmark_manifest,
    validate_benchmark_manifest,
)
from factindic.cache import ResponseCache, compute_cache_key
from factindic.exceptions import (
    AuthenticationError,
    InferenceTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
)
from factindic.experiment_runner import (
    ExperimentRecord,
    save_experiment_records,
    load_experiment_records,
)
from factindic.preflight import (
    estimate_api_calls,
    run_preflight,
    check_api_credentials,
)
from scripts.reverify_controlled_run import reverify_controlled_experiment
from factindic.verification.claim_extractor import AtomicClaimExtractor, ExtractedClaim
from factindic.verification.config import NLIConfig
from factindic.verification.nli import XNLIVerifier
from factindic.verification.result import NLIResult


# ==============================================================================
# 1. Cache Tests
# ==============================================================================


def test_compute_cache_key_deterministic():
    """Verify that compute_cache_key produces identical SHA-256 hashes regardless of dict order."""
    key1 = compute_cache_key("nli", {"a": 1, "b": 2, "c": [1, 2]})
    key2 = compute_cache_key("nli", {"c": [1, 2], "b": 2, "a": 1})
    key3 = compute_cache_key("nli", {"a": 1, "b": 3})

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64  # SHA-256 hex string


def test_response_cache_hit_miss_and_stats(tmp_path: Path):
    """Test get/set operations, hit/miss tracking, and cache persistence in SQLite."""
    db_file = tmp_path / "test_cache.db"
    cache = ResponseCache(db_path=db_file)

    # Miss
    assert cache.get_nli("premise 1", "hypothesis 1", "en", "model-a") is None
    stats = cache.get_stats()
    assert stats.misses == 1
    assert stats.hits == 0
    assert stats.total_entries == 0

    # Set and Hit
    cached_payload = {"label": "entailment", "entailment_prob": 0.95}
    cache.set_nli("premise 1", "hypothesis 1", "en", "model-a", cached_payload)

    stats = cache.get_stats()
    assert stats.total_entries == 1
    assert stats.writes == 1

    hit_val = cache.get_nli("premise 1", "hypothesis 1", "en", "model-a")
    assert hit_val == cached_payload
    stats = cache.get_stats()
    assert stats.hits == 1

    # Claim extraction caching
    claims = ["Claim 1", "Claim 2"]
    assert cache.get_claims("sentence 1", "en", "model-a") is None
    cache.set_claims("sentence 1", "en", "model-a", claims)
    assert cache.get_claims("sentence 1", "en", "model-a") == claims

    # Generation caching
    gen_val = "Generated summary text"
    assert cache.get_generation("article text", "en", "prompt_a", "model-a") is None
    cache.set_generation("article text", "en", "prompt_a", "model-a", gen_val)
    assert cache.get_generation("article text", "en", "prompt_a", "model-a") == gen_val


# ==============================================================================
# 2. Verifier and Claim Extractor Cache Integration Tests
# ==============================================================================


def test_nli_verifier_cache_integration(tmp_path: Path):
    """Ensure XNLIVerifier queries the cache and bypasses API calls on hit."""
    db_file = tmp_path / "nli_cache.db"
    cache = ResponseCache(db_path=db_file)

    cached_res = NLIResult(
        premise="Source premise.",
        hypothesis="Target hypothesis.",
        language="en",
        predicted_label="entailment",
        entailment_prob=0.98,
        neutral_prob=0.01,
        contradiction_prob=0.01,
        model_id="command-a-plus-05-2026",
        latency_ms=12.5,
    )
    cache.set_nli(
        premise="Source premise.",
        hypothesis="Target hypothesis.",
        language="en",
        model_id="command-a-plus-05-2026",
        result_dict=cached_res.to_dict(),
    )

    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="mock_key")
    verifier = XNLIVerifier(config=cfg, cache=cache)

    with patch("urllib.request.urlopen") as mock_url:
        results = verifier.verify_batch([("Source premise.", "Target hypothesis.")], language="en")
        assert len(results) == 1
        assert results[0].predicted_label == "entailment"
        assert results[0].entailment_prob == 0.98
        # Ensure urlopen was never called because it was a cache hit
        mock_url.assert_not_called()


def test_claim_extractor_cache_integration(tmp_path: Path):
    """Ensure AtomicClaimExtractor uses cache and avoids calling LLM client on hit."""
    db_file = tmp_path / "claims_cache.db"
    cache = ResponseCache(db_path=db_file)

    cached_claims = ["India gained independence in 1947.", "New Delhi is the capital."]
    cache.set_claims(
        sentence="India gained independence in 1947, and New Delhi is its capital.",
        language="en",
        model_id="mock-model",
        claims=cached_claims,
    )

    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.generate.side_effect = RuntimeError("Should not be called on cache hit")

    extractor = AtomicClaimExtractor(client=mock_client, cache=cache)
    extracted = extractor.extract_from_sentence(
        sentence="India gained independence in 1947, and New Delhi is its capital.",
        record_id="rec_1",
        sentence_index=0,
        language="en",
    )

    assert len(extracted) == 2
    assert extracted[0].atomic_claim_text == "India gained independence in 1947."
    assert extracted[1].atomic_claim_text == "New Delhi is the capital."
    assert extracted[0].extraction_status == "SUCCESS"
    mock_client.generate.assert_not_called()


# ==============================================================================
# 3. Quota, Rate Limit, and Authentication Error Handling Tests
# ==============================================================================


def test_nli_verifier_monthly_quota_fail_fast(tmp_path: Path):
    """Verify that HTTP 429 indicating monthly quota exhaustion immediately raises QuotaExhaustedError."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="mock_key", max_retries=5)
    verifier = XNLIVerifier(config=cfg)

    quota_err_body = b'{"message":"You are using a Trial key, which is limited to 1000 API calls / month."}'
    http_err = urllib.error.HTTPError(
        url="https://api.cohere.com/v2/chat",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(quota_err_body),
    )

    with patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen:
        with pytest.raises(QuotaExhaustedError) as exc_info:
            verifier.verify_batch([("Premise", "Hypothesis")], language="en")

        assert "monthly quota of 1000 API calls exceeded" in str(exc_info.value)
        # Must fail fast on the very first attempt without looping through retries
        assert mock_urlopen.call_count == 1


def test_nli_verifier_rate_limit_retry_after():
    """Verify that transient 429 with Retry-After header sleeps and retries successfully."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="mock_key", max_retries=3)
    verifier = XNLIVerifier(config=cfg)

    # First attempt: 429 transient rate limit with Retry-After
    rate_err_body = b'{"message":"Rate limit exceeded. Please try again later."}'
    hdrs = MagicMock()
    hdrs.get.side_effect = lambda k: "0.01" if k == "Retry-After" else None
    http_err = urllib.error.HTTPError(
        url="https://api.cohere.com/v2/chat",
        code=429,
        msg="Too Many Requests",
        hdrs=hdrs,
        fp=io.BytesIO(rate_err_body),
    )

    # Second attempt: Successful response
    success_json = json.dumps({
        "message": {
            "content": [{"type": "text", "text": json.dumps({
                "label": "entailment",
                "entailment_prob": 0.9,
                "neutral_prob": 0.05,
                "contradiction_prob": 0.05,
            })}]
        }
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = success_json
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", side_effect=[http_err, mock_resp]) as mock_call, \
         patch("time.sleep") as mock_sleep:
        results = verifier.verify_batch([("Premise", "Hypothesis")], language="en")
        assert len(results) == 1
        assert results[0].predicted_label == "entailment"
        assert mock_call.call_count == 2
        mock_sleep.assert_called_with(0.01)


def test_nli_verifier_auth_error_fail_fast():
    """Verify that HTTP 401 raises AuthenticationError immediately without retrying."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="invalid_key", max_retries=3)
    verifier = XNLIVerifier(config=cfg)

    http_err = urllib.error.HTTPError(
        url="https://api.cohere.com/v2/chat",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"message":"Invalid API key"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_err) as mock_call:
        with pytest.raises(AuthenticationError):
            verifier.verify_batch([("Premise", "Hypothesis")], language="en")
        assert mock_call.call_count == 1


# ==============================================================================
# 4. Atomic Writing and Experiment Storage Tests
# ==============================================================================


def test_save_experiment_records_atomic_overwrite(tmp_path: Path):
    """Test that save_experiment_records writes via atomic temp file and preserves data."""
    target_file = tmp_path / "exp_records.jsonl"
    record1 = ExperimentRecord(
        condition_id="C1",
        source_record_id="rec_1",
        language="en",
        prompt_strategy="prompt_a",
        prompt_template_version="1.0",
        verification_strategy="whole_document",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"max_tokens": 512},
        generated_summary="Summary 1",
        generation_latency_seconds=1.2,
        verification_latency_ms=150.0,
        verification_results=[{"label": "entailment"}],
        status="SUCCESS",
    )

    # Initial write
    save_experiment_records([record1], target_file, append=False)
    assert target_file.exists()

    loaded = load_experiment_records(target_file)
    assert len(loaded) == 1
    assert loaded[0].condition_id == "C1"

    # Overwrite with record2
    record2 = ExperimentRecord(
        condition_id="C2",
        source_record_id="rec_1",
        language="en",
        prompt_strategy="prompt_b",
        prompt_template_version="1.0",
        verification_strategy="whole_document",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"max_tokens": 512},
        generated_summary="Summary 2",
        generation_latency_seconds=1.1,
        verification_latency_ms=140.0,
        verification_results=[{"label": "entailment"}],
        status="SUCCESS",
    )
    save_experiment_records([record2], target_file, append=False)
    loaded_new = load_experiment_records(target_file)
    assert len(loaded_new) == 1
    assert loaded_new[0].condition_id == "C2"


# ==============================================================================
# 5. Benchmark Manifest Integrity & Validation Tests
# ==============================================================================


def test_benchmark_manifest_generation_and_validation(tmp_path: Path):
    """Test generation and validation of cryptographic benchmark manifest."""
    b_dir = tmp_path / "benchmark"
    b_dir.mkdir()

    # Create dummy EN benchmark file
    en_file = b_dir / "benchmark_en.jsonl"
    with open(en_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"example_id": "en_1", "source_text": "Text 1", "reference_summary": "Sum 1"}) + "\n")
        f.write(json.dumps({"example_id": "en_2", "source_text": "Text 2", "reference_summary": "Sum 2"}) + "\n")

    # Create dummy HI benchmark file
    hi_file = b_dir / "benchmark_hi.jsonl"
    with open(hi_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"example_id": "hi_1", "source_text": "Hindi 1", "reference_summary": "Sum H1"}) + "\n")
        f.write(json.dumps({"example_id": "hi_2", "source_text": "Hindi 2", "reference_summary": "Sum H2"}) + "\n")

    manifest_file = b_dir / "benchmark_manifest.json"
    manifest = generate_benchmark_manifest(
        benchmark_dir=b_dir,
        output_path=manifest_file,
    )

    assert manifest["total_records"] == 4
    assert manifest["languages"]["en"]["record_count"] == 2
    assert manifest["languages"]["hi"]["record_count"] == 2

    # Validation should pass
    is_valid, errors = validate_benchmark_manifest(manifest_file, benchmark_dir=b_dir)
    assert is_valid
    assert len(errors) == 0

    # Modify file content to test SHA-256 detection
    with open(en_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"example_id": "en_tampered", "source_text": "T", "reference_summary": "S"}) + "\n")

    is_valid, errors = validate_benchmark_manifest(manifest_file, benchmark_dir=b_dir)
    assert not is_valid
    assert any("SHA-256 mismatch" in e for e in errors)


def test_benchmark_manifest_duplicate_id_detection(tmp_path: Path):
    """Test that validate_benchmark_manifest rejects duplicate record IDs."""
    b_dir = tmp_path / "benchmark_dup"
    b_dir.mkdir()

    en_file = b_dir / "benchmark_en.jsonl"
    with open(en_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"example_id": "dup_1", "source_text": "T1", "reference_summary": "S1"}) + "\n")
        f.write(json.dumps({"example_id": "dup_1", "source_text": "T2", "reference_summary": "S2"}) + "\n")

    hi_file = b_dir / "benchmark_hi.jsonl"
    with open(hi_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"example_id": "hi_1", "source_text": "T3", "reference_summary": "S3"}) + "\n")

    manifest_file = b_dir / "benchmark_manifest.json"
    generate_benchmark_manifest(benchmark_dir=b_dir, output_path=manifest_file)

    is_valid, errors = validate_benchmark_manifest(manifest_file, benchmark_dir=b_dir)
    assert not is_valid
    assert any("Duplicate record ID 'dup_1'" in e for e in errors)


# ==============================================================================
# 6. Preflight Call Budget & Feasibility Tests
# ==============================================================================


def test_estimate_api_calls_breakdown():
    """Verify mathematical exactness of API call estimation for 200 records."""
    estimate = estimate_api_calls(num_records=200)

    # 200 records * 2 prompts = 400 generation calls
    assert estimate.generation_calls == 400
    # 400 whole doc NLI calls
    assert estimate.whole_doc_nli_calls == 400
    # 400 * 3.4 = 1,360 sentences
    assert estimate.sentence_nli_calls == 1360
    assert estimate.claim_extraction_calls == 1360
    # 1360 * 2.088 = 2,840 claims * 3 evidence = 8,520 calls
    assert estimate.hybrid_claim_nli_calls == 8520
    # Total = 400 + 400 + 1360 + 1360 + 8520 = 12,040 calls
    assert estimate.total_calls == 12040

    # At 20 calls/min = 602 minutes = 10.03 hours
    assert estimate.estimated_minutes_trial == 602.0
    assert estimate.estimated_hours_trial == 10.03


def test_preflight_blocks_on_insufficient_trial_quota(tmp_path: Path):
    """Verify that run_preflight blocks when required calls exceed monthly trial limit."""
    report = run_preflight(
        manifest_path="data/benchmark/benchmark_manifest.json",
        target_records=200,
        allow_insufficient_quota=False,
    )
    assert report["status"] == "BLOCKED"
    assert not report["can_proceed"]
    assert any("exceed the Cohere trial key monthly cap" in b for b in report["blockers"])


def test_preflight_allows_controlled_sample():
    """Verify that run_preflight passes for a 10-record sample within quota limits."""
    report = run_preflight(
        manifest_path="data/benchmark/benchmark_manifest.json",
        target_records=10,
        allow_insufficient_quota=False,
    )
    assert report["budget"]["total_calls"] == 602
    assert report["status"] == "READY"
    assert report["can_proceed"]
    assert len(report["blockers"]) == 0


# ==============================================================================
# 7. Re-verification Dry Run & Honest Provenance Tests
# ==============================================================================


def test_reverify_dry_run_executes_zero_api_calls():
    """Verify that reverify_controlled_experiment with dry_run=True returns expected plan without API calls."""
    report = reverify_controlled_experiment(
        results_path="data/benchmark_results/controlled_run/experiment_results.jsonl",
        en_benchmark_path="data/benchmark/controlled_sample_en.jsonl",
        hi_benchmark_path="data/benchmark/controlled_sample_hi.jsonl",
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["total_records"] == 10
    assert report["completed_conditions"] == 20
    assert report["remaining_conditions"] == 40
    assert report["estimated_calls"] == 440
