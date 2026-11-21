"""Unit tests for reproducible sampling."""

import pytest
from factindic.sampling import sample_records


def test_reproducible_sampling_fixed_seed():
    items = [f"item_{i}" for i in range(50)]

    sample_a = sample_records(items, sample_size=10, seed=42)
    sample_b = sample_records(items, sample_size=10, seed=42)

    assert sample_a == sample_b
    assert len(sample_a) == 10


def test_different_seeds_produce_different_orders():
    items = [f"item_{i}" for i in range(100)]

    sample_1 = sample_records(items, sample_size=10, seed=42)
    sample_2 = sample_records(items, sample_size=10, seed=999)

    assert sample_1 != sample_2


def test_sampling_exceeding_length():
    items = ["a", "b", "c"]
    result = sample_records(items, sample_size=10, seed=42)
    assert len(result) == 3
    assert set(result) == {"a", "b", "c"}


def test_sampling_zero_and_empty():
    items = ["a", "b", "c"]
    assert sample_records(items, sample_size=0, seed=42) == []
    assert sample_records([], sample_size=5, seed=42) == []


def test_negative_sample_size_raises_error():
    with pytest.raises(ValueError, match="non-negative"):
        sample_records(["a", "b"], sample_size=-1)
