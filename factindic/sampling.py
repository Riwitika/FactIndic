"""Reproducible sampling with configurable sample sizes and random seeds."""

import random
from typing import List, Optional, TypeVar

T = TypeVar("T")


def sample_records(
    records: List[T],
    sample_size: Optional[int] = None,
    seed: Optional[int] = 42,
    shuffle: bool = True,
) -> List[T]:
    """Sample records deterministically using a fixed random seed.

    Args:
        records: List of input elements (e.g., FactIndicRecord objects).
        sample_size: Number of records to sample. If None or >= len(records), all
            records are returned (shuffled if shuffle=True).
        seed: Random seed for reproducible selection. If None, non-deterministic.
        shuffle: Whether to randomize the ordering of selected records.

    Returns:
        List of sampled records.

    Raises:
        ValueError: If sample_size is negative.
    """
    if sample_size is not None and sample_size < 0:
        raise ValueError(f"sample_size must be non-negative, got {sample_size}")

    if not records:
        return []

    # Use an isolated random instance to avoid polluting global state
    rng = random.Random(seed)

    records_copy = list(records)

    if shuffle:
        rng.shuffle(records_copy)

    if sample_size is None or sample_size >= len(records_copy):
        return records_copy

    return records_copy[:sample_size]
