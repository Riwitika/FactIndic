"""Offline fixture dataset loader for local testing and validation."""

from pathlib import Path
from typing import Iterator, List, Optional, Union

from factindic.datasets.base import BaseDatasetLoader
from factindic.schema import FactIndicRecord
from factindic.serialization import load_jsonl


class FixtureDatasetLoader(BaseDatasetLoader):
    """Loads records from a local fixture file (no network calls required)."""

    def __init__(self, fixture_path: Union[str, Path]):
        self.fixture_path = Path(fixture_path)
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {self.fixture_path}")

    def load(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> List[FactIndicRecord]:
        """Load records from the local fixture."""
        records = load_jsonl(self.fixture_path)
        if max_samples is not None:
            return records[:max_samples]
        return records

    def stream(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Iterator[FactIndicRecord]:
        """Stream records one-by-one from the fixture."""
        count = 0
        for rec in self.load(split=split):
            yield rec
            count += 1
            if max_samples is not None and count >= max_samples:
                break
