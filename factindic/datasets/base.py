"""Abstract base class for FactIndic dataset loaders."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from factindic.schema import FactIndicRecord


class BaseDatasetLoader(ABC):
    """Abstract interface for dataset loaders."""

    @abstractmethod
    def load(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[FactIndicRecord]:
        """Load dataset split as a list of FactIndicRecord instances.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            max_samples: Maximum number of records to load. If None, load all.

        Returns:
            List of FactIndicRecord objects.
        """
        pass

    @abstractmethod
    def stream(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Iterator[FactIndicRecord]:
        """Stream dataset records as an iterator to avoid loading entire corpus into RAM.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            max_samples: Maximum number of records to yield.

        Yields:
            FactIndicRecord instances.
        """
        pass
