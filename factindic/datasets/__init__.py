"""Dataset loaders for FactIndic."""

from factindic.datasets.base import BaseDatasetLoader
from factindic.datasets.xlsum import XLSumLoader, SUPPORTED_LANGUAGES
from factindic.datasets.fixture_loader import FixtureDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "XLSumLoader",
    "FixtureDatasetLoader",
    "SUPPORTED_LANGUAGES",
]
