"""Unit tests for XL-Sum and fixture loaders."""

import pytest
from pathlib import Path
from factindic.datasets.xlsum import XLSumLoader
from factindic.datasets.fixture_loader import FixtureDatasetLoader

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def test_xlsum_loader_language_validation():
    loader_en = XLSumLoader(language="english")
    assert loader_en.hf_config == "english"
    assert loader_en.lang_alias == "en"

    loader_en_alias = XLSumLoader(language="en")
    assert loader_en_alias.hf_config == "english"

    loader_hi = XLSumLoader(language="hindi")
    assert loader_hi.hf_config == "hindi"
    assert loader_hi.lang_alias == "hi"

    with pytest.raises(ValueError, match="Unsupported language"):
        XLSumLoader(language="spanish")


def test_xlsum_loader_convert_hf_item():
    loader = XLSumLoader(language="en")
    raw_hf_item = {
        "id": "12345",
        "url": "https://www.bbc.com/news/12345",
        "title": "Article Title",
        "summary": "Article Summary Text",
        "text": "Full article source text content.",
    }
    record = loader._convert_hf_item(raw_hf_item, split="test")

    assert record.example_id == "xlsum_en_12345"
    assert record.language == "en"
    assert record.source_text == raw_hf_item["text"]
    assert record.reference_summary == raw_hf_item["summary"]
    assert record.metadata["title"] == "Article Title"
    assert record.metadata["split"] == "test"
    assert record.metadata["source_dataset"] == "csebuetnlp/xlsum"


def test_fixture_loader_english():
    en_path = FIXTURES_DIR / "sample_english.jsonl"
    loader = FixtureDatasetLoader(en_path)

    all_records = loader.load()
    assert len(all_records) == 5
    assert all(r.language == "en" for r in all_records)

    limited = loader.load(max_samples=2)
    assert len(limited) == 2


def test_fixture_loader_hindi():
    hi_path = FIXTURES_DIR / "sample_hindi.jsonl"
    loader = FixtureDatasetLoader(hi_path)

    all_records = loader.load()
    assert len(all_records) == 5
    assert all(r.language == "hi" for r in all_records)

    # Verify Devanagari script presence
    assert any("इसरो" in r.source_text for r in all_records)


def test_fixture_loader_missing_file(tmp_path: Path):
    non_existent = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        FixtureDatasetLoader(non_existent)


def test_xlsum_loader_parquet_format_validation():
    loader = XLSumLoader(language="english", data_format="parquet")
    assert loader.data_format == "parquet"

    loader_archive = XLSumLoader(language="hindi", data_format="archive")
    assert loader_archive.data_format == "archive"

    with pytest.raises(ValueError, match="Unsupported data_format"):
        XLSumLoader(language="english", data_format="invalid_fmt")


def test_xlsum_loader_parquet_streaming_mocked():
    """Verify Parquet streaming yields valid FactIndicRecord objects with complete provenance."""
    from unittest.mock import MagicMock, patch

    mock_batch = MagicMock()
    mock_batch.to_pydict.return_value = {
        "id": ["1001", "1002"],
        "url": ["https://bbc.com/news/1001", "https://bbc.com/news/1002"],
        "title": ["Parquet Title 1", "Parquet Title 2"],
        "summary": ["Summary for item 1.", "Summary for item 2."],
        "text": [
            "Source article text for item 1 with sufficient content length.",
            "Source article text for item 2 with sufficient content length.",
        ],
    }

    mock_pf = MagicMock()
    mock_pf.iter_batches.return_value = [mock_batch]

    with patch("fsspec.core.url_to_fs") as mock_url_to_fs, \
         patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
        mock_fs = MagicMock()
        mock_fs.open.return_value.__enter__.return_value = MagicMock()
        mock_url_to_fs.return_value = (mock_fs, "dummy/path")

        loader = XLSumLoader(language="english", data_format="parquet")
        records = loader.load(split="test", max_samples=2)

        assert len(records) == 2
        assert records[0].example_id == "xlsum_en_1001"
        assert records[0].language == "en"
        assert records[0].metadata["source_format"] == "parquet"
        assert records[0].metadata["branch"] == "refs/convert/parquet"
        assert records[0].metadata["commit_sha"] == "ab04842511fc357feae53281fd104aa3c36dff07"
        assert records[0].metadata["revision"] == "ab04842511fc357feae53281fd104aa3c36dff07"
        assert records[0].metadata["source_dataset"] == "csebuetnlp/xlsum"
        assert records[1].example_id == "xlsum_en_1002"

