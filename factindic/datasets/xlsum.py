"""XL-Sum dataset loader for English and Hindi.

Hugging Face dataset repository: csebuetnlp/xlsum
Paper: Hasan et al., Findings of ACL 2021 (XL-Sum: Large-Scale Multilingual Abstractive Summarization).
License: CC BY-NC-SA 4.0 (BBC articles).

Strict No-Fallback Policy:
- Connects exclusively to the official Hugging Face repository for csebuetnlp/xlsum.
- Never falls back to fixtures or synthetic data if live network fetching fails.
"""

import json
import tarfile
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from factindic.datasets.base import BaseDatasetLoader
from factindic.schema import FactIndicRecord

SUPPORTED_LANGUAGES = {
    "en": "english",
    "english": "english",
    "hi": "hindi",
    "hindi": "hindi",
}

LANGUAGE_CANONICAL_ALIASES = {
    "english": "en",
    "hindi": "hi",
}

HF_REPO_URL_TEMPLATE = (
    "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/{lang}_XLSum_v2.0.tar.bz2"
)

HF_DATASET_PARQUET_BRANCH = "refs/convert/parquet"
HF_DATASET_PARQUET_COMMIT_SHA = "ab04842511fc357feae53281fd104aa3c36dff07"

HF_PARQUET_URL_TEMPLATE = (
    "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/{revision}/{lang}/{split}/0000.parquet"
)

HF_DATASET_REVISION_PARQUET = HF_DATASET_PARQUET_BRANCH


class XLSumLoader(BaseDatasetLoader):
    """Dataset loader for the BBC XL-Sum multilingual benchmark from Hugging Face."""

    HF_DATASET_ID = "csebuetnlp/xlsum"

    def __init__(
        self,
        language: str = "english",
        timeout: int = 300,
        data_format: str = "parquet",
    ):
        lang_lower = language.lower().strip()
        if lang_lower not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(set(SUPPORTED_LANGUAGES.keys())))
            raise ValueError(
                f"Unsupported language '{language}' for XL-Sum. Supported: {supported}"
            )
        self.hf_config = SUPPORTED_LANGUAGES[lang_lower]
        self.lang_alias = LANGUAGE_CANONICAL_ALIASES[self.hf_config]
        self.timeout = timeout
        if data_format not in ("parquet", "archive"):
            raise ValueError(f"Unsupported data_format '{data_format}'. Supported: 'parquet', 'archive'")
        self.data_format = data_format

    def _convert_hf_item(
        self,
        item: dict,
        split: str,
        source_format: str = "archive",
        revision: Optional[str] = None,
    ) -> FactIndicRecord:
        """Convert a raw Hugging Face XL-Sum record into the standardized FactIndicRecord schema."""
        raw_id = str(item.get("id", ""))
        example_id = f"xlsum_{self.lang_alias}_{raw_id}"

        metadata = {
            "raw_id": raw_id,
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "split": split,
            "source_dataset": self.HF_DATASET_ID,
            "hf_config": self.hf_config,
            "source_format": source_format,
        }
        if revision:
            metadata["revision"] = revision

        return FactIndicRecord(
            example_id=example_id,
            language=self.lang_alias,
            source_text=str(item.get("text", "")),
            reference_summary=str(item.get("summary", "")),
            metadata=metadata,
        )

    def stream_parquet(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        batch_size: int = 100,
    ) -> Iterator[FactIndicRecord]:
        """Stream FactIndicRecords from the remote Hugging Face Parquet conversion branch.

        Uses fsspec and pyarrow to read row groups/batches without downloading
        large compressed tar.bz2 archives.

        Yields:
            FactIndicRecord instances.
        """
        import fsspec
        import pyarrow.parquet as pq

        url = HF_PARQUET_URL_TEMPLATE.format(
            revision=HF_DATASET_PARQUET_COMMIT_SHA,
            lang=self.hf_config,
            split=split,
        )
        headers = {"User-Agent": "FactIndic-Research-Pipeline/1.0 (NLP Research)"}

        try:
            fs, path = fsspec.core.url_to_fs(url, headers=headers)
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                count = 0
                for batch in pf.iter_batches(batch_size=batch_size):
                    batch_dict = batch.to_pydict()
                    num_rows = len(batch_dict["id"])
                    for i in range(num_rows):
                        raw_item = {
                            "id": batch_dict["id"][i],
                            "url": batch_dict["url"][i],
                            "title": batch_dict["title"][i],
                            "summary": batch_dict["summary"][i],
                            "text": batch_dict["text"][i],
                        }
                        record = self._convert_hf_item(
                            raw_item,
                            split=split,
                            source_format="parquet",
                            revision=HF_DATASET_PARQUET_COMMIT_SHA,
                        )
                        record.metadata["branch"] = HF_DATASET_PARQUET_BRANCH
                        record.metadata["commit_sha"] = HF_DATASET_PARQUET_COMMIT_SHA
                        yield record
                        count += 1
                        if max_samples is not None and count >= max_samples:
                            return
        except Exception as exc:
            raise ConnectionError(
                f"Failed to stream Parquet dataset '{self.HF_DATASET_ID}' ({self.hf_config}, split='{split}') at {url}: {exc}"
            ) from exc

    def stream_archive(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[Dict[str, Any], FactIndicRecord]]:
        """Stream raw dictionaries and mapped FactIndicRecords from the live HF tar.bz2 archive.

        Streams directly over HTTP without downloading or storing the full archive.

        Yields:
            Tuple of (raw_dict, mapped_FactIndicRecord)
        """
        url = HF_REPO_URL_TEMPLATE.format(lang=self.hf_config)
        target_file_pattern = f"{self.hf_config}_{split}.jsonl"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "FactIndic-Research-Pipeline/1.0 (NLP Research)",
                "Accept": "*/*",
            },
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to Hugging Face dataset '{self.HF_DATASET_ID}' ({self.hf_config}) at {url}: {exc}"
            ) from exc

        count = 0
        try:
            with tarfile.open(mode="r|bz2", fileobj=resp) as tar:
                for member in tar:
                    if target_file_pattern in member.name:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        for line_bytes in f:
                            line = line_bytes.decode("utf-8").strip()
                            if not line:
                                continue
                            raw_item = json.loads(line)
                            record = self._convert_hf_item(
                                raw_item,
                                split=split,
                                source_format="archive",
                            )
                            yield raw_item, record
                            count += 1
                            if max_samples is not None and count >= max_samples:
                                return
                        break
        except Exception as exc:
            if count >= (max_samples or 1):
                return
            raise RuntimeError(
                f"Error streaming split '{split}' for XL-Sum ({self.hf_config}): {exc}"
            ) from exc

    def stream_raw(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Iterator[Tuple[Dict[str, Any], FactIndicRecord]]:
        """Stream raw records (delegates to stream_archive for backwards compatibility)."""
        return self.stream_archive(split=split, max_samples=max_samples)

    def stream(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Iterator[FactIndicRecord]:
        """Stream FactIndicRecords from the live HF repository using the configured data_format."""
        if self.data_format == "parquet":
            yield from self.stream_parquet(split=split, max_samples=max_samples)
        else:
            for _, record in self.stream_archive(split=split, max_samples=max_samples):
                yield record

    def load(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> List[FactIndicRecord]:
        """Load records up to max_samples from the live dataset."""
        return list(self.stream(split=split, max_samples=max_samples))
