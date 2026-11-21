"""Summary generation execution, provenance tracking, and persistence."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from factindic.models.base import BaseLLMClient
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy, build_summarization_prompt
from factindic.schema import FactIndicRecord


@dataclass
class GenerationRecord:
    """Standardized provenance record for generated summaries."""

    source_dataset: str
    source_config: str
    source_split: str
    source_record_id: str
    language: str
    source_text: str
    reference_summary: str
    prompt_strategy: str
    prompt_template_version: str
    model_id: str
    provider: str
    generation_parameters: Dict[str, Any]
    generated_summary: str
    generation_timestamp_utc: str
    latency_seconds: float = 0.0
    status: str = "SUCCESS"  # SUCCESS or FAILED
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRecord":
        """Construct a GenerationRecord from dictionary."""
        return cls(
            source_dataset=str(data.get("source_dataset", "")),
            source_config=str(data.get("source_config", "")),
            source_split=str(data.get("source_split", "")),
            source_record_id=str(data.get("source_record_id", "")),
            language=str(data.get("language", "")),
            source_text=str(data.get("source_text", "")),
            reference_summary=str(data.get("reference_summary", "")),
            prompt_strategy=str(data.get("prompt_strategy", "")),
            prompt_template_version=str(data.get("prompt_template_version", "")),
            model_id=str(data.get("model_id", "")),
            provider=str(data.get("provider", "")),
            generation_parameters=dict(data.get("generation_parameters", {})),
            generated_summary=str(data.get("generated_summary", "")),
            generation_timestamp_utc=str(data.get("generation_timestamp_utc", "")),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
        )


def run_article_generation(
    record: FactIndicRecord,
    client: BaseLLMClient,
    strategy: PromptStrategy,
    temperature: float = 0.3,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
) -> GenerationRecord:
    """Execute summary generation for a single FactIndic record under a prompt strategy.

    Args:
        record: The source FactIndicRecord.
        client: Active BaseLLMClient instance.
        strategy: PromptStrategy (Prompt A or Prompt B).
        temperature: Sampling temperature (held identical across conditions).
        max_new_tokens: Maximum tokens in generated summary.
        top_p: Nucleus sampling parameter.

    Returns:
        Populated GenerationRecord with complete provenance.
    """
    prompt = build_summarization_prompt(
        source_text=record.source_text,
        strategy=strategy,
        language=record.language,
    )

    strat_val = strategy.value if isinstance(strategy, PromptStrategy) else str(strategy)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    raw_id = record.metadata.get("raw_id", record.example_id)
    source_ds = record.metadata.get("source_dataset", "csebuetnlp/xlsum")
    source_cfg = record.metadata.get("hf_config", record.language)
    source_spl = record.metadata.get("split", "test")

    try:
        gen_result = client.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

        return GenerationRecord(
            source_dataset=source_ds,
            source_config=source_cfg,
            source_split=source_spl,
            source_record_id=raw_id,
            language=record.language,
            source_text=record.source_text,
            reference_summary=record.reference_summary,
            prompt_strategy=strat_val,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            model_id=gen_result.model_id,
            provider=gen_result.provider,
            generation_parameters=gen_result.generation_parameters,
            generated_summary=gen_result.text,
            generation_timestamp_utc=timestamp_utc,
            latency_seconds=gen_result.latency_seconds,
            status="SUCCESS",
            error_message=None,
        )

    except Exception as exc:
        return GenerationRecord(
            source_dataset=source_ds,
            source_config=source_cfg,
            source_split=source_spl,
            source_record_id=raw_id,
            language=record.language,
            source_text=record.source_text,
            reference_summary=record.reference_summary,
            prompt_strategy=strat_val,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            model_id=getattr(client, "model_id", "unknown"),
            provider=getattr(client, "provider", "unknown"),
            generation_parameters={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
            },
            generated_summary="",
            generation_timestamp_utc=timestamp_utc,
            latency_seconds=0.0,
            status="FAILED",
            error_message=str(exc),
        )


def save_generation_records(
    records: List[GenerationRecord],
    file_path: Union[str, Path],
    append: bool = True,
) -> int:
    """Save generated summary records to a UTF-8 JSONL file.

    Args:
        records: List of GenerationRecord objects.
        file_path: Output file path.
        append: If True, appends to existing file; if False, overwrites.

    Returns:
        Number of records written.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    return count


def load_generation_records(file_path: Union[str, Path]) -> List[GenerationRecord]:
    """Load generation records from a UTF-8 JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[GenerationRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(GenerationRecord.from_dict(json.loads(line)))

    return records
