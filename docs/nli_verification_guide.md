# Multilingual NLI Verification Guide

This guide documents the Natural Language Inference (NLI) verification architecture used in FactIndic for assessing the factual consistency of generated summaries against source articles in English and Hindi.

---

## 1. Model Overview

- **Model Identifier**: [`joeddav/xlm-roberta-large-xnli`](https://huggingface.co/joeddav/xlm-roberta-large-xnli)
- **Official Model Card**: [https://huggingface.co/joeddav/xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli)
- **Base Architecture**: XLM-RoBERTa Large (`xlm-roberta-large`)
- **Parameter Count**: ~560 Million parameters (24 layers, 1024 hidden dimension, 16 attention heads)
- **Pretraining**: Self-supervised masked language modeling on CC-100 (2.5 TB filtered text across 100 languages)
- **Fine-Tuning**: Supervised sequence classification on the multilingual Cross-lingual Natural Language Inference (XNLI) benchmark and MNLI
- **Licence**: **MIT License** (permissive for research and commercial reuse with attribution)

---

## 2. Supported Languages

XLM-RoBERTa-large-XNLI natively supports 15 languages evaluated on the XNLI benchmark. FactIndic focuses on:
- **English (`en`)**: High-resource baseline
- **Hindi (`hi`)**: Target Indic language using native Devanagari script

Both languages share the unified multilingual sentencepiece vocabulary (250,000 subwords), ensuring direct cross-lingual representations without translation artifacts.

---

## 3. Premise-Hypothesis Formulation

In FactIndic's verification pipeline:
- **Premise ($P$)**: The reference source article (or relevant passage).
- **Hypothesis ($H$)**: The generated summary sentence or claim being verified.

$$\text{Input Sequence} = \text{<s> Premise </s></s> Hypothesis </s>}$$

The model produces 3 unnormalized logits $[z_{\text{contra}}, z_{\text{neut}}, z_{\text{entail}}]$, which are converted into a probability distribution via softmax:

$$P(\text{label}_i) = \frac{e^{z_i}}{\sum_{j \in \{\text{contra}, \text{neut}, \text{entail}\}} e^{z_j}}$$

---

## 4. Label Interpretation in Summarization Verification

| Label | Definition in NLI | Interpretation in Factuality Assessment |
|---|---|---|
| **Entailment** | Premise entails hypothesis | **Supported**: The claim is directly substantiated by the source article. |
| **Neutral** | Premise does not confirm or deny | **Extrinsic Hallucination / Unsupported**: The claim introduces unmentioned entities, events, or speculation. |
| **Contradiction** | Premise contradicts hypothesis | **Intrinsic Hallucination / Fact Distortion**: The claim directly conflicts with numbers, dates, relations, or entities in the source. |

---

## 5. Why NLI is a Proxy for Factuality (Not Ground Truth)

> [!WARNING]
> **Important Epistemic Distinction:**
> Entailment probability must **never** be treated as a definitive, ground-truth measure of factual truth.

NLI evaluates **factual consistency with the provided premise**, not absolute real-world accuracy:
1. **Source Fallibility**: If the source article contains factual errors, an entailed summary claim is still factually incorrect in the real world.
2. **Length & Context Limitations**: Standard transformer attention is bounded to 512 tokens. While controlled truncation (`only_first`) preserves the hypothesis claim, evidence located deep in long source articles may be truncated.
3. **Lexical Overlap Bias**: Neural NLI models can exhibit heuristic biases (e.g., favoring high lexical overlap or struggling with complex logical negations).
4. **Granularity**: Document-level summaries often contain multiple sub-claims; whole-summary NLI can conflate partially supported sentences with unsupported ones.

---

## 6. Hardware Considerations for 2017 MacBook Air

Running a 560M parameter model on an older machine (e.g., 2017 MacBook Air with dual-core Intel Core i5/i7 and 8 GB RAM):

- **Memory Footprint**: ~2.24 GB in FP32 weights; total process footprint during inference is ~4.0–4.5 GB RAM.
- **CPU Inference**: Fully supported; no GPU or MPS requirement.
- **Recommended Batch Size**: Set `batch_size = 1` to `4` (default in `NLIConfig` is 4).
- **Lazy Loading**: `XNLIVerifier` keeps models uninstantiated until explicit invocation, avoiding memory bloat at application startup.
- **Memory Release**: Calling `verifier.unload_model()` dereferences weights and triggers garbage collection to free system RAM.

---

## 7. Installation and Usage

### Prerequisites
Install the required verification packages:
```bash
pip install transformers torch sentencepiece
```

### Python API Example
```python
from factindic.verification import XNLIVerifier, NLIConfig

# Configure CPU verifier with controlled premise truncation
config = NLIConfig(
    model_id="joeddav/xlm-roberta-large-xnli",
    batch_size=2,
    max_length=512,
    device="cpu",
)
verifier = XNLIVerifier(config=config)

# English Premise-Hypothesis Verification
result_en = verifier.verify_premise_hypothesis(
    premise="The Reserve Bank of India raised interest rates by 25 basis points on Wednesday.",
    hypothesis="India's central bank increased interest rates.",
    language="en",
)
print(f"Predicted: {result_en.predicted_label}")
print(f"Entailment Prob: {result_en.entailment_prob:.4f}")

# Hindi Premise-Hypothesis Verification
result_hi = verifier.verify_premise_hypothesis(
    premise="भारतीय रिजर्व बैंक ने बुधवार को ब्याज दरों में 25 आधार अंकों की वृद्धि की।",
    hypothesis="केंद्रीय बैंक ने ब्याज दरों में बढ़ोतरी की है।",
    language="hi",
)
print(f"Predicted: {result_hi.predicted_label}")
print(f"Entailment Prob: {result_hi.entailment_prob:.4f}")

# Clean up memory when done
verifier.unload_model()
```

---

## 8. Whole-Document NLI Verification Strategy

FactIndic provides the `WholeDocumentVerifier` module (`factindic.verification.document_verifier`) to evaluate full generated summaries against complete source articles.

### Formulation
- **Premise**: Entire source article (`GenerationRecord.source_text`).
- **Hypothesis**: Entire generated summary (`GenerationRecord.generated_summary`).
- **Strategy Identifier**: `"whole_document"`.

### Python Usage
```python
from factindic.generation import load_generation_records
from factindic.verification import (
    WholeDocumentVerifier,
    save_document_verification_results,
)

# Load existing generated summaries (Prompt A / Prompt B)
records = load_generation_records("data/generations/controlled_generation_results.jsonl")

# Run verification (reuses single in-memory model across all records)
doc_verifier = WholeDocumentVerifier()
verification_results = doc_verifier.verify_records(records, batch_size=2)

# Save verification outputs separately, preserving original generation records
save_document_verification_results(
    verification_results,
    "data/generations/whole_document_verification_results.jsonl",
)
```

### Interpretation and Limitations
1. **Macro-Level Agreement**: Whole-document verification measures broad contextual entailment across the text pair. It assesses whether the central themes of the summary are substantiated by the article.
2. **Sub-Claim Masking**: A high document-level entailment score does **not** prove that every single claim or sentence in the summary is accurate. Minor hallucinations, unsupported numerical figures, or fine-grained entity errors can be overshadowed when scored as an aggregated hypothesis.
3. **Premise Length Asymmetry**: Full BBC news articles often exceed 512 tokens. While `truncation_strategy="only_first"` ensures the entire generated summary is retained, facts located deep in lengthy source articles may fall outside the model's active attention window.
4. **No Arbitrary Factuality Scores**: FactIndic reports raw, normalized class probabilities (`entailment`, `neutral`, `contradiction`) without imposing heuristic weighted composite scores that disguise model uncertainty.

---

## 9. Sentence-Level NLI Verification Strategy

FactIndic provides the `SentenceLevelVerifier` module (`factindic.verification.sentence_verifier`) to evaluate generated summaries at sentence granularity against the full source article.

### Formulation
- **Premise**: Entire source article (`GenerationRecord.source_text`).
- **Hypothesis**: Individual summary sentence extracted via multilingual segmentation (`sentence_text`).
- **Strategy Identifier**: `"sentence_level"`.

### Multilingual Sentence Segmentation
Summary texts are segmented into discrete sentences while respecting language-specific terminal punctuation:
- **English**: Terminal punctuation (`.`, `!`, `?`) followed by whitespace.
- **Hindi**: Native Devanagari purna viram (`।` U+0964), deergh viram (`॥` U+0965), and standard terminators (`?`, `!`, `.`) followed by whitespace.

### Python Usage
```python
from factindic.generation import load_generation_records
from factindic.verification import (
    SentenceLevelVerifier,
    save_sentence_verification_results,
)

# Load existing generated summaries (Prompt A / Prompt B)
records = load_generation_records("data/generations/controlled_generation_results.jsonl")

# Run sentence-level verification (preserves sentence order across all records)
sent_verifier = SentenceLevelVerifier()
sentence_results = sent_verifier.verify_records(records, batch_size=4)

# Save sentence verification outputs separately, preserving original generation records
save_sentence_verification_results(
    sentence_results,
    "data/generations/sentence_level_verification_results.jsonl",
)
```

### Interpretation and Sub-Claim Limitations
1. **Diagnostic Granularity**: Evaluating each sentence individually prevents whole-summary masking, isolating exactly which sentences in a multi-sentence summary are ungrounded or contradicted.
2. **Compound Sentence Sub-Claims**: A high entailment probability for a single sentence does **not** guarantee that every atomic sub-claim within that sentence is true. Compound sentences containing conjunctions, subordinate clauses, or multiple predicates can combine supported claims with minor unsupported facts that the model fails to penalize.
3. **No Synthetic Factuality Scores**: Probabilities are reported independently per sentence without inventing ad-hoc composite scores or claiming absolute factual ground truth.

---

## 10. Atomic Claim Extraction Strategy

FactIndic provides the `AtomicClaimExtractor` module (`factindic.verification.claim_extractor`) to decompose generated summary sentences into standalone, indivisible factual claims before evidence retrieval or verification.

### Motivation and Granularity
While sentence-level verification isolates ungrounded sentences, sentences in abstractive summaries are frequently compound or complex (e.g. *"The agency launched the mission on Friday and confirmed it reached lunar orbit on Sunday"* contains two distinct factual claims). Claim extraction breaks compound sentences into independent atomic claims:
1. *"The agency launched the mission on Friday."*
2. *"The agency confirmed the mission reached lunar orbit on Sunday."*

### Structured Output Schema (`ExtractedClaim`)
- **`record_id`**: Originating generation record identifier.
- **`language`**: Target language (`en` or `hi`).
- **`original_sentence_index`**: Position of the sentence in the summary.
- **`original_sentence_text`**: Full text of the source summary sentence.
- **`claim_id`**: Traceable identifier formatted as `{record_id}_s{sentence_index}_c{claim_index}`.
- **`atomic_claim_text`**: Standalone, self-contained factual claim.
- **`extraction_status`**: `"SUCCESS"` or `"FAILED"`.

### Multilingual Support Without Translation
Atomic claim extraction operates natively in both languages:
- **English**: English prompt extracting English atomic claims.
- **Hindi**: Native Devanagari prompt extracting Hindi atomic claims.
No cross-lingual translation is performed, preserving the original lexical fidelity of the generation.

### Python Usage
```python
from factindic.generation import load_generation_records
from factindic.models.aya import CohereClient
from factindic.verification import (
    AtomicClaimExtractor,
    save_extracted_claims,
)

# Load existing generated summaries
records = load_generation_records("data/generations/controlled_generation_results.jsonl")

# Configure extractor reusing existing Cohere Command A+ client
client = CohereClient()
extractor = AtomicClaimExtractor(client=client, temperature=0.0)

# Extract claims across all records while preserving source GenerationRecord instances
extracted_claims = extractor.extract_from_records(records)

# Save extracted claims separately to JSONL
save_extracted_claims(
    extracted_claims,
    "data/generations/extracted_claims.jsonl",
)
```

### Limitations & Quality Safeguards
1. **No Evidence Retrieval**: Claim extraction isolates claims; it does not verify them or retrieve evidence.
2. **Context Resolution Bounds**: Pronouns and definite references are resolved within the local sentence context. Long-distance discourse co-reference across paragraphs is intentionally not inferred to prevent hallucinating external assumptions.
3. **No Hallucinated Claims**: Prompts strictly prohibit adding external world knowledge or modifying the original factual meaning.
