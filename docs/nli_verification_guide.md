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
