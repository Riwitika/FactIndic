<div align="center">

# 🇮🇳 FactIndic

**Multilingual Factual Consistency & Hallucination Verification for Indic Summarization**

An NLP empirical research framework and verification engine measuring factual consistency and hallucinations in abstractive text summarization across English and Hindi using natural language inference (NLI) and large language model evaluations.

---

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![mDeBERTa-v3](https://img.shields.io/badge/NLI%20Engine-mDeBERTa--v3-00599C?style=flat-square)](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
[![Cohere](https://img.shields.io/badge/LLM-Command%20A%2B-39594C?style=flat-square)](https://cohere.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Lab-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

<br />

## 🌟 Overview

Abstractive text summarization models frequently hallucinate facts or distort source relationships. While factual verification has been thoroughly benchmarked in high-resource English datasets, Indic languages (such as Hindi) remain vulnerable to semantic drift, gender/morphological inaccuracies, and cross-lingual translation artifacts.

**FactIndic** establishes an empirical, reproducible $2 \times 3$ factorial evaluation protocol across English and Hindi:
- **Controlled Generation:** Strict model parity using Cohere Command A+ with identical decoding hyperparameters.
- **Multilingual NLI Verification:** Zero-shot and fine-tuned cross-lingual natural language inference powered by deterministic `mDeBERTa-v3`.
- **Claim-Level Granularity:** Deconstructs generated summaries into discrete atomic propositions and scores entailment/contradiction against source documents.
- **Epistemic Integrity:** Preserves raw class probabilities ($P(\text{Entailment})$, $P(\text{Neutral})$, $P(\text{Contradiction})$) without synthetic composite weighting.

---

## 🏗️ Verification Pipeline

```mermaid
flowchart TD
    subgraph Corpus["📚 Corpus Ingestion"]
        Src[Source Document<br/>English / Hindi]
        Sum[Abstractive Summary<br/>Model Generated]
    end

    subgraph Extraction["🔍 Proposition Decomposition"]
        Sum --> Splitter[Atomic Claim Extractor]
        Splitter --> C1[Claim 1]
        Splitter --> C2[Claim 2]
        Splitter --> Cn[Claim n]
    end

    subgraph Inference["⚖️ NLI Verification Engine"]
        Src & C1 & C2 & Cn --> NLI[Multilingual mDeBERTa-v3<br/>XNLI-calibrated]
        NLI --> Probs["Entailment / Contradiction Distribution"]
    end

    subgraph Metrics["📊 Factuality Ledger"]
        Probs --> FactualReport[Factual Consistency Matrix]
        Probs --> HallucinationAudit[Hallucination Error Taxonomy]
    end

    style Corpus fill:#0f172a,stroke:#38bdf8,color:#f8fafc
    style Extraction fill:#1e1b4b,stroke:#a855f7,color:#f8fafc
    style Inference fill:#1c1917,stroke:#10b981,color:#f8fafc
    style Metrics fill:#331400,stroke:#f97316,color:#f8fafc
```

---

## 🔬 Experimental Factorial Setup

- **Source Invariance:** Identical matched source documents evaluated across both English and Hindi.
- **Generation Model:** Cohere Command A+ (`command-a-plus-05-2026`)
  - Decoding: `temperature=0.3`, `top_p=0.9`, `max_new_tokens=256`.
- **Verification Engine:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Commit `b5113eb38ab`).
- **Benchmark Datasets:** XL-Sum English & Hindi (`csebuetnlp/xlsum`).

---

## 💻 Tech Stack

- **Frameworks:** Python 3.11, PyTorch, Transformers, Datasets.
- **LLM APIs:** Cohere Python SDK.
- **Interactive UI:** Streamlit, Pandas, Rich.
- **Testing & Quality:** Pytest, Flake8, Black.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Riwitika/FactIndic.git
cd FactIndic

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Set COHERE_API_KEY in .env
```

### 3. Run Interactive Evaluation Lab

```bash
streamlit run app.py
```

### 4. Execute Batch Evaluation

```bash
python scripts/evaluate.py --dataset xlsum --lang hi --samples 50
```

---

<div align="center">
<sub>Designed & Maintained by <a href="https://github.com/Riwitika">Riwitika</a></sub>
</div>
