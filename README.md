# FactIndic: Factuality-Aware Summarization for Indic and Code-Mixed Languages

FactIndic is an NLP research framework designed to investigate and improve factual consistency in abstractive text summarization across English, Hindi, and multilingual contexts.

This repository implements **Phase 1: Dataset Preparation, Validation, and Feasibility Analysis**.

---

## 1. Research Scope & Experimental Design

FactIndic evaluates a $2 \times 3$ factorial matrix across English and Hindi using a unified, reproducible pipeline:

| Condition | Prompting Strategy | Verification Strategy | Granularity & Formulation |
|:---:|:---|:---|:---|
| **C1** | **Prompt A (Standard)** | Whole-Document NLI | Premise: Source Article, Hypothesis: Entire Summary |
| **C2** | **Prompt B (Factuality-Aware)** | Whole-Document NLI | Premise: Source Article, Hypothesis: Entire Summary |
| **C3** | **Prompt A (Standard)** | Sentence-Level NLI | Premise: Source Article, Hypothesis: Summary Sentence |
| **C4** | **Prompt B (Factuality-Aware)** | Sentence-Level NLI | Premise: Source Article, Hypothesis: Summary Sentence |
| **C5** | **Prompt A (Standard)** | Hybrid Claim-Level NLI | Premise: BM25 Evidence Passage, Hypothesis: Atomic Claim |
| **C6** | **Prompt B (Factuality-Aware)** | Hybrid Claim-Level NLI | Premise: BM25 Evidence Passage, Hypothesis: Atomic Claim |

### Experimental Controls
- **Source Invariance**: Identical source records are evaluated across all six conditions.
- **Model Parity**: Cohere Command A+ (`command-a-plus-05-2026`) with identical sampling parameters (`temperature=0.3, max_new_tokens=256, top_p=0.9`).
- **Multilingual Separation**: English and Hindi records are evaluated and reported separately.
- **Verification Engine**: Multilingual XLM-RoBERTa Large XNLI (`joeddav/xlm-roberta-large-xnli`) running deterministically on CPU.
- **Epistemic Bounds**: No synthetic composite scores; raw class probabilities and claim-level evidence rankings are preserved intact.

---

## 2. Supported Datasets & Verification Status

| Dataset | Language | Verified Identifier | Status | Role in FactIndic |
| :--- | :--- | :--- | :--- | :--- |
| **XL-Sum** | English | `csebuetnlp/xlsum` (`"english"`) | **Verified & Active** | Primary experimental benchmark |
| **XL-Sum** | Hindi | `csebuetnlp/xlsum` (`"hindi"`) | **Verified & Active** | Primary experimental benchmark |
| **MUTANT** | Hinglish | `LingoIITGN/MUTANT` | **Evaluated & Excluded** | Lacks summaries and factuality labels. Excluded from main pipeline to avoid experimental confounding. |
| **FRANK** | English | `artidoro/frank` | **Analyzed & Taxonomy Adapted** | 7-category error typology adapted for human evaluation protocol. |

---

## 3. Standardized Output Schema

All datasets are normalized and exported to UTF-8 JSONL conforming to the following record schema:

```json
{
  "example_id": "xlsum_hi_sci_201",
  "language": "hi",
  "source_text": "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने अपने तीसरे चंद्र अभियान...",
  "reference_summary": "इसरो के चंद्रयान-3 रोवर प्रज्ञान ने चंद्रमा के दक्षिणी ध्रुव पर...",
  "metadata": {
    "title": "चंद्रयान-3 ने चंद्रमा पर खनिजों की खोज की",
    "url": "https://www.bbc.com/hindi/science-201",
    "split": "test",
    "source_dataset": "csebuetnlp/xlsum"
  }
}
```

### Schema Attributes:
- **`example_id`** (`str`): Unique identifier formatted as `xlsum_{lang}_{id}`.
- **`language`** (`str`): Canonical language code (`"en"` or `"hi"`).
- **`source_text`** (`str`): Full article body, NFC Unicode-normalized, whitespace-cleaned.
- **`reference_summary`** (`str`): Ground-truth reference summary.
- **`metadata`** (`dict`): Article title, canonical URL, split, and provenance tags.

---

## 4. Environment Setup

### Prerequisites
- macOS or Linux
- Python 3.10+ (tested with Python 3.12)

### Installation
```bash
# Clone or navigate to the repository
cd factindic

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Running Tests

The test suite runs completely offline using local test fixtures without requiring network access or external credentials:

```bash
# Run the complete test suite with pytest
pytest tests/ -v
```

All 59 unit tests verify:
- Schema validation and record serialization.
- Missing and empty text detection.
- Safe Unicode NFC normalization (preserving Indic Devanagari script integrity).
- Duplicate detection (catching duplicate IDs and duplicate texts without dropping distinct articles).
- Deterministic and reproducible sampling with random seeds.
- Comprehensive dataset statistics calculation.
- Offline fixture and Hugging Face loader interfaces.

---

## 6. Live Dataset Verification (Phase 1.5)

### Strict No-Fallback Policy
- The production experimental pipeline loads exclusively from live Hugging Face `csebuetnlp/xlsum`.
- Fixture data is strictly quarantined for offline unit tests and will never enter production experimental runs.
- If live fetching fails, the pipeline halts immediately and raises the full error.

### Running Live XL-Sum Verification
To verify genuine record retrieval from Hugging Face `csebuetnlp/xlsum` for both English and Hindi:
```bash
python scripts/verify_live_xlsum.py --languages english hindi --split test --count 2
```
This script:
1. Connects directly to Hugging Face's official repository stream for `csebuetnlp/xlsum`.
2. Fetches genuine `test` split records.
3. Confirms presence of all required fields (`id`, `url`, `title`, `text`, `summary`).
4. Verifies non-empty text and reference summaries.
5. Maps records to `FactIndicRecord` schema and checks for ID duplicates.
6. Writes an audit report to `data/audits/live_xlsum_verification_audit.json`.

### Preparing Production Dataset Samples (Live by Default)
```bash
# Ingest and prepare 5 genuine live English test records
python scripts/prepare_sample.py --language en --split test --size 5

# Ingest and prepare 5 genuine live Hindi test records
python scripts/prepare_sample.py --language hi --split test --size 5

# Offline testing ONLY with mock fixtures (visibly flagged and audited as test data):
python scripts/prepare_sample.py --language en --size 5 --fixture-test-mode-only
```

---

## 7. Real-Data Summary Generation (Phase 2)

### Two Prompting Strategies
- **Prompt A (Standard Summarization):**
  `Generate a concise summary of the supplied article while preserving its main information.`
- **Prompt B (Factuality-Aware Summarization):**
  `Generate a concise summary using only information supported by the supplied article. Preserve important entities, numbers, dates, events, and relationships. Do not introduce unsupported details or external knowledge. If the source is ambiguous, do not invent a resolution.`

Generation parameters (`temperature=0.3`, `max_new_tokens=256`, `top_p=0.9`) are kept strictly identical across both prompt conditions.

### Model Inference Configuration
The primary generative model is **Command A+** (`command-a-plus-05-2026`) via Cohere V2 Chat API, which natively supports multilingual summarization across English and Hindi. (Historical baseline candidate: Aya Expanse 8B).

To execute generations, export one of the following API keys:
```bash
# Option 1: Cohere API (Primary Recommended endpoint)
export COHERE_API_KEY="your-cohere-api-key"

# Option 2: Hugging Face Inference API
export HF_TOKEN="your-huggingface-token"

# Option 3: Custom OpenAI-compatible hosted endpoint
export AYA_API_BASE="https://api.together.xyz/v1"
export AYA_API_KEY="your-together-api-key"
```

### Running the Smoke Generation Experiment
```bash
python scripts/run_generation_experiment.py --languages english hindi --split test --count 2
```
Outputs are written to:
- `data/generations/factindic_smoke_experiment_results.jsonl`: Machine-readable records with full provenance.
- `data/generations/generation_experiment_audit.json`: Experiment audit log recording retrieved record IDs and success/failure counts.

### Controlled Summary Generation Experiment (Phase 1)
```bash
python scripts/run_controlled_experiment.py
```
Executes Prompt A and Prompt B across the 10 genuine XL-Sum test samples (5 English, 5 Hindi) using Cohere Command A+ (`command-a-plus-05-2026`). Outputs are written to:
- `data/generations/controlled_generation_results.jsonl`: 20 machine-readable generation records with complete provenance.
- `data/generations/controlled_generation_audit.json`: Experiment audit log recording execution counts, parameters, and validation report.

### Phase 2.1: Demo-Only Summary Generation (Temporary Alternative Model)
For interactive demonstration unblocking while the primary research model decision remains `UNDECIDED`:
```bash
# 1. Export an accessible model key, e.g.:
export GEMINI_API_KEY="your-gemini-api-key"
# Or GROQ_API_KEY="your-groq-api-key"

# 2. Execute demo generation across the target English & Hindi XL-Sum articles
python scripts/run_demo_generation.py
```
Outputs are recorded in:
- `data/generations/factindic_demo_only_results.jsonl`: Machine-readable demo generation records.
- `data/generations/factindic_demo_only_audit.json`: Demo audit log with strict credential tracking and zero fallback.

### Phase 3: Interactive Research Demo Website (Streamlit)
To launch the interactive academic demonstration UI for faculty inspection:
```bash
streamlit run app.py
```
Features:
- **Section A (Dataset Explorer):** Inspect live, schema-validated XL-Sum articles and human-written reference summaries for English and Hindi (`world-us-canada-38697908`, `uk-wales-politics-48586353`, `081027_sadhvi_test_pp`, `070929_pakistan_protest`).
- **Section B (Prompt Comparison):** Interactive side-by-side inspection of rendered Prompt A and Prompt B inputs and generation statuses.
- **Section C (Research Pipeline):** End-to-end visualization of the $2 \times 3$ factorial pipeline and planned NLI verification approaches.
- **Section D (Progress & Provenance):** Verifiable metrics and JSON audit expanders with zero fabricated outputs.

---

## 8. Project Architecture

```
factindic/
├── app.py                        # Streamlit interactive research demo application
├── data/
│   ├── audits/                   # Live dataset verification audits
│   ├── fixtures/                 # Local offline test fixtures (English & Hindi)
│   ├── generations/              # Machine-readable generated summaries and audits
│   └── samples/                  # Prepared samples and validation audits
├── factindic/
│   ├── __init__.py               # Package public API exports
│   ├── controlled_experiment.py  # Phase 1 controlled experiment runner & validator
│   ├── generation.py             # Generation runner & GenerationRecord schema
│   ├── schema.py                 # FactIndicRecord & DatasetStatistics dataclasses
│   ├── validation.py             # Normalization, schema validation, deduplication
│   ├── sampling.py               # Deterministic reproducible sampling
│   ├── serialization.py          # UTF-8 JSONL export and import
│   ├── ui_helpers.py             # Streamlit helpers & audit parsers
│   ├── verification/             # Multilingual NLI factuality verification
│   │   ├── __init__.py           # Verification API exports
│   │   ├── config.py             # NLIConfig dataclass
│   │   ├── result.py             # NLIResult and NLILabel schemas
│   │   └── nli.py                # XNLIVerifier (joeddav/xlm-roberta-large-xnli)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseLLMClient and GenerationResult
│   │   ├── aya.py                # Command A+ / CohereClient / AyaExpanseClient
│   │   └── alternative.py        # AlternativeDemoClient (Gemini, Groq, OpenAI, etc.)
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py          # Prompt A & B templates and builders
│   └── datasets/
│       ├── __init__.py
│       ├── base.py               # Abstract BaseDatasetLoader
│       ├── xlsum.py              # XL-Sum loader (English & Hindi with streaming)
│       └── fixture_loader.py     # Local fixture loader
├── tests/                        # Comprehensive unit tests (schema, models, NLI)
├── scripts/
│   ├── prepare_sample.py         # CLI entry point for data prep
│   ├── verify_live_xlsum.py      # Strict live XL-Sum verification CLI
│   ├── run_controlled_experiment.py # Phase 1 controlled generation runner
│   ├── run_nli_smoke_test.py     # Optional live NLI verification smoke test
│   ├── run_generation_experiment.py # Phase 2 generation runner
│   └── run_demo_generation.py    # Phase 2.1 demo-only runner
├── docs/
│   ├── research_scope.md         # Detailed experimental design & feasibility report
│   ├── human_annotation_guide.md # Annotation guidelines and FRANK error taxonomy
│   └── nli_verification_guide.md # Multilingual NLI verification guide and architecture
├── requirements.txt              # Project dependencies (including streamlit, transformers, torch)
└── README.md                     # Setup, testing, and usage guide
```

---

## 9. Feasibility Analysis Summary

- **MUTANT Feasibility:** The MUTANT dataset (`LingoIITGN/MUTANT`) was evaluated. It is a token-level corpus designed for language identification in multi-sentential code-mixed text. It contains **no summaries and no factuality annotations**. Incorporating it would require authoring manual summaries from scratch, introducing uncontrolled annotator variance. It is therefore excluded from the main pipeline.
- **FRANK Benchmark:** The FRANK dataset (`artidoro/frank`) was analyzed. While its English annotations on CNN/DM and XSum cannot be directly transferred as ground truth for FactIndic, its **7-class error typology** (`PredE`, `EntE`, `CircE`, `CorefE`, `LinkE`, `OutE`, `OthE`) has been adapted to structure FactIndic's human evaluation protocol.

---

## 10. Next Steps (Subsequent Phases)
Phase 2 implements the summary generation engine and smoke test. Subsequent phases will address:
- **Phase 3:** Automated factuality verification implementation (Whole-Doc NLI, Claim-Level NLI, Hybrid).
- **Phase 4:** Human evaluation, correlation analysis, and metrics dashboard.
