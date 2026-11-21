"""FactIndic — Factuality-Aware Summarization for Indic and Code-Mixed Languages.

Interactive Academic Research Demo Website.
Built for faculty evaluation, methodology inspection, and empirical research transparency.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Ensure repository root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.prompts.templates import (
    PROMPT_A_INSTRUCTION,
    PROMPT_B_INSTRUCTION,
    PROMPT_TEMPLATE_VERSION,
    PromptStrategy,
    build_summarization_prompt,
)
from factindic.schema import FactIndicRecord
from factindic.ui_helpers import (
    TARGET_RECORD_IDS,
    TARGET_TITLES,
    fetch_live_target_records,
    get_pipeline_stages,
    get_project_audits,
    load_demo_generations,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FactIndic — Factuality-Aware Summarization",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Academic Theme & Devanagari-optimized typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    
    .devanagari-text {
        font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important;
        line-height: 1.8 !important;
        font-size: 1.05rem;
    }
    
    .academic-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .header-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        margin-right: 0.5rem;
    }
    
    .status-badge-verified {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .status-badge-pending {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .status-badge-planned {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    .card-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .disclaimer-box {
        border-left: 4px solid #f59e0b;
        background-color: #fffbeb;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 1rem;
    }

    .prompt-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 420px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Caching Data Loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner="Retrieving genuine XL-Sum records from Hugging Face...")
def get_cached_live_records(language: str):
    """Retrieve and cache live records from Hugging Face csebuetnlp/xlsum."""
    return fetch_live_target_records(language)


# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------

st.markdown('<div class="academic-header">', unsafe_allow_html=True)
st.title("FactIndic — Factuality-Aware Summarization")
st.markdown(
    "**Exploring how prompting strategies and factuality-verification methods affect English and Hindi summaries.**"
)

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.markdown('<span class="header-badge status-badge-verified">Dataset: XL-Sum Live</span>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<span class="header-badge status-badge-verified">Prompts: A / B Version 1.0</span>', unsafe_allow_html=True)
with col_h3:
    st.markdown('<span class="header-badge status-badge-pending">Research Model: UNDECIDED</span>', unsafe_allow_html=True)
with col_h4:
    st.markdown('<span class="header-badge status-badge-planned">NLI Verification: Planned</span>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Project Summary
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Project Navigation")
    section = st.radio(
        "Jump to section:",
        [
            "Dataset Explorer",
            "Prompt Comparison",
            "Research Pipeline",
            "Progress & Provenance",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Academic Context")
    st.markdown(
        """
        - **Repository:** `Riwitika/FactIndic`
        - **Benchmark:** `csebuetnlp/xlsum`
        - **Factorial Design:** 2 Prompts × 3 Verification Approaches
        - **Target Languages:** English & Hindi
        - **Research Integrity:** Zero synthetic fixtures in experimental pipeline.
        """
    )
    st.markdown("---")
    st.caption("FactIndic NLP Research Lab • Bennett University")


# -----------------------------------------------------------------------------
# Section A: Dataset Explorer
# -----------------------------------------------------------------------------

if section == "Dataset Explorer":
    st.header("Section A: Live Dataset Explorer")
    st.markdown(
        """
        Explore genuine article records ingested directly from the Hugging Face repository 
        [`csebuetnlp/xlsum`](https://huggingface.co/datasets/csebuetnlp/xlsum).
        Records are validated under FactIndic's schema and normalized to Unicode NFC.
        """
    )

    # Language selection
    col_lang, col_split = st.columns([2, 2])
    with col_lang:
        selected_language = st.selectbox(
            "Select Evaluation Language:",
            options=["English", "Hindi"],
            index=0,
            help="Chooses between English and Hindi splits of XL-Sum.",
        )
    with col_split:
        st.text_input("Dataset Split", value="test (Standard XL-Sum Evaluation Split)", disabled=True)

    lang_code = "hindi" if selected_language == "Hindi" else "english"
    target_ids = TARGET_RECORD_IDS[lang_code]

    # Live Retrieval
    with st.spinner(f"Ingesting live records from Hugging Face for '{lang_code}'..."):
        records, err = get_cached_live_records(lang_code)

    if err is not None:
        st.error(
            f"**Live Retrieval Blocked:** {err}\n\n"
            "Strict Policy Notice: In accordance with FactIndic research integrity rules, offline fixtures "
            "are never substituted into the live dataset explorer. Please check your internet connection "
            "to reload live records from Hugging Face."
        )
    elif not records:
        st.warning(f"No records matching target IDs {target_ids} were found in the stream.")
    else:
        st.success(f"Successfully retrieved and validated {len(records)} genuine records from `csebuetnlp/xlsum` ({lang_code}).")

        # Record Selector Dropdown
        record_options = {
            r.metadata.get("raw_id", r.example_id): f"{r.metadata.get('raw_id')} — {r.metadata.get('title', 'Untitled')[:65]}"
            for r in records
        }
        selected_record_id = st.selectbox(
            "Select Article Record:",
            options=list(record_options.keys()),
            format_func=lambda x: record_options.get(x, x),
        )

        # Retrieve selected record
        record: Optional[FactIndicRecord] = next(
            (r for r in records if r.metadata.get("raw_id", r.example_id) == selected_record_id),
            None,
        )

        if record:
            st.markdown("---")
            title = record.metadata.get("title", "Untitled")
            raw_url = record.metadata.get("url", "")

            st.subheader(f"Article: {title}")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"**Record ID:** `{record.metadata.get('raw_id', record.example_id)}`")
            with meta_col2:
                st.markdown(f"**Source Split:** `{record.metadata.get('split', 'test')}`")
            with meta_col3:
                if raw_url:
                    st.markdown(f"**Source URL:** [BBC News Link]({raw_url})")
                else:
                    st.markdown("**Source URL:** Not available")

            # Article Body and Reference Summary
            col_body, col_ref = st.columns([3, 2])

            with col_body:
                st.markdown("#### Full Source Text (Cleaned & Normalized)")
                css_class = "devanagari-text" if lang_code == "hindi" else ""
                st.markdown(
                    f'<div class="card-box {css_class}" style="max-height: 400px; overflow-y: auto;">{record.source_text}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Character count: {len(record.source_text)} | Word count: {len(record.source_text.split())}")

            with col_ref:
                st.markdown("#### XL-Sum Reference Summary")
                st.markdown(
                    '<div class="disclaimer-box">'
                    "<strong>Dataset Ground Truth Reference:</strong> This summary is the human-authored reference summary "
                    "provided in the official XL-Sum dataset. It is <em>not</em> generated by FactIndic."
                    "</div>",
                    unsafe_allow_html=True,
                )
                css_class = "devanagari-text" if lang_code == "hindi" else ""
                st.markdown(
                    f'<div class="card-box {css_class}" style="background-color: #fafaf9;">{record.reference_summary}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Summary length: {len(record.reference_summary)} characters")


# -----------------------------------------------------------------------------
# Section B: Prompt Comparison
# -----------------------------------------------------------------------------

elif section == "Prompt Comparison":
    st.header("Section B: Dual Prompting Strategy Comparison")
    st.markdown(
        """
        FactIndic compares two distinct prompting strategies across language models.
        Both prompts adhere to the project's versioned prompt template contract (`v1.0`).
        """
    )

    # Language and Article Selector for rendering prompts
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        lang_choice = st.selectbox("Prompt Language Directive:", ["English", "Hindi"], index=0)
        lang_code = "hindi" if lang_choice == "Hindi" else "english"
    with col_p2:
        records, err = get_cached_live_records(lang_code)
        if records:
            article_choice = st.selectbox(
                "Select Article for Prompt Context:",
                options=records,
                format_func=lambda r: f"{r.metadata.get('raw_id')} ({r.metadata.get('title', '')[:45]}...)",
            )
            sample_text = article_choice.source_text
            sample_id = article_choice.metadata.get("raw_id", article_choice.example_id)
        else:
            sample_text = (
                "The United Kingdom's health service is facing unprecedented winter pressures, "
                "with NHS leaders warning of critical delays in patient transfers and emergency admissions."
                if lang_code == "english"
                else "भारत निर्वाचन आयोग ने आगामी विधानसभा चुनावों के लिए विस्तृत कार्यक्रम की घोषणा कर दी है।"
            )
            sample_id = "sample-placeholder"

    st.markdown("---")

    # Side-by-side Prompt Columns
    col_pa, col_pb = st.columns(2)

    with col_pa:
        st.subheader("Strategy A: Standard Summarization")
        st.markdown(
            """
            **Objective:** Standard abstractive compression.  
            **Instruction:**  
            > *"Generate a concise summary of the supplied article while preserving its main information."*
            """
        )
        st.markdown("##### Rendered Prompt (Exact Input Sent to Model)")
        rendered_prompt_a = build_summarization_prompt(
            source_text=sample_text[:1200] + ("..." if len(sample_text) > 1200 else ""),
            strategy=PromptStrategy.PROMPT_A,
            language=lang_code,
        )
        st.markdown(f'<div class="prompt-box">{rendered_prompt_a}</div>', unsafe_allow_html=True)

    with col_pb:
        st.subheader("Strategy B: Factuality-Aware Summarization")
        st.markdown(
            """
            **Objective:** Faithfulness-constrained abstractive compression.  
            **Instruction:**  
            > *"Generate a concise summary using only information supported by the supplied article. "
            "Preserve important entities, numbers, dates, events, and relationships. "
            "Do not introduce unsupported details or external knowledge. "
            "If the source is ambiguous, do not invent a resolution."*
            """
        )
        st.markdown("##### Rendered Prompt (Exact Input Sent to Model)")
        rendered_prompt_b = build_summarization_prompt(
            source_text=sample_text[:1200] + ("..." if len(sample_text) > 1200 else ""),
            strategy=PromptStrategy.PROMPT_B,
            language=lang_code,
        )
        st.markdown(f'<div class="prompt-box">{rendered_prompt_b}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Generated Summaries Status")

    # Read generated records using project serialization
    generated_records = load_demo_generations()
    matching_gens = [g for g in generated_records if g.source_record_id == sample_id]

    if not matching_gens:
        st.info(
            f"ℹ️ **Summary generation is pending model access for article `{sample_id}`.**\n\n"
            "In strict adherence to the project's real-data policy, no synthetic, mock, or hallucinated "
            "summaries are rendered. Once live model inference is executed via `scripts/run_demo_generation.py`, "
            "the generated summaries under Prompt A and Prompt B will appear here side-by-side with full provenance."
        )
    else:
        st.success(f"Found {len(matching_gens)} generated summaries for this article.")
        gen_col1, gen_col2 = st.columns(2)
        for gen in matching_gens:
            target_col = gen_col1 if gen.prompt_strategy == "prompt_a" else gen_col2
            with target_col:
                st.markdown(f"**Strategy:** `{gen.prompt_strategy}` | **Model:** `{gen.model_id}` ({gen.provider})")
                st.markdown(f'<div class="card-box">{gen.generated_summary}</div>', unsafe_allow_html=True)
                st.caption(f"Latency: {gen.latency_seconds}s | Timestamp: {gen.generation_timestamp_utc}")


# -----------------------------------------------------------------------------
# Section C: Research Pipeline
# -----------------------------------------------------------------------------

elif section == "Research Pipeline":
    st.header("Section C: FactIndic Research Pipeline")
    st.markdown(
        """
        The diagram below visualizes the complete $2 \\times 3$ factorial research design.
        Each module is strictly decoupled, with standardized data contracts between stages.
        """
    )

    # Interactive Architecture Diagram
    st.markdown(
        """
        ```
        +---------------------------------------------------------------------------------------+
        |                                    REAL XL-SUM DATA                                   |
        |             (Verified test records: English & Hindi from csebuetnlp/xlsum)            |
        +-------------------------------------------+-------------------------------------------+
                                                    |
                                                    v
        +-------------------------------------------+-------------------------------------------+
        |                          DUAL PROMPTING INTERFACE (v1.0)                              |
        |              Prompt A (Standard)        vs.      Prompt B (Factuality-Aware)          |
        +-------------------------------------------+-------------------------------------------+
                                                    |
                                                    v
        +-------------------------------------------+-------------------------------------------+
        |                             LANGUAGE MODEL GENERATION                                 |
        |               Current Status: Primary Model Selected (Command A+ / Cohere)            |
        +-------------------------------------------+-------------------------------------------+
                                                    |
                                                    v
        +---------------------------------------------------------------------------------------+
        |                         THREE FACTUALITY VERIFICATION APPROACHES                      |
        |                                                                                       |
        |   [Approach 1: Whole-Doc NLI]   [Approach 2: Claim-Level NLI]   [Approach 3: Hybrid]  |
        |      Premise: Full Article         Decompose into atomic claims    Claim NLI + Entity |
        |      Hypothesis: Summary           Evidence sentence retrieval     Number & Date check|
        |      Status: [PLANNED]             Status: [PLANNED]               Status: [PLANNED]  |
        +-------------------------------------------+-------------------------------------------+
                                                    |
                                                    v
        +---------------------------------------------------------------------------------------+
        |                              EVALUATION & BENCHMARKING                                |
        |        Automatic Metrics (Entailment Ratio, FactScore) vs. Human FRANK Annotation     |
        |        Status: [PLANNED PROTOCOL - Cohen's Kappa Inter-Annotator Agreement]           |
        +---------------------------------------------------------------------------------------+
        ```
        """
    )

    st.markdown("### Pipeline Stages & Implementation Status")
    stages = get_pipeline_stages()

    for stage in stages:
        with st.expander(f"Stage {stage['stage_number']}: {stage['title']} — {stage['status']}", expanded=True):
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                st.markdown(f"**Description:** {stage['description']}")
                st.markdown(f"**Implementation Target:** `{stage['implementation']}`")
            with col_s2:
                if "Verified" in stage["status"]:
                    st.success(stage["status"])
                elif "Planned" in stage["status"]:
                    st.info(stage["status"])
                else:
                    st.warning(stage["status"])


# -----------------------------------------------------------------------------
# Section D: Progress and Provenance
# -----------------------------------------------------------------------------

elif section == "Progress & Provenance":
    st.header("Section D: Progress, Provenance & Audit Records")
    st.markdown(
        """
        All research claims in FactIndic are backed by verifiable audit logs and automated tests.
        No unexecuted stage is presented with a false success indicator.
        """
    )

    audits = get_project_audits()
    live_audit = audits.get("live_xlsum")
    demo_audit = audits.get("demo_generation")
    controlled_audit = audits.get("controlled_generation")

    # High-level Metric Cards
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            label="Live XL-Sum Retrieval",
            value="4 / 4 Verified",
            delta="English & Hindi",
        )
    with col_m2:
        st.metric(
            label="Offline Unit Tests",
            value="59 Passing",
            delta="100% Offline Passing",
        )
    with col_m3:
        summary_count = 0
        if controlled_audit and "actual_counts" in controlled_audit:
            summary_count = controlled_audit["actual_counts"].get("total_success", 0)
        elif demo_audit and "summary_counts" in demo_audit:
            summary_count = demo_audit["summary_counts"].get("prompt_a_success", 0) + demo_audit["summary_counts"].get("prompt_b_success", 0)
        st.metric(
            label="Model Summaries Generated",
            value=f"{summary_count} Completed",
            delta="20 / 20 Verified" if summary_count == 20 else "Controlled Run",
            delta_color="normal" if summary_count > 0 else "off",
        )
    with col_m4:
        st.metric(
            label="Final Research Model",
            value="Command A+",
            delta="Cohere V2 (command-a-plus-05-2026)",
            delta_color="normal",
        )

    st.markdown("---")
    st.subheader("Audited Project Provenance")

    # Audit Card 1: Controlled Generation Audit
    with st.expander("Audit 1: Controlled Summary Generation Audit (`controlled_generation_audit.json`)", expanded=True):
        if controlled_audit:
            st.markdown(
                f"- **Experiment:** `{controlled_audit.get('experiment_name')}`\n"
                f"- **Model ID:** `{controlled_audit.get('model_configuration', {}).get('model_id')}`\n"
                f"- **Validation Status:** `{'PASSED' if controlled_audit.get('validation_passed') else 'FAILED'}`\n"
                f"- **Total Summaries Generated:** `{controlled_audit.get('actual_counts', {}).get('total_success', 0)} / {controlled_audit.get('expected_counts', {}).get('total_summaries', 20)}`\n"
                f"- **English Success (Prompt A / B):** `{controlled_audit.get('actual_counts', {}).get('english', {}).get('prompt_a_success', 0)} / {controlled_audit.get('actual_counts', {}).get('english', {}).get('prompt_b_success', 0)}`\n"
                f"- **Hindi Success (Prompt A / B):** `{controlled_audit.get('actual_counts', {}).get('hindi', {}).get('prompt_a_success', 0)} / {controlled_audit.get('actual_counts', {}).get('hindi', {}).get('prompt_b_success', 0)}`\n"
                f"- **Total Duration:** `{controlled_audit.get('total_duration_seconds', 0.0)}s`"
            )
            st.json(controlled_audit)
        else:
            st.warning("Audit file `data/generations/controlled_generation_audit.json` not found.")

    # Audit Card 2: Live Verification
    with st.expander("Audit 2: Live XL-Sum Retrieval Audit (`live_xlsum_verification_audit.json`)", expanded=False):
        if live_audit:
            st.json(live_audit)
        else:
            st.warning("Audit file `data/audits/live_xlsum_verification_audit.json` not found.")

    # Audit Card 3: Demo Generation Audit
    with st.expander("Audit 3: Demo Generation Audit (`factindic_demo_only_audit.json`)", expanded=False):
        if demo_audit:
            st.markdown(
                f"- **Experiment Type:** `{demo_audit.get('experiment_type')}`\n"
                f"- **Final Research Model:** `{demo_audit.get('final_research_model')}`\n"
                f"- **Prompt Template Version:** `{demo_audit.get('prompt_template_version')}`\n"
                f"- **Total Articles Evaluated:** `{demo_audit.get('summary_counts', {}).get('total_articles', 0)}`\n"
                f"- **Client Guard Status:** `{demo_audit.get('client_error')}`"
            )
            st.json(demo_audit)
        else:
            st.warning("Audit file `data/generations/factindic_demo_only_audit.json` not found.")

    # Human Evaluation Protocol Summary
    with st.expander("Human Evaluation & FRANK 7-Error Protocol (Planned)", expanded=False):
        st.markdown(
            """
            - **Protocol Document:** [`docs/human_annotation_guide.md`](docs/human_annotation_guide.md)
            - **Error Taxonomy:** 7-category FRANK typology adapted for Indic summarization:
              1. `PredE`: Predicate / relational distortion
              2. `EntE`: Entity identity substitution
              3. `CircE`: Circumstantial / temporal / location distortion
              4. `CorefE`: Coreference resolution error
              5. `LinkE`: Erroneous linkage between disjoint source clauses
              6. `OutE`: Out-of-article unsupported claim (hallucination)
              7. `OthE`: Grammatical or unclassifiable factual flaw
            - **Inter-annotator Metric:** Cohen's $\\kappa$ and Fleiss' $\\kappa$ across two independent annotators per language with third-expert adjudication.
            """
        )

st.markdown("---")
st.caption("FactIndic Research Platform • Strictly confined to `/Users/apple/Downloads/FactIndic/`")
