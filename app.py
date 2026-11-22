"""FactIndic — Factuality-Aware Summarization for Indic and Code-Mixed Languages.

Interactive Academic Research Platform & Faculty Evaluation Dashboard.
Presents the completed 60-output controlled benchmark experiment using local sequence-classification
NLI inference (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    filter_experiment_records,
    get_controlled_experiment_dir,
    get_hybrid_claim_explorer_detail,
    get_hybrid_claim_explorer_records,
    get_pipeline_stages,
    get_project_audits,
    get_qualitative_case_studies,
    load_controlled_experiment_audit,
    load_controlled_experiment_results,
    load_controlled_experiment_summary_table,
    load_demo_generations,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FactIndic — Research Evaluation Dashboard",
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
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .header-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        margin-right: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .status-badge-verified {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }

    .status-badge-info {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }

    .badge-entailment {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .badge-neutral {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .badge-contradiction {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .card-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .kpi-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f8fafc;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-card .kpi-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .disclaimer-box {
        border-left: 4px solid #f59e0b;
        background-color: #fffbeb;
        padding: 0.85rem 1.15rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 1.25rem;
        line-height: 1.6;
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

@st.cache_data(show_spinner=False)
def get_cached_live_records(language: str):
    """Retrieve and cache live records from Hugging Face csebuetnlp/xlsum."""
    return fetch_live_target_records(language)


@st.cache_data(show_spinner=False)
def get_cached_experiment_data():
    """Load and cache the complete 60-evaluation controlled experiment outputs."""
    audit = load_controlled_experiment_audit()
    table = load_controlled_experiment_summary_table()
    results = load_controlled_experiment_results()
    return audit, table, results


# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------

st.markdown('<div class="academic-header">', unsafe_allow_html=True)
st.title("FactIndic — Factuality-Aware Summarization")
st.markdown(
    "**Evaluating Factuality-Aware Prompting and Multi-Granularity Local NLI Verification Across English and Hindi Summaries**"
)

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.markdown('<span class="header-badge status-badge-verified">Dataset: BBC XL-Sum (20 Frozen)</span>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<span class="header-badge status-badge-verified">Generation: Command A+ (Prompt A & B)</span>', unsafe_allow_html=True)
with col_h3:
    st.markdown('<span class="header-badge status-badge-verified">NLI: mDeBERTa-v3 (Local CPU)</span>', unsafe_allow_html=True)
with col_h4:
    st.markdown('<span class="header-badge status-badge-verified">Evaluations: 60 / 60 Completed (100%)</span>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Academic Metadata
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Research Navigation")
    section = st.radio(
        "Select Dashboard View:",
        [
            "📊 Experimental Results (C1–C6)",
            "🌐 English vs. Hindi Comparison",
            "🔍 Hybrid Claim Explorer",
            "💡 Qualitative Case Studies",
            "⚠️ Research Limitations",
            "📑 Dataset & Prompt Explorer",
            "📜 Provenance & Audit Logs",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Faculty Evaluation Context")
    st.markdown(
        """
        - **Project:** FactIndic Benchmark
        - **Institution:** Bennett University
        - **Repository:** `Riwitika/FactIndic`
        - **Factorial Design:** 2 Prompts × 3 Verification Strategies × 2 Languages
        - **Total Evaluations:** 60 Completed (0 Quota Blocks)
        - **NLI Model:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
        - **Pinned Revision:** `b5113eb`
        - **Execution:** Offline Local CPU
        """
    )
    st.markdown("---")
    st.caption("FactIndic NLP Research Lab • Bennett University")


# Load experiment datasets
exp_audit, exp_summary_table, exp_results = get_cached_experiment_data()


# -----------------------------------------------------------------------------
# View 1: Experimental Results (C1–C6)
# -----------------------------------------------------------------------------

if section == "📊 Experimental Results (C1–C6)":
    st.header("Section 1: Controlled Benchmark Results (C1–C6)")
    st.markdown(
        """
        This dashboard presents the empirical findings from the completed **60-evaluation controlled experiment**.
        Summaries were evaluated across a $2 \\times 3 \\times 2$ factorial matrix:
        - **Generation Prompts:** Prompt A (Standard Abstractive) vs. Prompt B (Factuality-Aware Constrained).
        - **Verification Architectures:** Whole-Document NLI, Sentence-Level NLI, and Hybrid Claim-Level NLI.
        - **Languages:** English and Hindi (10 frozen summaries each).
        """
    )

    # Top KPI Metrics Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">60 / 60</div>
                <div class="kpi-label">Evaluations Completed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">100%</div>
                <div class="kpi-label">Evaluation Success Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">473.3s</div>
                <div class="kpi-label">Total CPU Execution Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-value">0</div>
                <div class="kpi-label">API Quota Dependencies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive Filters
    st.subheader("Filter Evaluation Records")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_lang = st.selectbox(
            "Filter by Language:",
            options=["All Languages", "English (en)", "Hindi (hi)"],
            index=0,
            help="Filter between English, Hindi, or combined cross-lingual views.",
        )
    with col_f2:
        selected_prompt = st.selectbox(
            "Filter by Prompt Strategy:",
            options=["All Prompts", "Prompt A (Standard)", "Prompt B (Factuality-Aware)"],
            index=0,
            help="Filter by generation prompt directive.",
        )
    with col_f3:
        selected_strategy = st.selectbox(
            "Filter by Verification Strategy:",
            options=["All Strategies", "Whole-Document NLI", "Sentence-Level NLI", "Hybrid Claim-Level NLI"],
            index=0,
            help="Filter by verification architecture granularity.",
        )

    # Filter summary table data
    lang_code = "all" if "All" in selected_lang else ("hi" if "Hindi" in selected_lang else "en")
    filtered_table = [r for r in exp_summary_table if r.get("language") == lang_code]

    if "Prompt A" in selected_prompt:
        filtered_table = [r for r in filtered_table if r.get("prompt_strategy") == "prompt_a"]
    elif "Prompt B" in selected_prompt:
        filtered_table = [r for r in filtered_table if r.get("prompt_strategy") == "prompt_b"]

    if "Whole" in selected_strategy:
        filtered_table = [r for r in filtered_table if r.get("verification_strategy") == "whole_document"]
    elif "Sentence" in selected_strategy:
        filtered_table = [r for r in filtered_table if r.get("verification_strategy") == "sentence_level"]
    elif "Hybrid" in selected_strategy:
        filtered_table = [r for r in filtered_table if r.get("verification_strategy") == "hybrid_claim_level"]

    df_filtered = pd.DataFrame(filtered_table)

    if not df_filtered.empty:
        # Visualizations: Grouped Bar Chart of NLI Label Ratios
        st.subheader("NLI Distribution Ratios by Condition")
        chart_data = []
        for _, row in df_filtered.iterrows():
            cid = f"{row['condition_id']} ({row['prompt_strategy'].upper()}, {row['verification_strategy'][:6]})"
            chart_data.append({"Condition": cid, "Metric": "Entailment Ratio", "Ratio": row["macro_entailment_ratio"]})
            chart_data.append({"Condition": cid, "Metric": "Neutral Ratio", "Ratio": row["macro_neutral_ratio"]})
            chart_data.append({"Condition": cid, "Metric": "Contradiction Ratio", "Ratio": row["macro_contradiction_ratio"]})

        df_chart = pd.DataFrame(chart_data)
        fig_ratios = px.bar(
            df_chart,
            x="Condition",
            y="Ratio",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "Entailment Ratio": "#22c55e",
                "Neutral Ratio": "#64748b",
                "Contradiction Ratio": "#ef4444",
            },
            title="Macro Entailment, Neutral, and Contradiction Ratios",
            height=380,
        )
        fig_ratios.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 1.05], tickformat=".1%"),
        )
        st.plotly_chart(fig_ratios, use_container_width=True)

        # Latency Comparison Chart
        st.subheader("Mean Verification Latency per Summary (CPU)")
        fig_lat = px.bar(
            df_filtered,
            x="condition_id",
            y="mean_latency_ms",
            color="verification_strategy",
            color_discrete_map={
                "whole_document": "#3b82f6",
                "sentence_level": "#f97316",
                "hybrid_claim_level": "#8b5cf6",
            },
            title="Mean Latency per Summary (Milliseconds on Local CPU)",
            labels={"condition_id": "Condition", "mean_latency_ms": "Mean Latency (ms)", "verification_strategy": "Strategy"},
            height=320,
        )
        fig_lat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_lat, use_container_width=True)

        # Full Data Table
        st.subheader("Experimental Summary Table (Filtered View)")
        display_cols = [
            "condition_id",
            "prompt_strategy",
            "verification_strategy",
            "language",
            "total_evaluations",
            "macro_mean_entailment_score",
            "macro_entailment_ratio",
            "macro_contradiction_ratio",
            "macro_neutral_ratio",
            "mean_latency_ms",
        ]
        df_display = df_filtered[display_cols].copy()
        df_display.columns = [
            "Condition",
            "Prompt",
            "Verification Strategy",
            "Language",
            "N",
            "Mean Entail Score",
            "Entail Ratio",
            "Contra Ratio",
            "Neutral Ratio",
            "Mean Latency (ms)",
        ]
        st.dataframe(df_display.style.format({
            "Mean Entail Score": "{:.4f}",
            "Entail Ratio": "{:.2%}",
            "Contra Ratio": "{:.2%}",
            "Neutral Ratio": "{:.2%}",
            "Mean Latency (ms)": "{:,.1f}",
        }), use_container_width=True)
    else:
        st.warning("No records matched the selected combination of filters.")


# -----------------------------------------------------------------------------
# View 2: English vs. Hindi Comparison
# -----------------------------------------------------------------------------

elif section == "🌐 English vs. Hindi Comparison":
    st.header("Section 2: English vs. Hindi Cross-Lingual Analysis")
    st.markdown(
        """
        To prevent obscuring typological and lexical differences, **English and Hindi results must be analyzed separately**.
        The benchmark evaluates 10 English and 10 Hindi summaries across all 6 conditions.
        """
    )

    tab_en, tab_hi, tab_contra = st.tabs(["🇬🇧 English Findings", "🇮🇳 Hindi Findings", "⚖️ Subgroup Contradiction Comparison"])

    with tab_en:
        st.subheader("English Results Across Conditions (N = 5 per condition)")
        en_rows = [r for r in exp_summary_table if r.get("language") == "en"]
        df_en = pd.DataFrame(en_rows)
        if not df_en.empty:
            col_en1, col_en2 = st.columns(2)
            with col_en1:
                st.markdown("##### English Entailment & Contradiction Ratios")
                fig_en = px.bar(
                    df_en,
                    x="condition_id",
                    y=["macro_entailment_ratio", "macro_contradiction_ratio"],
                    barmode="group",
                    color_discrete_map={"macro_entailment_ratio": "#22c55e", "macro_contradiction_ratio": "#ef4444"},
                    labels={"value": "Ratio", "condition_id": "Condition"},
                    title="English Verification Ratios by Condition",
                    height=320,
                )
                fig_en.update_layout(yaxis=dict(tickformat=".1%"), margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_en, use_container_width=True)

            with col_en2:
                st.markdown("##### Key English Observations")
                st.markdown(
                    """
                    - **Whole-Document NLI (C1 vs. C2):** Prompt A had 0.0% entailment (all 5 neutral). Prompt B jumped to 40.0% entailment (mean score: 0.0547 $\\to$ 0.4549).
                    - **Sentence-Level NLI (C3 vs. C4):** Macro contradiction ratio shifted slightly from 6.54% to 6.25% (-4.4% relative).
                    - **Hybrid Claim-Level (C5 vs. C6):** Contradiction ratio rose from 1.54% to 5.33% (2 contradicted claims out of 43 under Prompt B vs. 1 out of 63 under Prompt A).
                    """
                )

    with tab_hi:
        st.subheader("Hindi Results Across Conditions (N = 5 per condition)")
        hi_rows = [r for r in exp_summary_table if r.get("language") == "hi"]
        df_hi = pd.DataFrame(hi_rows)
        if not df_hi.empty:
            col_hi1, col_hi2 = st.columns(2)
            with col_hi1:
                st.markdown("##### Hindi Entailment & Contradiction Ratios")
                fig_hi = px.bar(
                    df_hi,
                    x="condition_id",
                    y=["macro_entailment_ratio", "macro_contradiction_ratio"],
                    barmode="group",
                    color_discrete_map={"macro_entailment_ratio": "#22c55e", "macro_contradiction_ratio": "#ef4444"},
                    labels={"value": "Ratio", "condition_id": "Condition"},
                    title="Hindi Verification Ratios by Condition",
                    height=320,
                )
                fig_hi.update_layout(yaxis=dict(tickformat=".1%"), margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_hi, use_container_width=True)

            with col_hi2:
                st.markdown("##### Key Hindi Observations")
                st.markdown(
                    """
                    - **Consistent Contradiction Reductions:** Prompt B substantially reduced contradictions in both fine-grained strategies:
                      - *Sentence-Level:* fell from **10.22% (C3) to 5.50% (C4)** (-46.2% relative).
                      - *Hybrid Claim-Level:* plunged from **10.67% (C5) to 2.86% (C6)** (-73.2% relative).
                    - **Sentence Entailment Sensitivity:** Hindi sentence-level entailment reached **80.25% in C4**, demonstrating strong cross-lingual semantic alignment.
                    - **Hybrid Neutrality Shift:** Hindi hybrid neutral ratio rose to 59.96% in C6, reflecting lexical retrieval bottlenecks when queries contained English planning text.
                    """
                )

    with tab_contra:
        st.subheader("Subgroup Contradiction Comparison (Prompt A vs. Prompt B)")
        st.markdown(
            """
            The empirical data reveals that Prompt B reduced the **overall macro contradiction ratio** in both fine-grained verification strategies.
            However, **its effect varied across languages and strategies**, and Prompt B should not be claimed to consistently improve factuality in every subgroup:
            """
        )
        contra_data = [
            {"Verification Strategy": "Sentence-Level", "Subgroup": "English", "Prompt A": 0.0654, "Prompt B": 0.0625, "Diff": -0.0029, "Rel Change": "-4.4%"},
            {"Verification Strategy": "Sentence-Level", "Subgroup": "Hindi", "Prompt A": 0.1022, "Prompt B": 0.0550, "Diff": -0.0472, "Rel Change": "-46.2%"},
            {"Verification Strategy": "Sentence-Level", "Subgroup": "Overall (Combined)", "Prompt A": 0.0838, "Prompt B": 0.0588, "Diff": -0.0250, "Rel Change": "-29.8%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Subgroup": "English", "Prompt A": 0.0154, "Prompt B": 0.0533, "Diff": +0.0379, "Rel Change": "+246.1%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Subgroup": "Hindi", "Prompt A": 0.1067, "Prompt B": 0.0286, "Diff": -0.0781, "Rel Change": "-73.2%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Subgroup": "Overall (Combined)", "Prompt A": 0.0610, "Prompt B": 0.0410, "Diff": -0.0200, "Rel Change": "-32.8%"},
        ]
        df_contra = pd.DataFrame(contra_data)
        st.dataframe(df_contra.style.format({
            "Prompt A": "{:.2%}",
            "Prompt B": "{:.2%}",
            "Diff": "{:+.2%}",
        }), use_container_width=True)

        st.caption("Note: Whole-document verification yielded 0.00% contradiction across all conditions due to neutral-class dominance.")


# -----------------------------------------------------------------------------
# View 3: Hybrid Claim-Level Verification Explorer
# -----------------------------------------------------------------------------

elif section == "🔍 Hybrid Claim Explorer":
    st.header("Section 3: Hybrid Claim-Level Verification Explorer")
    st.markdown(
        """
        Explore the fine-grained inner workings of **Hybrid Claim-Level Verification**:
        1. Summaries are segmented into standalone atomic factual claims via a deterministic rule-based extractor.
        2. Top-$K$ ($K=3$) relevant passages are retrieved from the source article using Okapi BM25.
        3. Each retrieved passage is evaluated independently via mDeBERTa cross-encoder sequence classification.
        4. Summary-level metrics are computed via deterministic two-level existential grounding.
        """
    )

    # Explanation of aggregation rule
    st.markdown(
        r"""
        <div class="disclaimer-box">
            <strong>Deterministic Decision Rule (Level 1 Passage-to-Claim Grounding):</strong><br>
            A claim \(c\) evaluated against retrieved passages \(\{p_1, p_2, p_3\}\) is assigned:
            <ul>
                <li><strong>Entailment:</strong> If \(\max_{k} P(\text{entailment} \mid p_k, c) \ge 0.50\)</li>
                <li><strong>Contradiction:</strong> If \(\max_{k} P(\text{contradiction} \mid p_k, c) \ge 0.50\) and \(\max_{k} P(\text{entailment}) < 0.50\)</li>
                <li><strong>Neutral:</strong> Otherwise.</li>
            </ul>
            <em>Important Research Note: This aggregation rule is the project's selected decision policy, not an empirically proven optimal threshold. Scores from different passages are evaluated independently and never combined into a single three-class probability distribution.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hybrid_recs = get_hybrid_claim_explorer_records()

    if not hybrid_recs:
        st.warning("No hybrid claim-level records found in the experiment results.")
    else:
        col_c_sel, col_a_sel = st.columns([1, 2])
        with col_c_sel:
            cond_choice = st.selectbox(
                "Select Condition:",
                options=["Condition C5 (Prompt A + Hybrid)", "Condition C6 (Prompt B + Hybrid)"],
                index=0,
            )
            target_cond = "C5" if "C5" in cond_choice else "C6"

        filtered_hybrid = [r for r in hybrid_recs if r.get("condition_id") == target_cond]

        with col_a_sel:
            article_options = {
                r["source_record_id"]: f"{r['source_record_id']} [{r['language'].upper()}] — {r['generated_summary'][:60]}..."
                for r in filtered_hybrid
            }
            selected_rid = st.selectbox(
                "Select Article Record / Summary:",
                options=list(article_options.keys()),
                format_func=lambda x: article_options.get(x, x),
            )

        # Retrieve selected record detail
        target_rec = next((r for r in filtered_hybrid if r["source_record_id"] == selected_rid), None)

        if target_rec:
            detail = get_hybrid_claim_explorer_detail(target_rec)
            agg = detail["aggregated_metrics"]

            st.markdown("---")
            st.subheader(f"Summary Evaluation: `{selected_rid}` ({detail['language'].upper()})")

            # Metadata and Summary Card
            col_sum1, col_sum2 = st.columns([3, 2])
            with col_sum1:
                st.markdown("##### Generated Summary")
                lang_css = "devanagari-text" if detail["language"] == "hi" else ""
                st.markdown(f'<div class="card-box {lang_css}">{detail["generated_summary"]}</div>', unsafe_allow_html=True)

            with col_sum2:
                st.markdown("##### Summary-Level Verification KPIs")
                st.markdown(
                    f"""
                    - **Total Atomic Claims:** `{agg.get('total_claims', 0)}`
                    - **Entailed Claims:** `{agg.get('entailed_claims', 0)}` ({agg.get('entailment_ratio', 0.0):.1%})
                    - **Neutral Claims:** `{agg.get('neutral_claims', 0)}` ({agg.get('neutral_ratio', 0.0):.1%})
                    - **Contradicted Claims:** `{agg.get('contradicted_claims', 0)}` ({agg.get('contradiction_ratio', 0.0):.1%})
                    - **Mean Claim Entailment Score:** `{agg.get('mean_claim_entailment', 0.0):.4f}`
                    - **Inference Latency:** `{detail['verification_latency_ms']:.1f} ms`
                    """
                )

            # Claims Inspector
            st.markdown("---")
            st.subheader(f"Extracted Atomic Claims & BM25 Evidence ({len(detail['claims'])} Claims)")

            for idx, claim in enumerate(detail["claims"], 1):
                label = claim["predicted_label"]
                badge_class = (
                    "badge-entailment" if label == "entailment"
                    else ("badge-contradiction" if label == "contradiction" else "badge-neutral")
                )

                with st.expander(f"Claim #{idx}: {claim['atomic_claim_text'][:80]}... — [{label.upper()}]", expanded=(idx <= 2)):
                    st.markdown(f"**Atomic Claim Text:**")
                    st.markdown(f"> *{claim['atomic_claim_text']}*")

                    col_cl1, col_cl2, col_cl3 = st.columns(3)
                    with col_cl1:
                        st.markdown(f"**Verdict:** <span class=\"{badge_class}\">{label.upper()}</span>", unsafe_allow_html=True)
                    with col_cl2:
                        st.markdown(f"**Max Entailment Probability:** `{claim['max_entailment_prob']:.4f}`")
                    with col_cl3:
                        st.markdown(f"**Max Contradiction Probability:** `{claim['max_contradiction_prob']:.4f}`")

                    st.markdown("##### Retrieved BM25 Evidence Passages ($K=3$)")
                    passages = claim.get("passages", [])
                    if not passages:
                        st.info("No passages were retrieved for this claim.")
                    else:
                        for p in passages:
                            p_label = p.get("predicted_label", "neutral")
                            p_badge = (
                                "badge-entailment" if p_label == "entailment"
                                else ("badge-contradiction" if p_label == "contradiction" else "badge-neutral")
                            )
                            st.markdown(
                                f"""
                                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.6rem;">
                                    <strong>Passage Rank {p.get('evidence_rank')}</strong> | 
                                    BM25 Score: <code>{p.get('evidence_score', 0.0):.2f}</code> | 
                                    NLI Output: <span class="{p_badge}">{p_label.upper()}</span><br>
                                    <div style="margin-top: 0.4rem; font-style: italic; color: #334155;">"{p.get('evidence_passage', '')}"</div>
                                    <div style="margin-top: 0.4rem; font-size: 0.82rem; color: #64748b;">
                                        P(Entailment): <strong>{p.get('entailment_prob', 0.0):.4f}</strong> | 
                                        P(Neutral): <strong>{p.get('neutral_prob', 0.0):.4f}</strong> | 
                                        P(Contradiction): <strong>{p.get('contradiction_prob', 0.0):.4f}</strong>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


# -----------------------------------------------------------------------------
# View 4: Qualitative Case Studies
# -----------------------------------------------------------------------------

elif section == "💡 Qualitative Case Studies":
    st.header("Section 4: Qualitative Error & Grounding Case Studies")
    st.markdown(
        """
        Concrete examples extracted directly from machine-readable benchmark outputs 
        illustrating premise-hypothesis interactions, hallucination detection, and retrieval dynamics.
        """
    )

    cases = get_qualitative_case_studies()
    case_titles = [f"{c['title']} ({c['language']})" for c in cases]
    selected_case_title = st.selectbox("Select Case Study to Inspect:", options=case_titles)

    chosen_case = next((c for c in cases if f"{c['title']} ({c['language']})" == selected_case_title), cases[0])

    st.markdown("---")
    st.subheader(chosen_case["title"])

    meta_c1, meta_c2, meta_c3 = st.columns(3)
    with meta_c1:
        st.markdown(f"**Record ID:** `{chosen_case['record_id']}`")
    with meta_c2:
        st.markdown(f"**Condition:** `{chosen_case['condition_id']}` ({chosen_case['strategy']})")
    with meta_c3:
        lbl = chosen_case["assigned_label"]
        b_class = "badge-entailment" if lbl == "entailment" else ("badge-contradiction" if lbl == "contradiction" else "badge-neutral")
        st.markdown(f"**Assigned Label:** <span class=\"{b_class}\">{lbl.upper()}</span>", unsafe_allow_html=True)

    st.markdown("##### Extracted Hypothesis (Claim)")
    st.markdown(f'> *"{chosen_case["claim_text"]}"*')

    st.markdown("##### Retrieved Premise (BM25 Evidence Passage)")
    st.markdown(f'> *"{chosen_case["retrieved_premise"]}"*')

    probs = chosen_case["probabilities"]
    col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
    with col_pr1:
        st.metric("P(Entailment)", f"{probs['entailment']:.4f}")
    with col_pr2:
        st.metric("P(Neutral)", f"{probs['neutral']:.4f}")
    with col_pr3:
        st.metric("P(Contradiction)", f"{probs['contradiction']:.4f}")
    with col_pr4:
        st.metric("BM25 Score", f"{chosen_case['bm25_score']:.2f}")

    st.markdown("##### Research Analysis & Epistemic Interpretation")
    st.info(chosen_case["analysis"])


# -----------------------------------------------------------------------------
# View 5: Research Limitations
# -----------------------------------------------------------------------------

elif section == "⚠️ Research Limitations":
    st.header("Section 5: Research Limitations & Epistemic Boundaries")
    st.markdown(
        """
        To maintain strict academic integrity and avoid overgeneralization, all findings presented in FactIndic
        must be evaluated within the following explicit limitations:
        """
    )

    limitations = [
        ("Small Sample Size (Exploratory Pilot)", "The experiment evaluates exactly 20 frozen summaries across 6 conditions (60 strategy-level evaluations). Each condition comprises 5 English and 5 Hindi summaries. These results represent an exploratory pilot benchmark, not a statistically powered clinical trial."),
        ("No Statistical Significance Claimed", "Because N=5 per language in each condition, percentage point shifts reflect empirical pilot trends rather than proven statistical significance. P-values and confidence intervals are omitted to avoid over-claiming."),
        ("Uncalibrated NLI Model Confidence Indicators", "Softmax probabilities from mDeBERTa-v3 reflect model-internal classification distributions over learned features. They are uncalibrated heuristic scores, not true Bayesian posterior probabilities of real-world factual truth."),
        ("Absence of Human Factuality Gold Standard", "Claims and summaries were evaluated exclusively through automated sequence-classification models. No double-blind expert human fact-checking annotations were collected for this automated run."),
        ("Deterministic Heuristic Boundaries of Claim Extraction", "The rule-based extractor splits on coordinated independent clauses, subordinating conjunctions, and punctuation. While it avoids noun-phrase over-splitting, it does not resolve anaphoric coreference, pronominal bindings, or elliptical omissions."),
        ("BM25 Lexical Retrieval Bottlenecks", "BM25 depends strictly on exact lexical term overlap. When summarization models exhibit code-switching or output English planning text for Hindi articles, lexical retrieval relevance drops substantially."),
        ("Heuristic Aggregation Threshold (0.50)", "The Level-1 decision rule (max_k P >= 0.50) is an engineering design convention, not an empirically validated optimal threshold."),
        ("Strict Isolation of Historical Runs", "All results presented herein were produced using the local mDeBERTa model. Historical hosted Cohere NLI evaluations remain isolated in legacy directories and are not mixed with these local benchmarks."),
    ]

    for title, desc in limitations:
        with st.expander(f"📌 {title}", expanded=True):
            st.markdown(desc)


# -----------------------------------------------------------------------------
# View 6: Dataset & Prompt Explorer
# -----------------------------------------------------------------------------

elif section == "📑 Dataset & Prompt Explorer":
    st.header("Section 6: Live Dataset & Prompt Template Explorer")
    st.markdown(
        """
        Inspect genuine source articles from the Hugging Face repository 
        [`csebuetnlp/xlsum`](https://huggingface.co/datasets/csebuetnlp/xlsum) 
        and review the exact prompt templates sent to generation models.
        """
    )

    col_lang, col_split = st.columns([2, 2])
    with col_lang:
        selected_language = st.selectbox("Select Language:", options=["English", "Hindi"], index=0)
    with col_split:
        st.text_input("Dataset Split", value="test (Standard XL-Sum Evaluation Split)", disabled=True)

    lang_code = "hindi" if selected_language == "Hindi" else "english"
    target_ids = TARGET_RECORD_IDS[lang_code]

    with st.spinner(f"Loading cached records for '{lang_code}'..."):
        records, err = get_cached_live_records(lang_code)

    if err is not None:
        st.error(f"Live Retrieval Blocked: {err}")
    elif records:
        record_options = {
            r.metadata.get("raw_id", r.example_id): f"{r.metadata.get('raw_id')} — {r.metadata.get('title', 'Untitled')[:65]}"
            for r in records
        }
        selected_record_id = st.selectbox("Select Article Record:", options=list(record_options.keys()), format_func=lambda x: record_options.get(x, x))
        record = next((r for r in records if r.metadata.get("raw_id", r.example_id) == selected_record_id), None)

        if record:
            st.markdown("---")
            title = record.metadata.get("title", "Untitled")
            st.subheader(f"Article: {title}")

            col_body, col_ref = st.columns([3, 2])
            with col_body:
                st.markdown("#### Source Text")
                css_class = "devanagari-text" if lang_code == "hindi" else ""
                st.markdown(f'<div class="card-box {css_class}" style="max-height: 380px; overflow-y: auto;">{record.source_text}</div>', unsafe_allow_html=True)
            with col_ref:
                st.markdown("#### Reference Summary")
                st.markdown(f'<div class="card-box {css_class}" style="background-color: #fafaf9; max-height: 380px; overflow-y: auto;">{record.reference_summary}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Dual Prompt Directives (Version 1.0)")
            col_pa, col_pb = st.columns(2)
            with col_pa:
                st.markdown("##### Prompt A (Standard Abstractive)")
                st.markdown(f'<div class="prompt-box">{build_summarization_prompt(record.source_text[:1200], PromptStrategy.PROMPT_A, lang_code)}</div>', unsafe_allow_html=True)
            with col_pb:
                st.markdown("##### Prompt B (Factuality-Aware Constrained)")
                st.markdown(f'<div class="prompt-box">{build_summarization_prompt(record.source_text[:1200], PromptStrategy.PROMPT_B, lang_code)}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# View 7: Provenance & Audit Logs
# -----------------------------------------------------------------------------

elif section == "📜 Provenance & Audit Logs":
    st.header("Section 7: Verifiable Audit Logs & Execution Provenance")
    st.markdown(
        """
        All research results in FactIndic are backed by verifiable on-disk audit logs, checksums, and automated tests.
        """
    )

    audits = get_project_audits()
    controlled_audit = exp_audit

    # Audit Card 1: Controlled Benchmark Experiment Audit
    with st.expander("Audit 1: Controlled Experiment Execution Audit (`experiment_audit.json`)", expanded=True):
        if controlled_audit:
            st.markdown(
                f"""
                - **Experiment Name:** `{controlled_audit.get('experiment_name')}`
                - **NLI Model ID:** `{controlled_audit.get('nli_model_id')}`
                - **Pinned Model Revision:** `{controlled_audit.get('nli_model_revision')}`
                - **PyTorch Version:** `{controlled_audit.get('dependencies', {}).get('torch')}` | **Transformers:** `{controlled_audit.get('dependencies', {}).get('transformers')}`
                - **Total Evaluations:** `{controlled_audit.get('results_summary', {}).get('total_evaluations')} / {controlled_audit.get('results_summary', {}).get('expected_evaluations')}` (100% Success)
                - **Execution Wall-Clock Time:** `{controlled_audit.get('elapsed_seconds')} seconds`
                - **Completed Timestamp (UTC):** `{controlled_audit.get('completed_timestamp_utc')}`
                """
            )
            st.json(controlled_audit)
        else:
            st.warning("Experiment audit file not found.")

    # Audit Card 2: Controlled Generation Audit
    gen_audit = audits.get("controlled_generation")
    with st.expander("Audit 2: Controlled Generation Audit (`controlled_generation_audit.json`)", expanded=False):
        if gen_audit:
            st.json(gen_audit)
        else:
            st.info("Generation audit not available.")

    # Audit Card 3: Live XL-Sum Verification Audit
    live_audit = audits.get("live_xlsum")
    with st.expander("Audit 3: Live XL-Sum Retrieval Audit (`live_xlsum_verification_audit.json`)", expanded=False):
        if live_audit:
            st.json(live_audit)
        else:
            st.info("Live audit not available.")

st.markdown("---")
st.caption("FactIndic Academic Evaluation Platform • Built for Faculty Inspection • Bennett University")
