"""FactIndic — Faculty Demonstration Dashboard.

A clean, minimal, research-grade Streamlit application for university faculty evaluation.
Allows interactive inspection of the 10 benchmark source articles, their Prompt A and Prompt B
generated summaries, and the corresponding evaluation outputs across all three verification strategies.
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

from factindic.ui_helpers import (
    get_article_evaluations,
    get_hybrid_claim_explorer_detail,
    load_benchmark_articles,
    load_controlled_experiment_results,
    load_controlled_experiment_summary_table,
)

# -----------------------------------------------------------------------------
# Page Configuration & Clean Academic Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="FactIndic — Research Demonstration",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, light academic typography and styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    .devanagari-text {
        font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important;
        line-height: 1.8 !important;
        font-size: 1.05rem;
    }

    .faculty-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }

    .card-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.15rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.85rem;
    }

    .kpi-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem;
        background-color: #f8fafc;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-card .kpi-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
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

    .info-callout {
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #1e40af;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Loading (Cached Offline)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_cached_dashboard_data():
    """Load and cache benchmark articles and evaluation outputs offline."""
    articles = load_benchmark_articles()
    summary_table = load_controlled_experiment_summary_table()
    results = load_controlled_experiment_results()
    return articles, summary_table, results


articles, summary_table, all_results = get_cached_dashboard_data()


# -----------------------------------------------------------------------------
# Simplified Faculty Navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### FactIndic")
    st.caption("Factuality-Aware Summarization")
    st.markdown("---")

    selected_page = st.radio(
        "Demonstration Section:",
        options=["Article Demonstration", "Research Results"],
        index=0,
        key="nav_section",
    )


# -----------------------------------------------------------------------------
# Page 1: Article Demonstration (Primary Faculty Interface)
# -----------------------------------------------------------------------------

if selected_page == "Article Demonstration":
    st.markdown('<div class="faculty-header">', unsafe_allow_html=True)
    st.title("Article Demonstration & Verification Explorer")
    st.markdown(
        "Select any of the **10 frozen benchmark articles** to inspect its original source text, "
        "compare its **Prompt A** and **Prompt B** summaries, and examine how all three NLI verification "
        "strategies evaluated factual consistency."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Article Selection Controls
    col_lang_filter, col_art_select = st.columns([1, 3])

    with col_lang_filter:
        lang_filter = st.selectbox(
            "Filter by Language:",
            options=["All (10 Articles)", "English (5)", "Hindi (5)"],
            index=0,
            key="lang_filter",
        )

    filtered_articles = articles
    if "English" in lang_filter:
        filtered_articles = [a for a in articles if a["language"] == "en"]
    elif "Hindi" in lang_filter:
        filtered_articles = [a for a in articles if a["language"] == "hi"]

    art_options = {
        a["article_id"]: f"[{a['language_display']}] {a['article_id']} — {a['title'][:55]}..."
        for a in filtered_articles
    }

    with col_art_select:
        selected_art_id = st.selectbox(
            "Select Benchmark Article:",
            options=list(art_options.keys()),
            format_func=lambda x: art_options.get(x, x),
            key="article_select",
        )

    # Retrieve selected article and its evaluation mapping
    article = next((a for a in articles if a["article_id"] == selected_art_id), articles[0])
    eval_map = get_article_evaluations(article["article_id"], all_results)

    # Section A: Original Source Article Text
    with st.expander(f"Original Source Article: {article['title']} ({article['language_display']})", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"**Article ID:** `{article['article_id']}`")
        with col_m2:
            st.markdown(f"**Language:** `{article['language_display']}`")
        with col_m3:
            st.markdown(f"**Word Count:** `{len(article['source_text'].split())}` words")

        lang_css = "devanagari-text" if article["language"] == "hi" else ""
        st.markdown(
            f'<div class="card-box {lang_css}" style="max-height: 280px; overflow-y: auto;">{article["source_text"]}</div>',
            unsafe_allow_html=True,
        )

        if article.get("reference_summary"):
            st.markdown(f"**XL-Sum Reference Summary (Ground Truth):**")
            st.markdown(
                f'<div class="card-box {lang_css}" style="background-color: #fafaf9; font-size: 0.95rem;">{article["reference_summary"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Section B: Generated Summaries Comparison (Prompt A vs Prompt B)
    st.subheader("Generated Summaries")
    st.markdown("Compare the frozen model outputs generated under standard and factuality-aware prompting directives:")

    pa_summary = eval_map.get("prompt_a", {}).get("whole_document", {}).get("generated_summary", "Summary not available.")
    pb_summary = eval_map.get("prompt_b", {}).get("whole_document", {}).get("generated_summary", "Summary not available.")

    col_pa, col_pb = st.columns(2)
    lang_css = "devanagari-text" if article["language"] == "hi" else ""

    with col_pa:
        st.markdown("#### Prompt A — Standard Abstractive")
        st.caption("Directive: *'Generate a concise summary preserving main information.'*")
        st.markdown(f'<div class="card-box {lang_css}" style="min-height: 140px;">{pa_summary}</div>', unsafe_allow_html=True)
        st.caption(f"Length: {len(pa_summary)} chars • {len(pa_summary.split())} words")

    with col_pb:
        st.markdown("#### Prompt B — Factuality-Aware")
        st.caption("Directive: *'Summarize using only information supported by the article. Do not introduce unsupported details.'*")
        st.markdown(f'<div class="card-box {lang_css}" style="min-height: 140px;">{pb_summary}</div>', unsafe_allow_html=True)
        st.caption(f"Length: {len(pb_summary)} chars • {len(pb_summary.split())} words")

    st.markdown("---")

    # Section C: Verification Strategy Explorer
    st.subheader("Verification Strategy Explorer")
    st.markdown("Inspect and compare how the three NLI verification architectures evaluated this article's summary:")

    prompt_choice = st.radio(
        "Select Summary Prompt to Inspect:",
        options=["Prompt A (Standard)", "Prompt B (Factuality-Aware)"],
        horizontal=True,
        key="prompt_choice_radio",
    )
    chosen_prompt_key = "prompt_a" if "Prompt A" in prompt_choice else "prompt_b"
    active_evals = eval_map.get(chosen_prompt_key, {})

    # Three Strategy Overview Cards
    col_s1, col_s2, col_s3 = st.columns(3)

    # Strategy 1: Whole-Document NLI
    s1_rec = active_evals.get("whole_document")
    with col_s1:
        st.markdown("##### 1. Whole-Document NLI")
        if s1_rec:
            agg1 = s1_rec.get("aggregated_metrics", {})
            lbl1 = agg1.get("predicted_label", "unknown").lower()
            badge_class1 = (
                "badge-entailment" if lbl1 == "entailment"
                else ("badge-contradiction" if lbl1 == "contradiction" else "badge-neutral")
            )
            st.markdown(
                f"""
                <div class="card-box">
                    <strong>Verdict:</strong> <span class="{badge_class1}">{lbl1.upper()}</span><br><br>
                    <strong>Entailment Score:</strong> {agg1.get('entailment_prob', 0.0):.4f}<br>
                    <strong>Neutral Score:</strong> {agg1.get('neutral_prob', 0.0):.4f}<br>
                    <strong>Contradiction Score:</strong> {agg1.get('contradiction_prob', 0.0):.4f}<br>
                    <small style="color: #64748b;">Latency: {s1_rec.get('verification_latency_ms', 0):.1f} ms</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Whole-document record not available.")

    # Strategy 2: Sentence-Level NLI
    s2_rec = active_evals.get("sentence_level")
    with col_s2:
        st.markdown("##### 2. Sentence-Level NLI")
        if s2_rec:
            agg2 = s2_rec.get("aggregated_metrics", {})
            st.markdown(
                f"""
                <div class="card-box">
                    <strong>Entailment Ratio:</strong> {agg2.get('entailment_ratio', 0.0):.1%} ({agg2.get('entailed_sentences', 0)}/{agg2.get('total_sentences', 0)})<br><br>
                    <strong>Neutral Ratio:</strong> {agg2.get('neutral_ratio', 0.0):.1%} ({agg2.get('neutral_sentences', 0)}/{agg2.get('total_sentences', 0)})<br>
                    <strong>Contradiction Ratio:</strong> {agg2.get('contradiction_ratio', 0.0):.1%} ({agg2.get('contradicted_sentences', 0)}/{agg2.get('total_sentences', 0)})<br>
                    <strong>Mean Sentence Entailment:</strong> {agg2.get('mean_sentence_entailment', 0.0):.4f}<br>
                    <small style="color: #64748b;">Latency: {s2_rec.get('verification_latency_ms', 0):.1f} ms</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Sentence-level record not available.")

    # Strategy 3: Hybrid Claim-Level NLI
    s3_rec = active_evals.get("hybrid_claim_level")
    with col_s3:
        st.markdown("##### 3. Hybrid Claim-Level NLI")
        if s3_rec:
            agg3 = s3_rec.get("aggregated_metrics", {})
            st.markdown(
                f"""
                <div class="card-box">
                    <strong>Entailment Ratio:</strong> {agg3.get('entailment_ratio', 0.0):.1%} ({agg3.get('entailed_claims', 0)}/{agg3.get('total_claims', 0)})<br><br>
                    <strong>Neutral Ratio:</strong> {agg3.get('neutral_ratio', 0.0):.1%} ({agg3.get('neutral_claims', 0)}/{agg3.get('total_claims', 0)})<br>
                    <strong>Contradiction Ratio:</strong> {agg3.get('contradiction_ratio', 0.0):.1%} ({agg3.get('contradicted_claims', 0)}/{agg3.get('total_claims', 0)})<br>
                    <strong>Mean Claim Entailment:</strong> {agg3.get('mean_claim_entailment', 0.0):.4f}<br>
                    <small style="color: #64748b;">Latency: {s3_rec.get('verification_latency_ms', 0):.1f} ms</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Hybrid claim-level record not available.")

    # Detailed Strategy Inspection Section
    st.markdown("---")
    st.subheader("Detailed Result Inspection")

    inspect_tabs = st.tabs([
        "Hybrid Claim-by-Claim & BM25 Evidence",
        "Sentence-by-Sentence Breakdown",
        "Whole-Document Cross-Encoder Output",
    ])

    # Tab 1: Hybrid Claim-by-Claim & BM25 Evidence Inspection
    with inspect_tabs[0]:
        st.markdown("#### Atomic Factual Claims & Retrieved Passages")
        with st.expander("Methodology: Hybrid Claim-Level Verification", expanded=False):
            st.markdown(
                r"""
                - **Decision Policy:** Each atomic claim is evaluated against its top-3 retrieved BM25 passages. A claim is marked **Entailment** if $\max_k P(\text{entailment}) \ge 0.50$; otherwise **Contradiction** if $\max_k P(\text{contradiction}) \ge 0.50$; otherwise **Neutral**.
                - **Score Independence:** Probabilities from different passages are evaluated independently and never combined into a single distribution.
                """
            )

        if s3_rec:
            detail3 = get_hybrid_claim_explorer_detail(s3_rec)
            claims = detail3.get("claims", [])

            if not claims:
                st.info("No atomic claims extracted for this summary.")
            else:
                st.markdown(f"**Extracted Claims ({len(claims)} total):**")
                for c_idx, cl in enumerate(claims, 1):
                    c_lbl = cl["predicted_label"]
                    c_badge = (
                        "badge-entailment" if c_lbl == "entailment"
                        else ("badge-contradiction" if c_lbl == "contradiction" else "badge-neutral")
                    )

                    with st.expander(f"Claim {c_idx}: {cl['atomic_claim_text'][:75]}... [{c_lbl.upper()}]", expanded=(c_idx <= 2)):
                        st.markdown(f"**Claim Text:** *{cl['atomic_claim_text']}*")

                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.markdown(f"**Assigned Verdict:** <span class=\"{c_badge}\">{c_lbl.upper()}</span>", unsafe_allow_html=True)
                        with col_c2:
                            st.markdown(f"**Max Entailment Prob:** `{cl['max_entailment_prob']:.4f}`")
                        with col_c3:
                            st.markdown(f"**Max Contradiction Prob:** `{cl['max_contradiction_prob']:.4f}`")

                        st.markdown("##### Retrieved BM25 Evidence Passages ($K=3$):")
                        passages = cl.get("passages", [])
                        if not passages:
                            st.info("No evidence passages retrieved.")
                        else:
                            for p in passages:
                                p_lbl = p.get("predicted_label", "neutral")
                                p_badge = (
                                    "badge-entailment" if p_lbl == "entailment"
                                    else ("badge-contradiction" if p_lbl == "contradiction" else "badge-neutral")
                                )
                                st.markdown(
                                    f"""
                                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;">
                                        <strong>Passage Rank {p.get('evidence_rank')}</strong> | 
                                        BM25 Score: <code>{p.get('evidence_score', 0.0):.2f}</code> | 
                                        NLI Output: <span class="{p_badge}">{p_lbl.upper()}</span><br>
                                        <div style="margin-top: 0.35rem; font-style: italic; color: #334155;">"{p.get('evidence_passage', '')}"</div>
                                        <div style="margin-top: 0.35rem; font-size: 0.82rem; color: #64748b;">
                                            P(Entailment): <strong>{p.get('entailment_prob', 0.0):.4f}</strong> | 
                                            P(Neutral): <strong>{p.get('neutral_prob', 0.0):.4f}</strong> | 
                                            P(Contradiction): <strong>{p.get('contradiction_prob', 0.0):.4f}</strong>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
        else:
            st.info("Hybrid claim-level verification record not available.")

    # Tab 2: Sentence-by-Sentence Breakdown
    with inspect_tabs[1]:
        st.markdown("#### Sentence-Level NLI Verification")
        st.markdown("Each summary sentence is evaluated independently against the entire source article premise:")

        if s2_rec:
            agg2 = s2_rec.get("aggregated_metrics", {})
            sents_detail = agg2.get("sentences_detail", [])

            if not sents_detail:
                # If detail not in agg, check raw verification results
                sents_detail = s2_rec.get("verification_results", [])

            if not sents_detail:
                st.info("No sentence-level details recorded.")
            else:
                for s_item in sents_detail:
                    s_idx = s_item.get("sentence_index", 0)
                    s_text = s_item.get("sentence_text", "")
                    s_lbl = s_item.get("predicted_label", "neutral")
                    s_badge = (
                        "badge-entailment" if s_lbl == "entailment"
                        else ("badge-contradiction" if s_lbl == "contradiction" else "badge-neutral")
                    )

                    st.markdown(
                        f"""
                        <div class="card-box" style="margin-bottom: 0.6rem;">
                            <strong>Sentence {s_idx + 1}:</strong> {s_text}<br>
                            <div style="margin-top: 0.4rem;">
                                <strong>NLI Verdict:</strong> <span class="{s_badge}">{s_lbl.upper()}</span> &nbsp;|&nbsp;
                                P(Entailment): <code>{s_item.get('entailment_prob', 0.0):.4f}</code> &nbsp;|&nbsp;
                                P(Neutral): <code>{s_item.get('neutral_prob', 0.0):.4f}</code> &nbsp;|&nbsp;
                                P(Contradiction): <code>{s_item.get('contradiction_prob', 0.0):.4f}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Sentence-level record not available.")

    # Tab 3: Whole-Document Cross-Encoder Output
    with inspect_tabs[2]:
        st.markdown("#### Whole-Document Cross-Encoder Verification")
        st.markdown(
            "The complete summary paragraph is evaluated directly against the full article premise in a single cross-encoder pass:"
        )

        if s1_rec:
            agg1 = s1_rec.get("aggregated_metrics", {})
            lbl1 = agg1.get("predicted_label", "neutral")
            badge1 = (
                "badge-entailment" if lbl1 == "entailment"
                else ("badge-contradiction" if lbl1 == "contradiction" else "badge-neutral")
            )

            col_w1, col_w2 = st.columns([1, 2])
            with col_w1:
                st.markdown(
                    f"""
                    <div class="card-box">
                        <h4>Single Document Verdict</h4>
                        <div style="margin-top: 0.5rem; margin-bottom: 0.75rem;">
                            <span class="{badge1}" style="font-size: 1rem; padding: 0.35rem 0.8rem;">{lbl1.upper()}</span>
                        </div>
                        <strong>P(Entailment):</strong> <code>{agg1.get('entailment_prob', 0.0):.4f}</code><br>
                        <strong>P(Neutral):</strong> <code>{agg1.get('neutral_prob', 0.0):.4f}</code><br>
                        <strong>P(Contradiction):</strong> <code>{agg1.get('contradiction_prob', 0.0):.4f}</code><br>
                        <strong>Inference Latency:</strong> <code>{s1_rec.get('verification_latency_ms', 0):.1f} ms</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_w2:
                with st.expander("Observation: Whole-Document Neutral Collapse", expanded=False):
                    st.markdown(
                        "In whole-document verification, cross-attention over multi-thousand-token articles leads the model "
                        "to default to **Neutral** in 80% of benchmark cases (90% in Prompt A, 70% in Prompt B) "
                        "due to background context not covered in the summary."
                    )
        else:
            st.info("Whole-document record not available.")


# -----------------------------------------------------------------------------
# Page 2: Research Results (Macro Empirical Findings)
# -----------------------------------------------------------------------------

elif selected_page == "Research Results":
    st.markdown('<div class="faculty-header">', unsafe_allow_html=True)
    st.title("Research Results: Controlled Benchmark Evaluation")
    st.markdown(
        "Evaluating prompt-based hallucination suppression and verification granularity across English and Hindi summaries.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Key Research Finding — Best Observed Combination
    # -------------------------------------------------------------------------
    st.markdown("### Key Research Finding — Best Observed Combination")

    col_c4, col_c6 = st.columns(2)

    with col_c4:
        st.markdown(
            """
            <div style="border: 1px solid #3b82f6; border-radius: 6px; padding: 1.15rem; background-color: #f8fafc; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #1e3a8a;">C4: Prompt B + Sentence-Level NLI</span>
                    <span style="background-color: #dbeafe; color: #1e40af; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 4px;">BEST BALANCE</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin: 0.75rem 0;">
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Entailment Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #166534;">61.38%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Contradiction Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #991b1b;">5.88%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Neutral Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #334155;">32.75%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Mean Latency</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">16,392.44 ms</div>
                    </div>
                </div>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.45; margin-bottom: 0.35rem;">
                    Among the six tested combinations, Prompt B with sentence-level NLI achieved the highest macro entailment ratio, with a lower contradiction ratio than its Prompt A counterpart.
                </p>
                <p style="font-size: 0.82rem; color: #1e40af; font-weight: 600; margin: 0;">
                    Distinction: Highest observed entailment ratio and a better overall balance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c6:
        st.markdown(
            """
            <div style="border: 1px solid #64748b; border-radius: 6px; padding: 1.15rem; background-color: #f8fafc; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">C6: Prompt B + Hybrid Claim-Level NLI</span>
                    <span style="background-color: #e2e8f0; color: #334155; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 4px;">LOWEST CONTRADICTION</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin: 0.75rem 0;">
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Entailment Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #166534;">42.64%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Contradiction Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #991b1b;">4.10%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Macro Neutral Ratio</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #334155;">53.27%</div>
                    </div>
                    <div style="background: #ffffff; padding: 0.5rem 0.75rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                        <span style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Mean Latency</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">3,645.25 ms</div>
                    </div>
                </div>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.45; margin-bottom: 0.35rem;">
                    Hybrid claim-level verification with Prompt B produced the lowest macro contradiction ratio, but also a substantially higher neutral ratio.
                </p>
                <p style="font-size: 0.82rem; color: #334155; font-weight: 600; margin: 0;">
                    Distinction: Lowest observed contradiction ratio, with more neutral predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "**Note on Pilot Scope:** Findings are from a small exploratory pilot with five summaries per language in each condition "
        "(pooled N=10 per condition, N=20 summaries total) and no human-annotated factuality gold standard. "
        "Results do not claim statistical significance or universal superiority."
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section 1: Conditions C1–C6 Visual Matrix
    # -------------------------------------------------------------------------
    st.subheader("1. Experimental Conditions Comparison (C1–C6)")
    st.markdown(
        "Macro label distribution across all six experimental conditions (N=10 summaries per condition). "
        "Key conditions: **C4** (Highest Entailment: 61.38%) and **C6** (Lowest Contradiction: 4.10%)."
    )

    df_summary = pd.DataFrame(summary_table)
    df_all = df_summary[df_summary["language"] == "all"].copy()

    condition_labels = {
        "C1": "C1 — Prompt A / Whole-document",
        "C2": "C2 — Prompt B / Whole-document",
        "C3": "C3 — Prompt A / Sentence-level",
        "C4": "C4 — Prompt B / Sentence-level",
        "C5": "C5 — Prompt A / Hybrid",
        "C6": "C6 — Prompt B / Hybrid",
    }

    if not df_all.empty:
        chart_data = []
        for _, row in df_all.iterrows():
            cid = condition_labels.get(row["condition_id"], row["condition_id"])
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
            title="Macro Label Distribution Across Conditions C1–C6 (Highlighting C4: Best Balance & C6: Lowest Contradiction)",
            height=380,
        )
        fig_ratios.update_layout(
            margin=dict(l=10, r=10, t=40, b=60),
            yaxis=dict(range=[0, 1.05], tickformat=".1%"),
            xaxis=dict(tickangle=-15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ratios, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 2: Prompt A vs. Prompt B Comparison
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Prompt A vs. Prompt B: Factuality Suppression")
    st.markdown(
        "Comparing standard (Prompt A) and factuality-aware (Prompt B) directives across fine-grained verification strategies:"
    )

    col_cp1, col_cp2 = st.columns([3, 2])

    with col_cp1:
        contra_records = [
            {"Verification Strategy": "Sentence-Level", "Language": "English", "Prompt A": 0.0654, "Prompt B": 0.0625, "Difference": -0.0029, "Relative Change": "-4.4%"},
            {"Verification Strategy": "Sentence-Level", "Language": "Hindi", "Prompt A": 0.1022, "Prompt B": 0.0550, "Difference": -0.0472, "Relative Change": "-46.2%"},
            {"Verification Strategy": "Sentence-Level", "Language": "Combined (Overall)", "Prompt A": 0.0838, "Prompt B": 0.0588, "Difference": -0.0250, "Relative Change": "-29.8%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Language": "English", "Prompt A": 0.0154, "Prompt B": 0.0533, "Difference": +0.0379, "Relative Change": "+246.1%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Language": "Hindi", "Prompt A": 0.1067, "Prompt B": 0.0286, "Difference": -0.0781, "Relative Change": "-73.2%"},
            {"Verification Strategy": "Hybrid Claim-Level", "Language": "Combined (Overall)", "Prompt A": 0.0610, "Prompt B": 0.0410, "Difference": -0.0200, "Relative Change": "-32.8%"},
        ]
        df_cp = pd.DataFrame(contra_records)
        st.dataframe(
            df_cp.style.format({
                "Prompt A": "{:.2%}",
                "Prompt B": "{:.2%}",
                "Difference": "{:+.2%}",
            }),
            use_container_width=True,
        )

    with col_cp2:
        st.markdown(
            """
            <div class="info-callout">
                <strong>Subgroup Findings:</strong>
                <ul>
                    <li><strong>Overall Contradiction Suppression:</strong> Prompt B reduced macro contradiction ratios by <strong>-29.8% relative</strong> in sentence-level verification and <strong>-32.8% relative</strong> in hybrid claim-level verification.</li>
                    <li><strong>Cross-Lingual Variation:</strong> Contradiction reduction was most pronounced in Hindi (plunging by <strong>-73.2% relative</strong> in hybrid claims). English hybrid verification showed a slight rise (<strong>1.54% to 5.33%</strong>, reflecting 2 contradicted claims vs. 1).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Section 3: Verification Architecture Latency Profile
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Verification Architecture Throughput Profile")
    st.markdown("Mean CPU execution latency per summary across verification strategies:")

    if not df_all.empty:
        fig_lat = px.bar(
            df_all,
            x="condition_id",
            y="mean_latency_ms",
            color="verification_strategy",
            color_discrete_map={
                "whole_document": "#3b82f6",
                "sentence_level": "#f97316",
                "hybrid_claim_level": "#8b5cf6",
            },
            labels={"condition_id": "Condition", "mean_latency_ms": "Mean Latency (ms)", "verification_strategy": "Strategy"},
            title="Mean Verification Latency per Summary (CPU)",
            height=300,
        )
        fig_lat.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_lat, use_container_width=True)

    # -------------------------------------------------------------------------
    # Section 4: Comprehensive Experimental Table (Expandable)
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("Complete Experimental Results Table (All 18 Rows)", expanded=False):
        st.markdown("Full results across all six conditions (C1–C6) broken down by Language (`all`, `en`, `hi`):")
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
        df_table_display = df_summary[display_cols].copy()
        df_table_display.columns = [
            "Condition",
            "Prompt",
            "Strategy",
            "Language",
            "N",
            "Mean Entail Score",
            "Entail Ratio",
            "Contra Ratio",
            "Neutral Ratio",
            "Mean Latency (ms)",
        ]
        st.dataframe(
            df_table_display.style.format({
                "Mean Entail Score": "{:.4f}",
                "Entail Ratio": "{:.2%}",
                "Contra Ratio": "{:.2%}",
                "Neutral Ratio": "{:.2%}",
                "Mean Latency (ms)": "{:,.1f}",
            }),
            use_container_width=True,
        )

st.markdown("---")
st.caption("FactIndic Academic Evaluation Platform • Bennett University • Faculty Demonstration Interface")
