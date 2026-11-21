# FactIndic: Research Scope, Feasibility Analysis, and Evaluation Methodology

## 1. Project Title and Purpose
**FactIndic: Factuality-Aware Summarization for Indic and Code-Mixed Languages**

Large Language Models (LLMs) frequently generate fluent but unfaithful or hallucinated content when summarizing complex texts. This failure mode is particularly acute in Indic languages such as Hindi and multilingual settings, where training resources are scarcer and cross-lingual alignment can introduce distortion of entities, numbers, and relations.

The goal of FactIndic is to investigate whether factuality-aware prompting combined with fine-grained Natural Language Inference (NLI) verification can systematically mitigate factual inconsistencies in abstractive summarization across English and Hindi.

---

## 2. Experimental Design ($2 \times 3$ Factorial Setup)

The core research experiment evaluates six distinct configurations ($2 \times 3$) across two language benchmarks (English and Hindi), reported independently:

```
                      +------------------------------------+
                      |    Prompting Strategy (2 levels)   |
                      |  Prompt A (Std) | Prompt B (Fact)  |
+---------------------+-----------------+------------------+
| NLI 1 (Whole-Doc)   |     Cfg A-1     |     Cfg B-1      |
| NLI 2 (Claim-Level) |     Cfg A-2     |     Cfg B-2      |
| NLI 3 (Hybrid)      |     Cfg A-3     |     Cfg B-3      |
+---------------------+-----------------+------------------+
```

### 2.1 Summarization Prompting Strategies
1. **Prompt A (Standard Summarization):**
   A standard task prompt instructing the generation model to produce a concise, fluent summary of the source document without explicit faithfulness constraints.
2. **Prompt B (Factuality-Aware Summarization):**
   A constrained, factuality-focused prompt explicitly instructing the model to:
   - Ground all statements strictly in the provided source article.
   - Preserve exact named entities, numeric quantities, calendar dates, temporal markers, and core relational structures.
   - Avoid extrinsic assertions or speculation not directly supported by the source text.

### 2.2 Factuality-Verification Approaches
1. **Approach 1: Whole-Document NLI:**
   Directly treats the entire source article as the premise and the complete generated summary as the hypothesis, obtaining a document-level entailment probability using a cross-lingual NLI model (e.g., `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`).
2. **Approach 2: Claim-Level NLI:**
   Decomposes the generated summary into atomic propositions/claims. For each claim, retrieves the top-$k$ most semantically relevant source sentences as localized evidence, and computes claim-level entailment, contradiction, and neutral probabilities.
3. **Approach 3: Hybrid Verification:**
   Combines claim-level NLI with targeted, rule-guided factual checks for high-risk factual elements:
   - Named entity preservation (cross-checking named entities extracted via NER).
   - Quantity/number consistency (extracting numeric tokens and verifying identical quantity attribution).
   - Date and calendar temporal consistency.
   - Negation marker detection (flagging polarity flips between source and hypothesis).
   - Relational triple comparison.

### 2.3 Language Support & Separation
- **Language Configurations:** English (`"en"`) and Hindi (`"hi"`).
- **Separation:** All evaluation metrics, error distributions, and qualitative analyses are computed and reported separately for English and Hindi.
- **Cross-Lingual Considerations:** While modern multilingual models support both languages, Hindi features Subject-Object-Verb (SOV) syntax, complex postpositional case marking, and split-ergative alignment. NLI confidence scores are not assumed to be directly comparable across languages without separate calibration.

---

## 3. Dataset Verification: XL-Sum

### 3.1 Dataset Overview
- **Identifier:** `csebuetnlp/xlsum` (Hugging Face Datasets)
- **Paper:** Hasan et al., *"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"*, Findings of ACL 2021.
- **Source:** British Broadcasting Corporation (BBC) articles and journalist-written summaries.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Splits:** `train`, `validation`, `test`.
- **Raw Schema:**
  - `id` (string): Unique BBC article identifier.
  - `url` (string): Canonical URL of the source article.
  - `title` (string): Article headline.
  - `summary` (string): Ground-truth human reference summary.
  - `text` (string): Full body text of the article.

### 3.2 Standardization in FactIndic
Each XL-Sum record is mapped to the standard FactIndic schema:
- `example_id`: `f"xlsum_{lang}_{id}"`
- `language`: `"en"` or `"hi"`
- `source_text`: Cleaned, NFC-normalized article body
- `reference_summary`: Cleaned, NFC-normalized reference summary
- `metadata`: `{"title": ..., "url": ..., "split": ..., "source_dataset": "csebuetnlp/xlsum"}`

---

## 4. MUTANT Feasibility Analysis & Recommendation

### 4.1 Dataset Profile
- **Title:** *MUTANT: A Multi-sentential Code-mixed Hinglish Dataset*
- **Authors:** Rahul Gupta, Vivek Srivastava, Mayank Singh (IIT Gandhinagar & TCS Research; Findings of EACL 2023).
- **Repository:** Hugging Face `LingoIITGN/MUTANT`.
- **Content:** $\sim 67,000$ multilingual articles containing over $85,000$ Multi-sentential Code-mixed Text (MCT) passages sourced from government speeches (*Mann Ki Baat*) and Hindi-English newspaper articles (*Dainik Jagran*).

### 4.2 Feasibility Findings
1. **Absence of Summaries:** MUTANT was created for token-level language identification and multi-sentential code-mixed text extraction. It contains **no reference summaries**.
2. **Absence of Factuality Annotations:** The dataset lacks any factual consistency labels, entailment pairs, or hallucination tags.
3. **High Human Annotation Burden:** To evaluate summarization against reference summaries or to establish a ground-truth factuality benchmark, researchers would need to manually author summaries and error tags from scratch.
4. **Experimental Confounding:** Evaluating MUTANT alongside XL-Sum would confound the research question: any performance discrepancy could stem from differences in text genre, journalistic quality, or ad-hoc summary writing rather than the code-mixing phenomenon itself.

### 4.3 Recommendation
> [!IMPORTANT]
> **Exclude MUTANT from the main experimental pipeline.**
> Retain XL-Sum as the sole benchmark for Phase 1 and subsequent experimental phases. MUTANT is documented as a candidate for future exploratory extensions requiring dedicated reference authoring.

---

## 5. FRANK Benchmark Analysis & Human Annotation Protocol

### 5.1 Overview of FRANK
- **Reference:** Artidoro Pagnoni, Vidhisha Balachandran, Yulia Tsvetkov. *"Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics"*, NAACL-HLT 2021.
- **Repository:** `artidoro/frank` (MIT License).
- **Core Contribution:** A typology of factual errors grounded in frame semantics and linguistic discourse analysis, categorizing inconsistencies into 7 classes:
  1. **Predicate Error (`PredE`):** Mistaken action or relational predicate.
  2. **Entity Error (`EntE`):** Incorrect participants, swapped agent/patient roles, or false attributes.
  3. **Circumstance Error (`CircE`):** Inaccurate spatial, temporal, numeric, or manner modifiers.
  4. **Coreference Error (`CorefE`):** Ambiguous, incorrect, or ungrounded anaphora/pronoun reference.
  5. **Linking Error (`LinkE`):** Erroneous discourse connectives or causal/temporal linkers.
  6. **Out-of-Article Error (`OutE`):** Extrinsic hallucination of entities or assertions absent from the source.
  7. **Other Error (`OthE`):** Unclassified residual factual errors.

### 5.2 Compatibility & Reusability Constraints
- **Direct Label Reuse:** Not possible. FRANK annotations evaluate older model outputs (e.g., BART, BERTSUM) on English CNN/DailyMail and XSum datasets. They do not contain annotations for XL-Sum or newer multilingual LLMs.
- **Taxonomy Adoption:** Highly recommended. The 7-class taxonomy provides a sound linguistic framework for categorizing factual errors in both English and Hindi summaries.
- **Hindi Transfer Considerations:**
  - Hindi case markers (`ने`, `को`, `से`, `का/के/की`) govern grammatical relation assignment; misinterpretation of case markers frequently leads to `EntE` (role swapping).
  - Hindi discourse markers (`लेकिन`, `क्योंकि`, `इसलिए`) map systematically to `LinkE`.
  - Annotators must possess native fluency in Hindi to accurately identify nuanced semantic role violations.

### 5.3 Proposed Human Evaluation Protocol for FactIndic
To validate automated NLI predictions against human ground truth, we establish the following protocol:

1. **Sampling Subset:**
   - Stratified random sample of 50 source articles per language.
   - For each article, evaluate summaries generated by Prompt A and Prompt B ($50 \times 2 = 100$ summaries per language; $200$ total).
2. **Annotation Process:**
   - Two independent native/fluent annotators per language evaluate each summary sentence-by-sentence.
   - Annotators break each summary into atomic claims.
   - For each claim, annotators assign:
     - Binary consistency label: `Factual (Supported)` vs. `Non-Factual (Unsupported/Contradicted)`.
     - Error taxonomy label (for non-factual claims): One of `PredE`, `EntE`, `CircE`, `CorefE`, `LinkE`, `OutE`, `OthE`.
3. **Inter-Annotator Agreement (IAA):**
   - Binary factuality agreement: Measured using Cohen's $\kappa$ (pairwise) and Krippendorff's $\alpha$.
   - Error category agreement: Measured using multi-class Fleiss' $\kappa$.
   - Target benchmark: $\kappa \ge 0.65$ (substantial agreement).
4. **Adjudication:**
   - Any conflicting annotations are submitted to a third expert adjudicator who determines the final consensus gold-standard label.
5. **Validation of Automated Verification:**
   - The human consensus annotations serve as the ground truth to benchmark automated verification metrics (Whole-doc NLI, Claim-level NLI, Hybrid) in terms of Precision, Recall, F1 score, and Pearson/Spearman correlation with human consistency judgments.

---

## 6. Citations & References
1. **XL-Sum:** Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, Rifat Shahriyar. *"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"*, Findings of ACL 2021.
2. **FRANK:** Artidoro Pagnoni, Vidhisha Balachandran, Yulia Tsvetkov. *"Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics"*, Findings of NAACL-HLT 2021.
3. **MUTANT:** Rahul Gupta, Vivek Srivastava, Mayank Singh. *"MUTANT: A Multi-sentential Code-mixed Hinglish Dataset"*, Findings of EACL 2023.
4. **SummaC:** Philippe Laban, Tobias Schnabel, Paul N. Bennett, Marti A. Hearst. *"SummaC: Re-Visiting NLI-based Factuality Evaluation for Abstractive Summarization"*, TACL 2022.
5. **FactCC:** Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher. *"Evaluating the Factual Consistency of Abstractive Text Summarization"*, EMNLP 2020.
