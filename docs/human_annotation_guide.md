# FactIndic: Human Annotation Guidelines & Error Taxonomy

These guidelines define the protocol for human evaluation of factual consistency in generated summaries for English and Hindi, adapting the FRANK taxonomy (Pagnoni et al., 2021).

---

## 1. Annotation Objective
Evaluate each generated summary against its original source article to determine whether all claims in the summary are factually faithful to and supported by the source document.

---

## 2. Unit of Annotation: Atomic Claims
Before labeling, each summary sentence is split into atomic claims (propositions containing a subject, predicate, and associated arguments or modifiers).

Example:
- Summary Sentence: *"ISRO launched Chandrayaan-3 in July 2023 and the rover Pragyan discovered sulfur on the lunar south pole."*
  - **Claim 1:** ISRO launched Chandrayaan-3 in July 2023.
  - **Claim 2:** Rover Pragyan discovered sulfur on the lunar south pole.

---

## 3. Classification Steps

For each claim:
1. **Determine Factuality:**
   - **Supported (Factual):** The source text directly entails or corroborates this claim.
   - **Unsupported / Inconsistent (Non-Factual):** The claim is contradicted by the source or contains details not present in the source.

2. **Categorize Error Type (for Non-Factual claims):**

| Code | Error Class | Description | English Example | Hindi Example |
| :--- | :--- | :--- | :--- | :--- |
| **`PredE`** | **Predicate Error** | The verb/action/relation contradicts or distorts the source. | Source: *"X criticized Y"*; Summary: *"X praised Y"* | स्रोत: *"भारत ने प्रस्ताव का विरोध किया"*; सारांश: *"भारत ने समर्थन किया"* |
| **`EntE`** | **Entity Error** | Swapped participants, wrong subject/object, or misassigned role. | Source: *"Alice defeated Bob"*; Summary: *"Bob defeated Alice"* | स्रोत: *"रोहित ने शतक बनाया"*; सारांश: *"विराट ने शतक बनाया"* |
| **`CircE`** | **Circumstance Error** | Erroneous time, location, numeric quantity, or condition. | Source: *"Launched in 2023"*; Summary: *"Launched in 2021"* | स्रोत: *"रेपो दर 6.5% रही"*; सारांश: *"रेपो दर 7.5% कर दी गई"* |
| **`CorefE`** | **Coreference Error** | Ambiguous, missing, or mismatched pronoun/anaphora. | Pronoun *"He"* refers to the wrong person in context. | सर्वनाम *"उन्होंने"* गलत व्यक्ति की ओर संकेत करता है। |
| **`LinkE`** | **Discourse Link Error** | Faulty causal, temporal, or logical connector between clauses. | Using *"because"* when no causal connection exists. | *"क्योंकि"* या *"इसलिए"* का अनुचित उपयोग जहां कारण संबंध नहीं है। |
| **`OutE`** | **Out-of-Article / Hallucination** | Completely new assertion or entity not found in the source. | Mentioning a third party not present in the article. | स्रोत में अनुपस्थित किसी बाहरी घटना या संगठन का उल्लेख। |
| **`OthE`** | **Other Error** | Factual distortion that does not fit the above categories. | Unclear or idiomatically distorted statements. | अन्य अवर्गीकृत तथ्यात्मक त्रुटियां। |

---

## 4. Hindi-Specific Annotation Notes
- **Case Markers:** Pay close attention to Hindi case markers (`ने`, `को`, `से`, `का`, `की`, `के`). A shift in case marking often changes the semantic agent of an action (`EntE`).
- **Devanagari Numerals & Words:** Check both numeric digits (e.g., `6.5%`, `2023`) and written number words (e.g., `साढ़े छह प्रतिशत`, `दो हज़ार तेईस`) for equivalence (`CircE`).
- **Discourse Markers:** Check conjunctive particles (`किंतु`, `परंतु`, `लेकिन`, `इसलिए`, `फलस्वरूप`) to verify that the logical implication matches the source (`LinkE`).

---

## 5. Adjudication & Agreement
- Each item is annotated independently by two evaluators.
- Agreement is computed via Cohen's $\kappa$ and Fleiss' $\kappa$.
- Disagreements are adjudicated by a third expert annotator to produce the consensus gold standard.
