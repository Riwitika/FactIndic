"""Rule-Based Atomic Factual Claim Extraction for English and Hindi.

Provides deterministic, rule-based decomposition of summary sentences into
standalone, independently verifiable atomic factual claims without external
API calls. Fully compatible with ExtractedClaim, BM25EvidenceRetriever,
and HybridClaimVerifier.

Methodology & Epistemic Boundaries:
    - Extracts atomic claims in their original language (English or Hindi Devanagari)
      without translating or paraphrasing out of context.
    - Preserves exact mappings between every atomic claim and its originating sentence.
    - Deterministically decomposes compound sentences, coordinate structures,
      and non-restrictive relative clauses.
    - Propagates governing subjects across coordinate clauses to prevent headless fragments.
    - Filters out generative reasoning preambles, prompt echoes, and meta-commentary.
    - Does NOT perform NLI verification or evidence retrieval in this phase.
    - Does NOT invent claims, add external world knowledge, or alter original semantics.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from factindic.generation import GenerationRecord
from factindic.verification.claim_extractor import ExtractedClaim
from factindic.verification.nli import SUPPORTED_VERIFICATION_LANGUAGES
from factindic.verification.sentence_verifier import split_into_sentences

# Patterns matching generative reasoning traces and prompt echoes
REASONING_PREFIX_PATTERNS = [
    # Prompt echoes and instructions
    r'(?i)^\s*the\s+user\s+(?:asks|wants|requests)[^:\n]*:?\s*"(?:[^"\\]|\\.)*"',
    r'(?i)^\s*the\s+user\s+(?:asks|wants|requests).*?(?:\.|$)',
    r'(?i)^\s*the\s+task\s*:\s*generate\s+a\s+concise\s+summary.*?(?:\.|$)',
    r'(?i)^\s*task\s*:\s*generate\s+a\s+concise\s+summary.*?(?:\.|$)',
    r'(?i)^\s*you\s+are\s+a\s+professional\s+abstractive\s+summarization\s+assistant.*?(?:\.|$)',
    r'(?i)^\s*generate\s+a\s+concise\s+summary\s+using\s+only\s+information\s+supported.*?(?:\.|$)',
    r'(?i)^\s*preserve\s+important\s+entities,\s+numbers,\s+dates,\s+events.*?(?:\.|$)',
    r'(?i)^\s*do\s+not\s+introduce\s+unsupported\s+details\s+or\s+external\s+knowledge.*?(?:\.|$)',
    r'(?i)^\s*output\s+the\s+summary\s+in\s+hindi.*?(?:\.|$)',
    r'(?i)^\s*if\s+the\s+source\s+is\s+ambiguous,\s+do\s+not\s+invent\s+a\s+resolution.*?(?:\.|$)',
    # Internal model deliberations
    r'(?i)^\s*we\s+need\s+to\s+produce\s+a?\s*concise\s+summary.*?(?:\.|$)',
    r'(?i)^\s*we\s+need\s+to\s+summarize\s+the\s+article.*?(?:\.|$)',
    r'(?i)^\s*we\s+need\s+to\s+extract\s+key\s+points.*?(?:\.|$)',
    r'(?i)^\s*need\s+to\s+capture\s+key\s+points\s*:?',
    r'(?i)^\s*let\s*[\'\u2019]?s\s+(?:craft|extract|summarize|produce|draft).*?(?:\.|$)',
    r'(?i)^\s*key\s+points(?:\s+from\s+(?:the\s+)?article)?\s*:?',
    r'(?i)^\s*potential\s+claims\s*:?',
    r'(?i)^\s*draft\s*:?',
    r'(?i)^\s*refinement\s*:?',
    r'(?i)^\s*final\s+output\s*:?',
    r'(?i)^\s*step\s+\d+\s*:?',
    r'(?i)^\s*summary\s*:?',
    r'(?i)^\s*note\s*:?',
    r'(?i)^\s*proceed\s*\.?',
    r'(?i)^\s*just\s+summary\s*\.?',
]

# Patterns for pure meta sentences that should be filtered out
META_SENTENCE_PATTERNS = [
    r'(?i)^\s*the\s+user\s+(?:asks|wants|requests)',
    r'(?i)^\s*we\s+need\s+to\s+(?:produce|summarize|craft|extract)',
    r'(?i)^\s*need\s+to\s+capture\s+key\s+points',
    r'(?i)^\s*let\s*[\'\u2019]?s\s+(?:craft|extract|summarize|produce|draft)',
    r'(?i)^\s*key\s+points(?:\s+from\s+(?:the\s+)?article)?\s*:',
    r'(?i)^\s*task\s*:',
    r'(?i)^\s*the\s+task\s*:',
    r'(?i)^\s*output\s+(?:the\s+summary\s+)?in\s+hindi',
    r'(?i)^\s*(?:must\s+)?preserve\s+important\s+entities',
    r'(?i)^\s*no\s+unsupported\s+details',
    r'(?i)^\s*summarize\s*\.?\s*$',
    r'(?i)^\s*do\s+not\s+introduce\s+unsupported',
    r'(?i)^\s*if\s+the\s+source\s+is\s+ambiguous',
    r'(?i)^\s*you\s+are\s+a\s+professional',
    r'(?i)^\s*generate\s+a\s+concise\s+summary',
    r'(?i)^\s*proceed\s*\.?\s*$',
    r'(?i)^\s*just\s+summary\s*\.?\s*$',
    r'(?i)^\s*write\s+in\s+hindi',
    r'(?i)^\s*ensure\s+no\s+external\s+knowledge',
    r'(?i)^\s*use\s+only\s+article\s+info',
    r'^\s*[\*\-•]\s*$',
]

# Hindi copular, auxiliary, and common verb root/inflection tokens
HINDI_VERB_WORDS = {
    # Copula and auxiliaries
    "है", "हैं", "था", "थी", "थे", "थीं", "होगा", "होगी", "होंगे", "हो",
    "सकता", "सकती", "सकते", "सके", "सकें", "चाहिए",
    # Aspectual / compound auxiliaries
    "रहा", "रही", "रहे", "गया", "गई", "गए", "हुआ", "हुई", "हुए",
    # Common verbs in perfective or general forms
    "किया", "की", "किए", "कहा", "बताया", "सुनाया", "बनाया", "बनाए",
    "पाया", "पाए", "पाई", "दिया", "दी", "दिए", "लिया", "ली", "लिए",
    "मिला", "मिली", "मिले", "देखा", "सुना", "रखा", "रखे", "रखी",
    "पहुँचा", "पहुँची", "पहुँचे", "आया", "आई", "आए",
    "लगा", "लगे", "लगी", "पड़ा", "पड़े", "पड़ी", "जीता", "जिताया", "हराया",
    "जड़ा", "खोया", "छोड़ा", "छोड़े", "दागा", "दागेगा", "गिरेंगे", "बढ़ा",
    "बढ़ी", "बढ़े", "घटा", "घटी", "घटे", "मरा", "मरे", "मरी", "ठहराया",
    "सुनाया", "बोला", "बोले", "बताते", "शामिल", "होती", "समाप्त", "बिक्री",
}


def tokenize_hindi_words(text: str) -> List[str]:
    """Extract Devanagari word tokens, excluding purna viram and punctuation."""
    return re.findall(r"[\u0900-\u0963\u0966-\u097F]+", text)


def has_hindi_predicate(text: str) -> bool:
    """Check if a Hindi text fragment contains a finite verb or auxiliary predicate."""
    tokens = tokenize_hindi_words(text)
    for t in tokens:
        if t in HINDI_VERB_WORDS:
            return True
        if len(t) >= 3 and (t.endswith("ता") or t.endswith("ती") or t.endswith("ते")):
            return True
    return False


def get_current_clause_before(before_text: str) -> str:
    """Isolate the immediate clause preceding the conjunction (since last delimiter)."""
    delims = re.split(r"[,;—\-]|(?:\s|^)(?:जिसमें|जिसने|जिसे|जो|कि|जहाँ)(?:\s|$)", before_text)
    return delims[-1].strip()


def get_current_clause_after(after_text: str) -> str:
    """Isolate the immediate clause following the conjunction (up to next delimiter)."""
    delims = re.split(r"[,;—\-]|(?:\s|^)(?:जिसमें|जिसने|जिसे|जो|कि|जहाँ)(?:\s|$)|[।?!]", after_text)
    return delims[0].strip()


def is_clausal_hindi_conjunction(before_text: str, after_text: str) -> bool:
    """Determine whether a Hindi conjunction coordinates independent clauses rather than noun lists.

    Returns True ONLY if both the clause immediately preceding and the clause immediately
    following the conjunction contain an independent predicate.
    """
    cb = get_current_clause_before(before_text)
    ca = get_current_clause_after(after_text)
    return has_hindi_predicate(cb) and has_hindi_predicate(ca)

# English coordinating conjunctions
ENGLISH_COORDINATING_SPLITS = [
    r',\s+and\s+',
    r',\s+but\s+',
    r',\s+while\s+',
    r',\s+whereas\s+',
    r';\s+',
]


def clean_preamble(text: str, language: str = "en") -> str:
    """Strip code fences, special tokens, and generative reasoning preambles from summary text.

    Args:
        text: Raw generated summary text.
        language: Language code ('en' or 'hi').

    Returns:
        Cleaned text stripped of meta-tokens and instructions.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strip special end-of-sequence / beginning-of-sequence tokens
    cleaned = re.sub(r"<[A-Z0-9_]+>", " ", cleaned)

    # For Hindi: if text contains Devanagari, strip any leading English reasoning block
    if language == "hi":
        devanagari_match = re.search(r"[\u0900-\u097F]", cleaned)
        if devanagari_match:
            dev_start = devanagari_match.start()
            # If significant English preamble precedes Devanagari, drop the preamble
            if dev_start > 20:
                preamble_chunk = cleaned[:dev_start]
                # Check if preamble consists predominantly of English reasoning
                if any(re.search(pat, preamble_chunk) for pat in REASONING_PREFIX_PATTERNS) or "summary" in preamble_chunk.lower():
                    cleaned = cleaned[dev_start:].strip()

    # Clean line-by-line reasoning prefixes
    lines = cleaned.split("\n")
    filtered_lines: List[str] = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        # Check if line matches reasoning prefix patterns
        is_reasoning = False
        for pat in REASONING_PREFIX_PATTERNS:
            if re.search(pat, stripped_line):
                # Remove the matching reasoning prefix
                stripped_line = re.sub(pat, "", stripped_line).strip()
                if not stripped_line:
                    is_reasoning = True
                    break
        if not is_reasoning and stripped_line:
            # Strip bullet markers
            stripped_line = re.sub(r"^[\*\-•]\s*", "", stripped_line).strip()
            if stripped_line:
                filtered_lines.append(stripped_line)

    cleaned = "\n".join(filtered_lines).strip()
    return cleaned


def is_meta_statement(sentence: str) -> bool:
    """Determine whether a sentence is a generative meta-commentary statement.

    Args:
        sentence: Sentence string.

    Returns:
        True if the sentence is purely instructional or meta-commentary, False otherwise.
    """
    s = sentence.strip()
    if not s or len(s) < 10:
        return True

    for pat in META_SENTENCE_PATTERNS:
        if re.search(pat, s):
            return True

    return False


def _extract_hindi_subject(clause: str) -> Optional[str]:
    """Extract the governing subject noun phrase from the beginning of a Hindi clause.

    Looks for noun phrases marked by ergative 'ने' or nominative NP before case postpositions.
    """
    s = clause.strip()
    # Pattern: [NP] ने
    ne_match = re.match(r"^([^,।?!]+?\s+ने)\b", s)
    if ne_match:
        return ne_match.group(1).strip()

    # Pattern: [NP] (before first verb or postposition, max 6 words)
    words = s.split()
    if len(words) >= 2:
        for idx in range(min(len(words), 6)):
            if words[idx] in ("ने", "को", "से", "का", "के", "की", "में", "पर"):
                return " ".join(words[: idx + 1]).strip()
        # Fallback to first 2-3 words if plausible subject NP
        return " ".join(words[: min(3, len(words))]).strip()

    return None


def decompose_hindi_sentence(sentence: str) -> List[str]:
    """Decompose a Hindi sentence into standalone atomic factual claims.

    Handles:
    - Relative clauses (जो ... वह / उसने).
    - Coordinating conjunctions (और, तथा, एवं, लेकिन, आदि) with subject propagation.
    - Subordinating conjunctions (क्योंकि, हालांकि, आदि).
    - Elliptical coordinate clauses.

    Args:
        sentence: Clean Hindi sentence.

    Returns:
        List of standalone atomic claims in Hindi Devanagari.
    """
    s = sentence.strip()
    if not s:
        return []

    # Clean punctuation
    s = re.sub(r"\s+", " ", s)
    purna_viram = "।"

    # 1. Non-restrictive Relative Clause: X, जो Y, Z
    rel_match = re.search(r"^(.*?),\s*(जो\s+.*?),\s*(.*)$", s)
    if rel_match:
        antecedent = rel_match.group(1).strip()
        rel_clause = rel_match.group(2).strip()
        main_end = rel_match.group(3).strip()

        # Strip 'जो' from relative clause
        rel_body = re.sub(r"^जो\s+", "", rel_clause).strip()
        claims: List[str] = []

        # Claim 1: Main proposition
        main_claim = f"{antecedent} {main_end}".strip()
        if not main_claim.endswith(purna_viram) and not main_claim.endswith("?"):
            main_claim += purna_viram
        claims.append(main_claim)

        # Claim 2: Relative attribute proposition
        rel_claim = f"{antecedent} {rel_body}".strip()
        if not rel_claim.endswith(purna_viram) and not rel_claim.endswith("?"):
            rel_claim += purna_viram
        claims.append(rel_claim)

        return claims

    # 2. Coordinate Conjunction Splitting with Subject Propagation
    # 2. Coordinate Conjunction Splitting with Clausal Boundary Verification
    coord_pattern = re.compile(r"(\s*(?:,\s*)?(?:और|तथा|एवं|लेकिन|किन्तु|परन्तु|मगर|जबकि)\s+)")
    matches = list(coord_pattern.finditer(s))
    clausal_matches = []
    for m in matches:
        b = s[: m.start()]
        a = s[m.end() :]
        if is_clausal_hindi_conjunction(b, a):
            clausal_matches.append(m)

    if clausal_matches:
        clauses: List[str] = []
        last_idx = 0
        for m in clausal_matches:
            c = s[last_idx : m.start()].strip()
            if c:
                clauses.append(c)
            last_idx = m.end()
        fc = s[last_idx:].strip()
        if fc:
            clauses.append(fc)

        governing_subject = _extract_hindi_subject(clauses[0])
        extracted: List[str] = []
        for idx, part in enumerate(clauses):
            clean_part = re.sub(r"^[,;—\s]+", "", part).strip()
            clean_part = re.sub(r"[,;—\s]+$", "", clean_part).strip()
            if not clean_part or len(tokenize_hindi_words(clean_part)) < 2:
                continue

            # If clause 2+ lacks a subject and we have a governing subject, propagate it
            if idx > 0 and governing_subject:
                part_words = tokenize_hindi_words(clean_part)
                first_w = part_words[0] if part_words else ""
                part_subject = _extract_hindi_subject(clean_part)
                if not part_subject or ("ने" not in part_subject and len(part_words) < 8):
                    if first_w not in ("उसका", "उनकी", "उनका", "उनके", "वह", "उसने", "उन्होंने"):
                        clean_part = f"{governing_subject} {clean_part}".strip()

            if not clean_part.endswith(purna_viram) and not clean_part.endswith("?"):
                clean_part += purna_viram

            extracted.append(clean_part)

        if len(extracted) >= 2:
            return extracted

    # 3. Subordinating Conjunction Splitting with Clausal Boundary Verification
    sub_pattern = re.compile(r"(\s*(?:,\s*)?(?:क्योंकि|चूंकि|जिससे|ताकि|हालांकि)\s+)")
    sub_matches = list(sub_pattern.finditer(s))
    clausal_sub = []
    for m in sub_matches:
        b = s[: m.start()]
        a = s[m.end() :]
        if has_hindi_predicate(get_current_clause_before(b)) and has_hindi_predicate(get_current_clause_after(a)):
            clausal_sub.append(m)

    if clausal_sub:
        clauses = []
        last_idx = 0
        for m in clausal_sub:
            c = s[last_idx : m.start()].strip()
            if c:
                clauses.append(c)
            last_idx = m.end()
        fc = s[last_idx:].strip()
        if fc:
            clauses.append(fc)

        extracted = []
        for part in clauses:
            clean_part = re.sub(r"^[,;—\s]+", "", part).strip()
            clean_part = re.sub(r"[,;—\s]+$", "", clean_part).strip()
            if not clean_part or len(tokenize_hindi_words(clean_part)) < 2:
                continue
            if not clean_part.endswith(purna_viram) and not clean_part.endswith("?"):
                clean_part += purna_viram
            extracted.append(clean_part)
        if len(extracted) >= 2:
            return extracted

    # Fallback: Indivisible atomic sentence
    if not s.endswith(purna_viram) and not s.endswith("?"):
        s += purna_viram
    return [s]


def _extract_english_subject(clause: str) -> Optional[str]:
    """Extract a simple governing subject noun phrase from an English clause."""
    s = clause.strip()
    words = s.split()
    if len(words) < 2:
        return None

    # Check for common subject determiners / noun phrases
    if words[0].lower() in ("a", "an", "the", "this", "that", "these", "those"):
        return " ".join(words[: min(3, len(words))]).strip()

    # Proper nouns or short noun phrases
    return " ".join(words[: min(2, len(words))]).strip()


def decompose_english_sentence(sentence: str) -> List[str]:
    """Decompose an English sentence into standalone atomic factual claims.

    Handles:
    - Attribution clauses ("X said that Y", "According to X, Y", "..., police said").
    - Non-restrictive relative clauses ("..., which ...", "..., who ...").
    - Coordinating conjunctions (", and ", ", but ", ", while ", "; ") with subject propagation.

    Args:
        sentence: Clean English sentence.

    Returns:
        List of standalone atomic claims in English.
    """
    s = sentence.strip()
    if not s:
        return []

    s = re.sub(r"\s+", " ", s)

    # 1. Trailing attribution: "..., [Source] have said / said [Source]."
    attrib_trailing = re.search(
        r"^(.*?),\s*([A-Za-z\s]+?\s+(?:have\s+said|said|confirmed|stated|reported))\s*\.?$",
        s,
        re.IGNORECASE,
    )
    if attrib_trailing:
        content = attrib_trailing.group(1).strip()
        source_said = attrib_trailing.group(2).strip()
        claims: List[str] = []

        # Claim 1: The core factual content
        content_claim = content
        if not content_claim.endswith((".", "!", "?")):
            content_claim += "."
        claims.append(content_claim)

        # Claim 2: The attribution statement
        attrib_claim = f"{source_said} that {content[:1].lower() + content[1:]}".strip()
        if not attrib_claim.endswith((".", "!", "?")):
            attrib_claim += "."
        claims.append(attrib_claim)

        return claims

    # 2. Leading attribution: "According to X, Y"
    attrib_leading = re.search(r"^According\s+to\s+([^,]+),\s*(.+)$", s, re.IGNORECASE)
    if attrib_leading:
        source = attrib_leading.group(1).strip()
        content = attrib_leading.group(2).strip()
        claims = []
        if not content.endswith((".", "!", "?")):
            content += "."
        claims.append(content)
        claims.append(f"According to {source}, {content[:1].lower() + content[1:]}")
        return claims

    # 3. Relative Clauses: "X, which Y, Z" or "X, who Y, Z"
    rel_match = re.search(r"^(.*?),\s*((?:which|who|whose)\s+.*?),\s*(.*)$", s, re.IGNORECASE)
    if rel_match:
        antecedent = rel_match.group(1).strip()
        rel_clause = rel_match.group(2).strip()
        main_end = rel_match.group(3).strip()

        claims = []
        # Main claim
        main_claim = f"{antecedent} {main_end}".strip()
        if not main_claim.endswith((".", "!", "?")):
            main_claim += "."
        claims.append(main_claim)

        # Relative claim
        rel_pronoun_match = re.match(r"^(which|who|whose)\s+(.*)$", rel_clause, re.IGNORECASE)
        if rel_pronoun_match:
            rel_body = rel_pronoun_match.group(2).strip()
            rel_claim = f"{antecedent} {rel_body}".strip()
            if not rel_claim.endswith((".", "!", "?")):
                rel_claim += "."
            claims.append(rel_claim)

        if len(claims) >= 2:
            return claims

    # 4. Coordinating Conjunction Splitting with Subject Propagation
    governing_subject = _extract_english_subject(s)

    for conj_pat in ENGLISH_COORDINATING_SPLITS:
        parts = re.split(conj_pat, s, flags=re.IGNORECASE)
        if len(parts) >= 2:
            extracted: List[str] = []
            for idx, part in enumerate(parts):
                clean_part = part.strip()
                if not clean_part or len(clean_part.split()) < 3:
                    continue

                clean_part = re.sub(r"^[,;—\s]+", "", clean_part)
                clean_part = re.sub(r"[,;—\s]+$", "", clean_part)

                # Capitalize first letter
                if clean_part:
                    clean_part = clean_part[0].upper() + clean_part[1:]

                # Check if subject is missing in subsequent clauses
                first_word = clean_part.split()[0].lower() if clean_part.split() else ""
                # If starts with a verb-like or adverb token and we have a governing subject
                if idx > 0 and governing_subject:
                    if first_word in ("confirmed", "added", "noted", "warned", "said", "also", "met", "visited", "took"):
                        clean_part = f"{governing_subject} {clean_part[0].lower() + clean_part[1:]}".strip()

                if not clean_part.endswith((".", "!", "?")):
                    clean_part += "."

                extracted.append(clean_part)

            if len(extracted) >= 2:
                return extracted

    # Fallback: Indivisible sentence
    if not s.endswith((".", "!", "?")):
        s += "."
    return [s]


class RuleBasedClaimExtractor:
    """Deterministic, rule-based atomic factual claim extractor for English and Hindi.

    Extracts standalone atomic claims from summary sentences without external API dependencies.
    Produces standard ExtractedClaim dataclass instances identical to AtomicClaimExtractor.
    """

    def __init__(self, filter_preamble: bool = True) -> None:
        """Initialize RuleBasedClaimExtractor.

        Args:
            filter_preamble: Whether to automatically strip generative reasoning preambles.
        """
        self.filter_preamble = filter_preamble

    def extract_from_sentence(
        self,
        sentence: str,
        record_id: str,
        sentence_index: int,
        language: str,
        prompt_strategy: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> List[ExtractedClaim]:
        """Extract atomic factual claims from an individual summary sentence.

        Args:
            sentence: Individual summary sentence.
            record_id: Originating record identifier.
            sentence_index: 0-indexed position of sentence in summary.
            language: Language code ('en' or 'hi').
            prompt_strategy: Optional prompt condition ('prompt_a' or 'prompt_b').
            source_dataset: Optional source benchmark dataset.

        Returns:
            List of structured ExtractedClaim instances.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        claim_base_id = f"{record_id}_s{sentence_index}"

        # 1. Validation
        if not sentence or not sentence.strip():
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=language,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message="Sentence is empty or whitespace.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        lang_clean = str(language).strip().lower()
        if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=language,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message=f"Unsupported language '{language}'. Supported: en, hi.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        # 2. Preamble cleaning on sentence level
        clean_s = sentence.strip()
        if self.filter_preamble:
            clean_s = clean_preamble(clean_s, language=lang_clean)

        # 3. Check if sentence is purely a meta-instruction / reasoning trace
        if not clean_s or is_meta_statement(clean_s):
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=lang_clean,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message="Sentence is meta-commentary / prompt preamble.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        # 4. Decompose based on language
        if lang_clean == "hi":
            # If the cleaned sentence has Devanagari, use Hindi decomposition
            if re.search(r"[\u0900-\u097F]", clean_s):
                decomposed = decompose_hindi_sentence(clean_s)
            else:
                # Truncated or code-mixed English reasoning inside Hindi summary
                decomposed = decompose_english_sentence(clean_s)
        else:
            decomposed = decompose_english_sentence(clean_s)

        # 5. Build ExtractedClaim instances
        claims: List[ExtractedClaim] = []
        for claim_idx, claim_str in enumerate(decomposed):
            clean_claim = claim_str.strip()
            if not clean_claim:
                continue

            claims.append(
                ExtractedClaim(
                    record_id=record_id,
                    language=lang_clean,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c{claim_idx}",
                    atomic_claim_text=clean_claim,
                    extraction_status="SUCCESS",
                    error_message=None,
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            )

        # Fallback if decomposition produced nothing
        if not claims:
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=lang_clean,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text=clean_s,
                    extraction_status="SUCCESS",
                    error_message=None,
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        return claims

    def extract_from_text(
        self,
        text: str,
        record_id: str,
        language: str,
        prompt_strategy: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> List[ExtractedClaim]:
        """Extract atomic claims from a full summary text.

        Splits text into sentences and applies rule-based claim extraction to each.

        Args:
            text: Full summary text.
            record_id: Originating record identifier.
            language: Language code ('en' or 'hi').
            prompt_strategy: Optional prompt condition.
            source_dataset: Optional source benchmark dataset.

        Returns:
            List of ExtractedClaim instances across all sentences.
        """
        if not text or not text.strip():
            return []

        lang_clean = SUPPORTED_VERIFICATION_LANGUAGES.get(str(language).strip().lower(), "en")
        cleaned_text = clean_preamble(text, language=lang_clean) if self.filter_preamble else text
        sentences = split_into_sentences(cleaned_text, language=lang_clean)

        all_claims: List[ExtractedClaim] = []
        for s_idx, sentence in enumerate(sentences):
            claims = self.extract_from_sentence(
                sentence=sentence,
                record_id=record_id,
                sentence_index=s_idx,
                language=lang_clean,
                prompt_strategy=prompt_strategy,
                source_dataset=source_dataset,
            )
            all_claims.extend(claims)

        return all_claims

    def extract_from_record(self, record: GenerationRecord) -> List[ExtractedClaim]:
        """Extract atomic claims from a GenerationRecord.

        Args:
            record: GenerationRecord with generated summary text.

        Returns:
            List of ExtractedClaim instances.
        """
        rec_id = (
            getattr(record, "source_record_id", None)
            or getattr(record, "record_id", None)
            or "unknown_record"
        )
        return self.extract_from_text(
            text=record.generated_summary,
            record_id=rec_id,
            language=record.language,
            prompt_strategy=record.prompt_strategy,
            source_dataset=record.source_dataset,
        )
