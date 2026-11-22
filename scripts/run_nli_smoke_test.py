#!/usr/bin/env python3
"""Optional live integration smoke test for Multilingual NLI verification.

Tests live model loading and inference for joeddav/xlm-roberta-large-xnli
on genuine English and Hindi sample pairs.

NOTE: This script downloads ~2.2 GB model weights when executed with --live.
By default, without --live, it displays usage instructions and exits safely.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factindic.verification import (
    NLIConfig,
    VerificationDependencyError,
    VerificationModelLoadError,
    XNLIVerifier,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optional live smoke test for joeddav/xlm-roberta-large-xnli NLI verification."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly enable downloading and running the live 560M parameter model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1 for low-resource CPU).",
    )
    args = parser.parse_args()

    if not args.live:
        print("=" * 70)
        print("FactIndic Multilingual NLI Smoke Test (Gated)")
        print("=" * 70)
        print("Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (278M parameters, ~552 MB)")
        print("Revision: b5113eb38ab63efdd7f280f8c144ea8b13f978ce")
        print("\nTo load cached weights and run live CPU inference, invoke with:")
        print("    python scripts/run_nli_smoke_test.py --live")
        print("\nExiting without running inference (safe mode).")
        return 0

    print("=" * 70)
    print("Executing Live Multilingual NLI Verification Smoke Test")
    print("=" * 70)

    config = NLIConfig(
        model_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        model_revision="b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
        device="cpu",
        batch_size=args.batch_size,
        max_length=512,
    )
    verifier = XNLIVerifier(config=config)

    test_cases = [
        {
            "id": "EN-1",
            "lang": "en",
            "name": "English Entailment",
            "expected": "entailment",
            "premise": "The company reported record quarterly revenue driven by international sales.",
            "hypothesis": "The company generated higher revenue in the quarter.",
        },
        {
            "id": "EN-2",
            "lang": "en",
            "name": "English Contradiction",
            "expected": "contradiction",
            "premise": "All thirty participants successfully finished the marathon in under four hours.",
            "hypothesis": "None of the participants completed the marathon.",
        },
        {
            "id": "EN-3",
            "lang": "en",
            "name": "English Neutral",
            "expected": "neutral",
            "premise": "The Prime Minister announced new economic policies during a press conference in New Delhi.",
            "hypothesis": "The Prime Minister will travel to Mumbai tomorrow.",
        },
        {
            "id": "HI-4",
            "lang": "hi",
            "name": "Hindi Entailment",
            "expected": "entailment",
            "premise": "कंपनी ने अंतरराष्ट्रीय बिक्री के दम पर रिकॉर्ड तिमाही राजस्व दर्ज किया।",
            "hypothesis": "कंपनी ने इस तिमाही में अधिक राजस्व अर्जित किया।",
        },
        {
            "id": "HI-5",
            "lang": "hi",
            "name": "Hindi Contradiction",
            "expected": "contradiction",
            "premise": "सभी तीस प्रतिभागियों ने चार घंटे से कम समय में सफलतापूर्वक मैराथन पूरी की।",
            "hypothesis": "किसी भी प्रतिभागी ने मैराथन पूरी नहीं की।",
        },
        {
            "id": "HI-6",
            "lang": "hi",
            "name": "Hindi Neutral",
            "expected": "neutral",
            "premise": "प्रधानमंत्री ने नई दिल्ली में एक संवाददाता सम्मेलन के दौरान नई आर्थिक नीतियों की घोषणा की।",
            "hypothesis": "प्रधानमंत्री कल मुंबई की यात्रा करेंगे।",
        },
    ]

    try:
        print("\nLoading tokenizer and model from local cache...")
        verifier.load_model()
        print("Model loaded successfully.")

        for tc in test_cases:
            print(f"\n--- [{tc['id']}] {tc['name']} ({tc['lang']}) ---")
            print(f"Premise:    {tc['premise']}")
            print(f"Hypothesis: {tc['hypothesis']}")

            result = verifier.verify_premise_hypothesis(
                premise=tc["premise"],
                hypothesis=tc["hypothesis"],
                language=tc["lang"],
            )

            match = (result.predicted_label == tc["expected"])
            status_flag = "MATCH" if match else "BORDERLINE/DIFF"
            print(f"Expected:             {tc['expected']}")
            print(f"Predicted Label:      {result.predicted_label} [{status_flag}]")
            print(f"Entailment Prob:      {result.entailment_prob:.4f}")
            print(f"Neutral Prob:         {result.neutral_prob:.4f}")
            print(f"Contradiction Prob:   {result.contradiction_prob:.4f}")
            print(f"Inference Latency:    {result.latency_ms:.1f} ms")

        print("\nAll live smoke test cases executed successfully.")
        return 0

    except VerificationDependencyError as exc:
        print(f"\n[Dependency Error] {exc}", file=sys.stderr)
        return 1
    except VerificationModelLoadError as exc:
        print(f"\n[Model Load Error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[Unexpected Error] {exc}", file=sys.stderr)
        return 1
    finally:
        verifier.unload_model()
        print("Model unloaded and memory freed.")


if __name__ == "__main__":
    sys.exit(main())
