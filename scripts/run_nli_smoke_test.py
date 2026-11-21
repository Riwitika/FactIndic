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
        print("Model: joeddav/xlm-roberta-large-xnli (560M parameters, ~2.2 GB)")
        print("\nTo download model weights and run live CPU inference, invoke with:")
        print("    python scripts/run_nli_smoke_test.py --live")
        print("\nExiting without downloading weights (safe mode).")
        return 0

    print("=" * 70)
    print("Executing Live Multilingual NLI Verification Smoke Test")
    print("=" * 70)

    config = NLIConfig(
        model_id="joeddav/xlm-roberta-large-xnli",
        device="cpu",
        batch_size=args.batch_size,
        max_length=512,
    )
    verifier = XNLIVerifier(config=config)

    test_cases = [
        {
            "lang": "en",
            "name": "English Entailment Test",
            "premise": "The Reserve Bank of India raised interest rates by 25 basis points on Wednesday amid inflation concerns.",
            "hypothesis": "India's central bank increased interest rates.",
        },
        {
            "lang": "hi",
            "name": "Hindi Entailment Test",
            "premise": "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने चंद्रयान-3 के लैंडर विक्रम को चंद्रमा की सतह पर सफलतापूर्वक उतारा।",
            "hypothesis": "इसरो का चंद्रयान-3 चंद्रमा पर उतरा।",
        },
    ]

    try:
        print("\nLoading tokenizer and model (this may take time on first download)...")
        verifier.load_model()
        print("Model loaded successfully.")

        for tc in test_cases:
            print(f"\n--- {tc['name']} ({tc['lang']}) ---")
            print(f"Premise:    {tc['premise']}")
            print(f"Hypothesis: {tc['hypothesis']}")

            result = verifier.verify_premise_hypothesis(
                premise=tc["premise"],
                hypothesis=tc["hypothesis"],
                language=tc["lang"],
            )

            print(f"Predicted Label:      {result.predicted_label}")
            print(f"Entailment Prob:      {result.entailment_prob:.4f}")
            print(f"Neutral Prob:         {result.neutral_prob:.4f}")
            print(f"Contradiction Prob:   {result.contradiction_prob:.4f}")
            print(f"Inference Latency:    {result.latency_ms:.1f} ms")

        print("\nAll live smoke test cases passed successfully.")
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
