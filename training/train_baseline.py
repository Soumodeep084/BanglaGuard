"""
Training and Evaluation Script for BanglaGuard Baseline Model (TF-IDF + Logistic Regression).

Usage:
    uv run python training/train_baseline.py
    uv run python training/train_baseline.py --dataset data/processed/processed_dataset.csv --test-size 0.20 --models-dir models
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path when script is executed directly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from training.baseline_model import (
    BaselinePredictor,
    train_and_evaluate_pipeline,
)


# Curated sample Bengali SMS messages for testing inference
SAMPLE_SMS_TEST_CASES = [
    {
        "description": "Obvious Promotional/Spam SMS with money and links",
        "text": "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! অবিলম্বে আপনার পুরস্কার দাবি করতে ভিজিট করুন https://gift-prize.bd/claim অথবা কল করুন 01712345678",
        "expected_label": "spam",
    },
    {
        "description": "Telecom Spam Offer with shortcode and discount",
        "text": "মাত্র ১০ টাকায় ১ জিবি ইন্টারনেট মেয়াদ ৩ দিন! অফারটি নিতে ডায়াল করুন *১২১*১# এখনই।",
        "expected_label": "spam",
    },
    {
        "description": "Legitimate Personal / Ham message",
        "text": "দোস্ত, তুই কি কালকে ক্যাম্পাসে আসবি? আমরা লাইব্রেরিতে বসে প্রজেক্টের কাজ শেষ করব।",
        "expected_label": "ham",
    },
    {
        "description": "Legitimate Family Greeting / Ham message",
        "text": "শুভ সকাল মা, আমি বাসায় পৌঁছে গেছি। চিন্তা করো না, ভালো আছি।",
        "expected_label": "ham",
    },
    {
        "description": "Suspicious Bank / Phishing Alert Spam",
        "text": "জরুরি বার্তা: আপনার ব্যাংক অ্যাকাউন্ট সাময়িকভাবে স্থগিত করা হয়েছে। পুনরায় সচল করতে info@secure-login.com ঠিকানায় ইমেইল করুন।",
        "expected_label": "spam",
    },
]


def test_sample_predictions(predictor: BaselinePredictor) -> None:
    """Run and display test predictions on sample Bengali SMS messages."""
    print("\n" + "=" * 65)
    print("🧪 TESTING INFERENCE PREDICTION FUNCTION ON SAMPLE BENGALI SMS")
    print("=" * 65)

    for idx, test_case in enumerate(SAMPLE_SMS_TEST_CASES, 1):
        text = test_case["text"]
        expected = test_case["expected_label"]
        desc = test_case["description"]

        result = predictor.predict(text)

        status_symbol = "✅" if result["label"] == expected else "⚠️"
        print(f"\n[Example {idx}] {desc}")
        print(f"  • Raw Text:       \"{text}\"")
        print(f"  • Preprocessed:   \"{result['processed_text']}\"")
        print(f"  • Prediction:     {status_symbol} {result['label'].upper()} (Confidence: {result['confidence'] * 100:.2f}%)")
        print(f"  • Probabilities:  Ham: {result['ham_probability'] * 100:.2f}%, Spam: {result['spam_probability'] * 100:.2f}%")
        print(f"  • Expected:       {expected.upper()}")

    print("\n" + "=" * 65)


def main():
    # Configure UTF-8 encoding for standard output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="BanglaGuard Baseline NLP Model Training (TF-IDF + Logistic Regression)"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to processed dataset CSV (auto-detected if omitted)",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.20,
        help="Test partition size ratio (default: 0.20 for 80/20 split)",
    )
    parser.add_argument(
        "--models-dir",
        "-m",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--random-state",
        "-s",
        type=int,
        default=42,
        help="Random state seed (default: 42)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Max TF-IDF features (default: 10000)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("🚀 BANGLAGUARD BASELINE MODEL TRAINING (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 65)

    try:
        vectorizer, model, metrics = train_and_evaluate_pipeline(
            dataset_path=args.dataset,
            model_dir=args.models_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
            save_models=True,
        )

        # Test prediction function with saved artifacts
        print("\n🔍 Validating reusable prediction function with saved artifacts...")
        predictor = BaselinePredictor.from_saved(model_dir=args.models_dir)
        test_sample_predictions(predictor)

        print("\n🎉 Baseline NLP pipeline training and evaluation finished successfully!\n")
        return 0

    except Exception as e:
        print(f"\n❌ Baseline training pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
