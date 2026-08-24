"""
Command-line script to run the complete BanglaGuard dataset preparation & NLP preprocessing pipeline.

Usage:
    uv run python training/run_prep.py
    uv run python training/run_prep.py --input bangla_dataset.csv --output data/processed/processed_dataset.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path when running script directly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from training.data_prep import prepare_and_preprocess_pipeline


def main():
    # Force UTF-8 encoding for standard output on Windows consoles
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="BanglaGuard Bengali SMS Spam Dataset Preparation & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to raw dataset CSV file (auto-detects if not provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path where processed CSV dataset should be saved",
    )

    args = parser.parse_args()

    print("🚀 Starting BanglaGuard Dataset Preparation & NLP Preprocessing Pipeline...\n")
    try:
        processed_df, summary = prepare_and_preprocess_pipeline(
            input_path=args.input,
            output_path=args.output,
        )
        print(f"✅ Pipeline executed successfully! Ready for NLP model training.")
        return 0
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
