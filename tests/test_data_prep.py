"""
Unit tests for BanglaGuard data preparation module.
"""

import unittest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from training.data_prep import (
    locate_dataset,
    load_raw_dataset,
    analyze_dataset,
    clean_dataset,
    preprocess_dataset,
    prepare_and_preprocess_pipeline,
)
from training.preprocessor import BengaliTextPreprocessor


class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_data = pd.DataFrame({
            "type": ["spam", "ham", "spam", "SPAM", "ham", None, "ham", "spam"],
            "text": [
                "অফার! পান ৫০০ টাকা ক্যাশব্যাক https://offer.com",
                "কেমন আছেন ভাই?",
                "অফার! পান ৫০০ টাকা ক্যাশব্যাক https://offer.com",  # duplicate
                "লটারি জিতেছেন! কল করুন 01712345678",
                "   ",  # empty
                "কোনো টাইপ নেই",  # missing type
                "কাল দেখা হবে।",
                "প্রতি রেফারেলে ২০০ টাকা জিতুন!",
            ]
        })
        self.mock_csv_path = Path(self.test_dir) / "mock_dataset.csv"
        self.mock_data.to_csv(self.mock_csv_path, index=False, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_locate_dataset(self):
        located = locate_dataset(self.mock_csv_path)
        self.assertEqual(located, self.mock_csv_path)

    def test_load_raw_dataset(self):
        df = load_raw_dataset(self.mock_csv_path)
        self.assertEqual(len(df), 8)
        self.assertIn("type", df.columns)
        self.assertIn("text", df.columns)

    def test_analyze_dataset(self):
        df = load_raw_dataset(self.mock_csv_path)
        stats = analyze_dataset(df)
        self.assertEqual(stats["total_records"], 8)
        self.assertEqual(stats["missing_type"], 1)
        self.assertGreater(stats["duplicate_texts"], 0)

    def test_clean_dataset(self):
        df = load_raw_dataset(self.mock_csv_path)
        cleaned_df, metrics = clean_dataset(df)

        # Missing & empty removed: 2 rows (None type and empty text)
        # Duplicate removed: 1 row
        self.assertEqual(metrics["missing_removed"], 2)
        self.assertEqual(metrics["duplicates_removed"], 1)
        self.assertEqual(len(cleaned_df), 5)

        # Verify binary label mapping
        self.assertIn("label", cleaned_df.columns)
        self.assertTrue(set(cleaned_df["label"].unique()).issubset({0, 1}))
        self.assertIn("original_text", cleaned_df.columns)

    def test_preprocess_dataset(self):
        df = load_raw_dataset(self.mock_csv_path)
        cleaned_df, _ = clean_dataset(df)
        proc_df = preprocess_dataset(cleaned_df)

        self.assertIn("processed_text", proc_df.columns)
        self.assertIn("original_text", proc_df.columns)
        self.assertIn("type", proc_df.columns)
        self.assertIn("label", proc_df.columns)
        self.assertEqual(len(proc_df), len(cleaned_df))

    def test_end_to_end_pipeline_mock(self):
        out_csv = Path(self.test_dir) / "output" / "processed.csv"
        proc_df, summary = prepare_and_preprocess_pipeline(
            input_path=self.mock_csv_path,
            output_path=out_csv,
        )
        self.assertTrue(out_csv.exists())
        self.assertEqual(summary["original_dataset_size"], 8)
        self.assertEqual(summary["final_dataset_size"], 5)
        self.assertEqual(summary["missing_rows_removed"], 2)
        self.assertEqual(summary["duplicates_removed"], 1)


if __name__ == "__main__":
    unittest.main()
