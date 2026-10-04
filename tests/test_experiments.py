"""
Unit tests for BanglaGuard NLP Experiments Module.

Tests:
1. Levenshtein edit distance calculations (insertions, deletions, substitutions).
2. Levenshtein edit distance with Bengali Unicode characters and matras.
3. Normalized similarity calculations.
4. Trigram and character n-gram vectorizer factories.
5. End-to-end evaluation pipeline on synthetic data.
"""

import unittest
import numpy as np
import pandas as pd

from training.nlp_experiments import (
    build_char_ngram_vectorizer,
    build_trigram_vectorizer,
    evaluate_model_pipeline,
    levenshtein_distance,
    levenshtein_similarity,
    run_comparative_experiments,
    run_edit_distance_demonstration,
)
from sklearn.linear_model import LogisticRegression


class TestLevenshteinDistance(unittest.TestCase):
    """Test suite for pure-Python Levenshtein edit distance utilities."""

    def test_identical_strings(self):
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("bKash", "bKash"), 0)
        self.assertEqual(levenshtein_distance("পুরস্কার", "পুরস্কার"), 0)

    def test_empty_string_edge_cases(self):
        self.assertEqual(levenshtein_distance("", "SPAM"), 4)
        self.assertEqual(levenshtein_distance("SPAM", ""), 4)
        self.assertEqual(levenshtein_distance("", "টাকা"), 4)

    def test_single_operations(self):
        # Insertion
        self.assertEqual(levenshtein_distance("cat", "cats"), 1)
        # Deletion
        self.assertEqual(levenshtein_distance("bKash", "bkash"), 1)  # Case change is substitution
        self.assertEqual(levenshtein_distance("bkashh", "bkash"), 1)
        # Substitution
        self.assertEqual(levenshtein_distance("FREE", "FR3E"), 1)
        self.assertEqual(levenshtein_distance("FREE", "FR33"), 2)

    def test_bengali_unicode_edit_distance(self):
        # Bengali vowel matra variation: লটারি (হ্রস্ব-ইকার) vs লটারী (দীর্ঘ-ঈকার)
        # 'ি' (\u09bf) vs 'ী' (\u09c0) is 1 substitution
        self.assertEqual(levenshtein_distance("লটারি", "লটারী"), 1)

        # Bengali vowel variation: ফ্রি vs ফ্রী
        self.assertEqual(levenshtein_distance("ফ্রি", "ফ্রী"), 1)

        # Bengali character insertion: টাকা vs টাক্কা
        self.assertEqual(levenshtein_distance("টাকা", "টাক্কা"), 2)

    def test_similarity_score(self):
        self.assertEqual(levenshtein_similarity("", ""), 1.0)
        self.assertEqual(levenshtein_similarity("bKash", "bKash"), 1.0)
        # "FREE" vs "FR33": dist=2, max_len=4 -> sim=0.5
        self.assertEqual(levenshtein_similarity("FREE", "FR33"), 0.5)
        # Symmetry check
        self.assertEqual(
            levenshtein_similarity("লটারি", "লটারী"),
            levenshtein_similarity("লটারী", "লটারি"),
        )

    def test_edit_distance_demonstration(self):
        demo_results = run_edit_distance_demonstration()
        self.assertGreater(len(demo_results), 0)
        for item in demo_results:
            self.assertIn("string_1", item)
            self.assertIn("string_2", item)
            self.assertIn("levenshtein_distance", item)
            self.assertIn("similarity_score", item)
            self.assertIn("category", item)
            self.assertIsInstance(item["levenshtein_distance"], int)
            self.assertGreaterEqual(item["similarity_score"], 0.0)
            self.assertLessEqual(item["similarity_score"], 1.0)


class TestNLPExperimentPipelines(unittest.TestCase):
    """Test suite for experiment vectorizers and training pipelines."""

    def test_trigram_vectorizer_config(self):
        vec = build_trigram_vectorizer(max_features=500, ngram_range=(1, 3))
        self.assertEqual(vec.ngram_range, (1, 3))
        self.assertEqual(vec.max_features, 500)
        self.assertTrue(vec.sublinear_tf)

    def test_char_ngram_vectorizer_config(self):
        vec = build_char_ngram_vectorizer(max_features=500, ngram_range=(3, 5), analyzer="char_wb")
        self.assertEqual(vec.ngram_range, (3, 5))
        self.assertEqual(vec.analyzer, "char_wb")

    def test_pipeline_evaluation_synthetic(self):
        X_train = pd.Series([
            "আপনি লটারি জিতেছেন টাকা নিন",
            "জরুরি মিটিং আছে কাল",
            "ক্যাশব্যাক অফার <MONEY> টাকা",
            "কেমন আছেন ভাই কেমন চলছে",
            "win free cash bonus now",
            "see you tomorrow at office",
        ])
        y_train = pd.Series([1, 0, 1, 0, 1, 0])

        X_test = pd.Series([
            "লটারি জিতেছেন <MONEY>",
            "কাল মিটিং কখন",
        ])
        y_test = pd.Series([1, 0])

        vec = build_trigram_vectorizer(min_df=1)
        clf = LogisticRegression(random_state=42)

        res = evaluate_model_pipeline(vec, clf, X_train, y_train, X_test, y_test, "Synthetic Trigram")

        self.assertEqual(res["experiment_name"], "Synthetic Trigram")
        self.assertIn("accuracy", res)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1_score", res)
        self.assertIn("confusion_matrix", res)
        self.assertGreater(res["vocab_size"], 0)
        self.assertGreaterEqual(res["train_time_ms"], 0.0)

    def test_comparative_experiments_execution(self):
        # Runs full comparative experiments on processed dataset
        results = run_comparative_experiments()
        self.assertIn("experiments", results)
        self.assertEqual(len(results["experiments"]), 3)
        exp_names = [e["experiment_name"] for e in results["experiments"]]
        self.assertIn("Baseline: Word TF-IDF (1,2)", exp_names)
        self.assertIn("Trigram: Word TF-IDF (1,3)", exp_names)
        self.assertIn("Character: Char TF-IDF (3,5)", exp_names)

        for exp in results["experiments"]:
            self.assertGreater(exp["accuracy"], 0.70)
            self.assertGreater(exp["f1_score"], 0.70)
            self.assertGreater(exp["vocab_size"], 100)

    def test_extract_nlp_features(self):
        text = "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! পুরস্কার পেতে কল করুন ০১৭১২৩৪৫৬৭৮"
        from training.nlp_experiments import extract_nlp_features, STATIC_MODEL_BENCHMARKS
        res = extract_nlp_features(text)

        self.assertEqual(res["original_text"], text)
        self.assertIn("<MONEY>", res["preprocessed_text"])
        self.assertIn("<PHONE>", res["preprocessed_text"])
        self.assertGreaterEqual(len(res["unigrams"]), 5)
        self.assertGreaterEqual(len(res["bigrams"]), 4)
        self.assertGreaterEqual(len(res["trigrams"]), 3)
        self.assertIn("<MONEY>", res["detected_entities"])
        self.assertIn("<PHONE>", res["detected_entities"])

        # Subword char n-grams
        self.assertIn("3_grams", res["character_ngrams"])
        self.assertIn("4_grams", res["character_ngrams"])
        self.assertIn("5_grams", res["character_ngrams"])

        # Static benchmarks verification
        self.assertEqual(len(STATIC_MODEL_BENCHMARKS), 3)
        self.assertTrue(STATIC_MODEL_BENCHMARKS[0]["is_production"])


if __name__ == "__main__":
    unittest.main()

