"""
Unit tests for BanglaGuard Baseline Model (TF-IDF + Logistic Regression).
"""

import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from training.baseline_model import (
    BaselinePredictor,
    build_tfidf_vectorizer,
    evaluate_baseline_model,
    load_baseline_artifacts,
    predict_sms,
    save_baseline_artifacts,
    split_dataset,
    train_baseline_model,
)


class TestBaselineModel(unittest.TestCase):

    def setUp(self):
        # Synthetic Bengali toy dataset for quick isolated unit tests
        self.sample_df = pd.DataFrame(
            {
                "type": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
                "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "original_text": [
                    "কেমন আছো বন্ধু?",
                    "জিতুন ৫০০ টাকা পুরস্কার!",
                    "আজকে বিকেলে দেখা হবে।",
                    "লটারি জিততে ডায়াল করুন ১৬১২৩",
                    "কাল ক্যাম্পাসে যাবে কি?",
                    "অফার পান ফ্রি ইন্টারনেট কল করুন",
                    "বাসায় পৌঁছে গেছি মা।",
                    "বোনাস পেতে ভিজিট করুন http://spam.bd",
                    "শুভ জন্মদিন প্রিয়!",
                    "বিজ্ঞাপন: ক্যাশব্যাক পেতে রিচার্জ করুন ৫০০ টাকা",
                ],
                "processed_text": [
                    "কেমন আছো বন্ধু?",
                    "জিতুন <MONEY> পুরস্কার!",
                    "আজকে বিকেলে দেখা হবে।",
                    "লটারি জিততে ডায়াল করুন <PHONE>",
                    "কাল ক্যাম্পাসে যাবে কি?",
                    "অফার পান ফ্রি ইন্টারনেট কল করুন",
                    "বাসায় পৌঁছে গেছি মা।",
                    "বোনাস পেতে ভিজিট করুন <URL>",
                    "শুভ জন্মদিন প্রিয়!",
                    "বিজ্ঞাপন ক্যাশব্যাক পেতে রিচার্জ করুন <MONEY>",
                ],
            }
        )

    def test_split_dataset_stratification_and_ratio(self):
        X_train, X_test, y_train, y_test = split_dataset(
            self.sample_df,
            test_size=0.20,
            random_state=42,
            stratify=True,
        )

        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 8)
        self.assertEqual(len(y_test), 2)

        # In stratified split with balanced 50/50 dataset, test set should have 1 spam and 1 ham
        self.assertEqual((y_test == 1).sum(), 1)
        self.assertEqual((y_test == 0).sum(), 1)

    def test_build_tfidf_vectorizer_entity_token_preservation(self):
        vec = build_tfidf_vectorizer(min_df=1)
        corpus = [
            "জিতুন <MONEY> টাকা <URL>",
            "যোগাযোগ <PHONE> এবং <EMAIL>",
        ]
        X = vec.fit_transform(corpus)
        vocab = vec.vocabulary_

        # Special entity tokens should be preserved as distinct features
        self.assertIn("<money>", vocab)
        self.assertIn("<url>", vocab)
        self.assertIn("<phone>", vocab)
        self.assertIn("<email>", vocab)
        self.assertIn("জিতুন", vocab)

    def test_train_and_evaluate_baseline_model(self):
        X_train = self.sample_df["processed_text"].iloc[:8]
        y_train = self.sample_df["label"].iloc[:8]
        X_test = self.sample_df["processed_text"].iloc[8:]
        y_test = self.sample_df["label"].iloc[8:]

        vec = build_tfidf_vectorizer(min_df=1)
        fitted_vec, model = train_baseline_model(
            X_train=X_train,
            y_train=y_train,
            vectorizer=vec,
            random_state=42,
        )

        self.assertIsInstance(fitted_vec, TfidfVectorizer)
        self.assertIsInstance(model, LogisticRegression)

        metrics = evaluate_baseline_model(
            model=model,
            vectorizer=fitted_vec,
            X_test=X_test,
            y_test=y_test,
        )

        # Check required evaluation keys
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("matrix", metrics["confusion_matrix"])

        # Check valid metric ranges
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["test_samples"], 2)

    def test_save_and_load_artifacts(self):
        X_train = self.sample_df["processed_text"]
        y_train = self.sample_df["label"]

        vec = build_tfidf_vectorizer(min_df=1)
        fitted_vec, model = train_baseline_model(X_train, y_train, vectorizer=vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_paths = save_baseline_artifacts(
                vectorizer=fitted_vec,
                model=model,
                metrics={"accuracy": 0.95},
                model_dir=tmpdir,
            )

            self.assertTrue(save_paths["vectorizer"].exists())
            self.assertTrue(save_paths["model"].exists())
            self.assertTrue(save_paths["metadata"].exists())

            loaded_vec, loaded_model, loaded_meta = load_baseline_artifacts(tmpdir)
            self.assertIsInstance(loaded_vec, TfidfVectorizer)
            self.assertIsInstance(loaded_model, LogisticRegression)
            self.assertEqual(loaded_meta.get("metrics", {}).get("accuracy"), 0.95)

    def test_predict_sms_function(self):
        X_train = self.sample_df["processed_text"]
        y_train = self.sample_df["label"]

        vec = build_tfidf_vectorizer(min_df=1)
        fitted_vec, model = train_baseline_model(X_train, y_train, vectorizer=vec)

        # Test single prediction
        result = predict_sms(
            text="অভিনন্দন! আপনি জিতেছেন ৫০০ টাকা",
            vectorizer=fitted_vec,
            model=model,
        )

        self.assertIn("label", result)
        self.assertIn(result["label"], ["spam", "ham"])
        self.assertIn("label_id", result)
        self.assertIn(result["label_id"], [0, 1])
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("spam_probability", result)
        self.assertIn("ham_probability", result)
        self.assertAlmostEqual(result["spam_probability"] + result["ham_probability"], 1.0, places=3)

    def test_baseline_predictor_class(self):
        X_train = self.sample_df["processed_text"]
        y_train = self.sample_df["label"]

        vec = build_tfidf_vectorizer(min_df=1)
        fitted_vec, model = train_baseline_model(X_train, y_train, vectorizer=vec)

        predictor = BaselinePredictor(vectorizer=fitted_vec, model=model)
        batch_results = predictor.predict_batch(["কেমন আছো?", "লটারি অফার"])

        self.assertEqual(len(batch_results), 2)
        self.assertIn(batch_results[0]["label"], ["ham", "spam"])
        self.assertIn(batch_results[1]["label"], ["ham", "spam"])


if __name__ == "__main__":
    unittest.main()
