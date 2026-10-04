"""
Unit tests for FastAPI Web interface and NLP Prediction endpoints in BanglaGuard.
"""

import unittest
from starlette.testclient import TestClient
from app.main import app


class TestBanglaGuardAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_root_renders_html_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("BanglaGuard", response.text)
        self.assertIn("Bengali SMS Spam Detection", response.text)
        self.assertIn("sms-textarea", response.text)
        self.assertIn("analyze-btn", response.text)
        self.assertIn("result-container", response.text)
        self.assertIn("Class Probabilities", response.text)

    def test_get_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("model_type"), "TF-IDF + Logistic Regression")

    def test_static_css_accessible(self):
        response = self.client.get("/static/style.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.headers.get("content-type", ""))
        self.assertIn("var(--bg-primary)", response.text)

    def test_post_predict_spam_message(self):
        sample_spam = "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! পুরস্কার পেতে ভিজিট করুন https://prize-claim.bd"
        response = self.client.post("/predict", json={"text": sample_spam})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("prediction"), "spam")
        self.assertEqual(data.get("label"), "spam")
        self.assertEqual(data.get("label_id"), 1)
        self.assertGreater(data.get("confidence"), 0.5)
        self.assertGreater(data.get("spam_probability"), 0.5)
        self.assertLess(data.get("ham_probability"), 0.5)
        self.assertIn("<MONEY>", data.get("processed_text"))
        self.assertIn("<URL>", data.get("processed_text"))
        self.assertEqual(data.get("model_name"), "TF-IDF + Logistic Regression")

    def test_post_predict_ham_message(self):
        sample_ham = "শুভ সকাল মা, আমি বাসায় পৌঁছে গেছি। চিন্তা করো না, ভালো আছি।"
        response = self.client.post("/predict", json={"text": sample_ham})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("prediction"), "ham")
        self.assertEqual(data.get("label"), "ham")
        self.assertEqual(data.get("label_id"), 0)
        self.assertGreater(data.get("confidence"), 0.5)
        self.assertGreater(data.get("ham_probability"), 0.5)
        self.assertLess(data.get("spam_probability"), 0.5)
        self.assertIn("বাসায় পৌঁছে গেছি", data.get("processed_text"))

    def test_post_predict_empty_or_whitespace_validation(self):
        # Empty string
        res_empty = self.client.post("/predict", json={"text": ""})
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("cannot be empty", res_empty.json().get("detail", ""))

        # Whitespace only
        res_whitespace = self.client.post("/predict", json={"text": "   \n\t  "})
        self.assertEqual(res_whitespace.status_code, 400)
        self.assertIn("cannot be empty", res_whitespace.json().get("detail", ""))

    def test_post_analyze_alias(self):
        response = self.client.post("/analyze", json={"text": "দোস্ত কাল দেখা হবে"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)

    def test_post_test_sms_backward_compatibility(self):
        response = self.client.post("/test-sms", json={"text": "টেস্ট মেসেজ"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("received_text"), "টেস্ট মেসেজ")

    def test_post_nlp_analyze_endpoint(self):
        sample_text = "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! পুরস্কার পেতে কল করুন ০১৭১২৩৪৫৬৭৮"
        response = self.client.post("/nlp/analyze", json={"text": sample_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("original_text"), sample_text)
        self.assertIn("<MONEY>", data.get("preprocessed_text"))
        self.assertIn("<PHONE>", data.get("preprocessed_text"))
        self.assertIsInstance(data.get("tokens"), list)
        self.assertIsInstance(data.get("unigrams"), list)
        self.assertIsInstance(data.get("bigrams"), list)
        self.assertIsInstance(data.get("trigrams"), list)
        self.assertGreater(len(data.get("unigrams")), 0)
        self.assertGreater(len(data.get("bigrams")), 0)
        self.assertGreater(len(data.get("trigrams")), 0)
        self.assertIn("<MONEY>", data.get("detected_entities"))
        self.assertIn("<PHONE>", data.get("detected_entities"))

        # Character N-Grams
        char_ngrams = data.get("character_ngrams", {})
        self.assertIn("3_grams", char_ngrams)
        self.assertIn("4_grams", char_ngrams)
        self.assertIn("5_grams", char_ngrams)
        self.assertGreater(len(char_ngrams["3_grams"]), 0)

        # Benchmarks
        benchmarks = data.get("benchmarks", [])
        self.assertEqual(len(benchmarks), 3)
        self.assertTrue(benchmarks[0]["is_production"])

    def test_post_nlp_analyze_empty_validation(self):
        response = self.client.post("/nlp/analyze", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_post_nlp_edit_distance_endpoint(self):
        response = self.client.post(
            "/nlp/edit-distance",
            json={"string_1": "লটারি", "string_2": "লটারী"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("string_1"), "লটারি")
        self.assertEqual(data.get("string_2"), "লটারী")
        self.assertEqual(data.get("levenshtein_distance"), 1)
        self.assertEqual(data.get("similarity_score"), 0.8)

    def test_get_nlp_benchmarks_endpoint(self):
        response = self.client.get("/nlp/benchmarks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(len(data.get("benchmarks", [])), 3)
        self.assertTrue(any(b["is_production"] for b in data["benchmarks"]))


if __name__ == "__main__":
    unittest.main()

