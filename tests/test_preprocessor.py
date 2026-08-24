"""
Unit tests for BengaliTextPreprocessor.
"""

import unittest
from training.preprocessor import BengaliTextPreprocessor, preprocess_bengali_text


class TestBengaliTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = BengaliTextPreprocessor()

    def test_unicode_normalization_and_zero_width(self):
        # Text with zero-width spaces (\u200b)
        text_with_zwsp = "হ্যালো\u200b বিশ্ব!"
        result = self.preprocessor.preprocess(text_with_zwsp)
        self.assertEqual(result, "হ্যালো বিশ্ব!")

    def test_url_replacement(self):
        sample = "বিস্তারিত জানতে ভিজিট করুন https://www.example.com/promo অথবা bit.ly/offer"
        result = self.preprocessor.preprocess(sample)
        self.assertIn("<URL>", result)
        self.assertNotIn("https://", result)
        self.assertNotIn("bit.ly", result)

    def test_email_replacement(self):
        sample = "যোগাযোগ করুন info.support@company.com.bd এই ঠিকানায়"
        result = self.preprocessor.preprocess(sample)
        self.assertIn("<EMAIL>", result)
        self.assertNotIn("info.support@company.com.bd", result)

    def test_phone_replacement_english_digits(self):
        sample = "যেকোনো অনুসন্ধানে কল করুন +8801712345678 বা 01912345678"
        result = self.preprocessor.preprocess(sample)
        self.assertIn("<PHONE>", result)
        self.assertNotIn("01712345678", result)
        self.assertNotIn("01912345678", result)

    def test_phone_replacement_bengali_digits(self):
        sample = "বিদেশ থেকে যোগাযোগ করুন +৮৮০৯৬১০১০২০৩০ নম্বরে"
        result = self.preprocessor.preprocess(sample)
        self.assertIn("<PHONE>", result)
        self.assertNotIn("৮৮০৯৬১০১০২০৩০", result)

    def test_money_replacement_bengali_and_english(self):
        samples = [
            "প্রতি রেফারেলে ২০০ টাকা জিতুন!",
            "রিচার্জ করুন ৳500 এবং পান বোনাস",
            "বর্তমান ব্যালেন্স 220.01 টাকা।",
            "মাত্র 500/- মূল্যে কিনুন",
        ]
        for s in samples:
            res = self.preprocessor.preprocess(s)
            self.assertIn("<MONEY>", res, f"Failed for {s} -> {res}")

    def test_standalone_numbers_replacement(self):
        sample = "আগামী ৩০ সেপ্টেম্বর রাত ৮ টায় অনুষ্ঠান"
        result = self.preprocessor.preprocess(sample)
        self.assertIn("<NUMBER>", result)
        self.assertNotIn("৩০", result)
        self.assertNotIn("৮", result)

    def test_preserves_bengali_words_and_punctuation(self):
        sample = "নববর্ষের শুভেচ্ছা!! আল্লাহ আপনার সকল কষ্ট দূর করুন।"
        result = self.preprocessor.preprocess(sample)
        self.assertEqual(result, "নববর্ষের শুভেচ্ছা!! আল্লাহ আপনার সকল কষ্ট দূর করুন।")

    def test_whitespace_cleanup(self):
        sample = "  এই   মেসেজটি    শেয়ার \n\n করুন  "
        result = self.preprocessor.preprocess(sample)
        self.assertEqual(result, "এই মেসেজটি শেয়ার করুন")

    def test_empty_and_none_input(self):
        self.assertEqual(self.preprocessor.preprocess(None), "")
        self.assertEqual(self.preprocessor.preprocess(""), "")
        self.assertEqual(self.preprocessor.preprocess("   "), "")

    def test_functional_helper(self):
        result = preprocess_bengali_text("১০০০ টাকা বোনাস")
        self.assertIn("<MONEY>", result)


if __name__ == "__main__":
    unittest.main()
