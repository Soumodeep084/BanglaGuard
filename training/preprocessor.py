"""
Bengali Text Preprocessor for BanglaGuard SMS Spam Detection.

Provides Unicode normalization, entity token replacement (URLs, phone numbers,
emails, money amounts, and standalone numbers), and whitespace normalization
while preserving Bengali vocabulary, punctuation, and contextual signals.
"""

import re
import unicodedata
from typing import Optional


class BengaliTextPreprocessor:
    """
    NLP Preprocessor specifically designed for Bengali SMS messages.

    Attributes:
        normalize_unicode (bool): Whether to normalize Unicode characters (NFKC).
        replace_urls (bool): Replace URLs with <URL>.
        replace_phones (bool): Replace phone numbers with <PHONE>.
        replace_emails (bool): Replace email addresses with <EMAIL>.
        replace_money (bool): Replace money expressions with <MONEY>.
        replace_numbers (bool): Replace remaining numbers with <NUMBER>.
        clean_whitespace (bool): Normalize multiple spaces, tabs, and newlines.
    """

    # Bengali digits mapping
    BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
    ALL_DIGITS = "0-9" + BENGALI_DIGITS

    # Regex patterns
    # 1. URL pattern: matches http(s), www, domain names with common TLDs, short URLs
    URL_PATTERN = re.compile(
        r"(?:https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|mil|bd|io|co|me|ly|info|xyz|app|ai|tech|site|online|store)(?:/[^\s()<>]*[^\s`!()\[\]{};:'\".,<>?«»“”‘’])?)",
        flags=re.IGNORECASE,
    )

    # 2. Email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        flags=re.IGNORECASE,
    )

    # 3. Phone pattern (Bangladeshi mobile, hotlines, and international numbers in English & Bengali digits)
    # Examples: +88017XXXXXXXX, 017XXXXXXXX, +৮৮০১৭XXXXXXXX, ০১৭১১XXXXXX, 16135, etc.
    PHONE_PATTERN = re.compile(
        r"(?:"
        # International & Bangladeshi full/mobile phone numbers
        r"(?:\+?880|\+?৮৮০|01|০১)[0-9০-৯\s\-]{8,12}\b"
        r"|"
        # Standard mobile 11 digits: 013..., 014..., 015..., 016..., 017..., 018..., 019...
        r"\b01[3-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b"
        r"|"
        r"\b০১[৩-৯][০-৯]{2}[\s\-]?[০-৯]{3}[\s\-]?[০-৯]{3}\b"
        r"|"
        # Bangladeshi 5-digit shortcodes/helplines (16xxx, 106, 999, 109, 333, etc.)
        r"\b(?:16[0-9]{3}|১৬[০-৯]{3}|106|999|333|109|১০৬|৯৯৯|৩৩৩|১০৯)\b"
        r")",
        flags=re.IGNORECASE,
    )

    # 4. Money pattern (Bengali & English amounts with currency indicators like টাকা, ৳, tk, taka, bdt, etc.)
    # Examples: ২০০ টাকা, 20 টাকা, 220.01 টাকা, ৳ ৫০০, ৳500, TK 500, 500tk, 500/-
    MONEY_PATTERN = re.compile(
        r"(?:"
        # Currency prefix: e.g. ৳500, ৳ 500, TK 500, Tk. 500, BDT 500, $500
        r"(?:৳|tk\.?|taka|bdt|\$|rs\.?|টাকা)\s*[" + ALL_DIGITS + r"]+(?:[.,][" + ALL_DIGITS + r"]+)?(?:\s*(?:/-|হাজার|লাখ|কোটি))?"
        r"|"
        # Amount followed by currency suffix: e.g. 500 টাকা, 20.01 টাকা, 500/- , 500tk, ৫০০ টাকা
        r"[" + ALL_DIGITS + r"]+(?:[.,][" + ALL_DIGITS + r"]+)?\s*(?:৳|টাকা|টাকার|টাকায়|tk\.?|taka|bdt|/-)"
        r")",
        flags=re.IGNORECASE,
    )

    # 5. Remaining standalone numbers (English and Bengali numerals)
    NUMBER_PATTERN = re.compile(
        r"\b[" + ALL_DIGITS + r"]+(?:[.,][" + ALL_DIGITS + r"]+)?\b"
    )

    # Extra whitespace pattern
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        normalize_unicode: bool = True,
        replace_urls: bool = True,
        replace_phones: bool = True,
        replace_emails: bool = True,
        replace_money: bool = True,
        replace_numbers: bool = True,
        clean_whitespace: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.replace_urls = replace_urls
        self.replace_phones = replace_phones
        self.replace_emails = replace_emails
        self.replace_money = replace_money
        self.replace_numbers = replace_numbers
        self.clean_whitespace = clean_whitespace

    def preprocess(self, text: Optional[str]) -> str:
        """
        Preprocess a single text message according to the configured options.

        Args:
            text: Raw input string.

        Returns:
            Preprocessed Bengali text string.
        """
        if text is None:
            return ""

        text = str(text)

        # 1. Unicode normalization (NFKC)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
            # Remove zero-width spaces and non-functional invisible characters
            text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")

        # 2. Entity replacements in order of specificity
        # Email addresses first (so domain in email doesn't get matched by URL pattern)
        if self.replace_emails:
            text = self.EMAIL_PATTERN.sub(" <EMAIL> ", text)

        # URLs next (before dots, slashes, or numbers in URLs get matched)
        if self.replace_urls:
            text = self.URL_PATTERN.sub(" <URL> ", text)

        # Phone numbers before generic money/numbers
        if self.replace_phones:
            text = self.PHONE_PATTERN.sub(" <PHONE> ", text)

        # Money expressions before generic numbers
        if self.replace_money:
            text = self.MONEY_PATTERN.sub(" <MONEY> ", text)

        # Standalone remaining numbers
        if self.replace_numbers:
            text = self.NUMBER_PATTERN.sub(" <NUMBER> ", text)

        # 3. Clean up extra whitespaces
        if self.clean_whitespace:
            text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text


# Default global instance for quick functional usage
_default_preprocessor = BengaliTextPreprocessor()


def preprocess_bengali_text(text: Optional[str]) -> str:
    """
    Convenience function to preprocess Bengali text using default settings.

    Args:
        text: Input SMS text string.

    Returns:
        Cleaned and preprocessed text string.
    """
    return _default_preprocessor.preprocess(text)
