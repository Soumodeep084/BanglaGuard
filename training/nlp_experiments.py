"""
BanglaGuard NLP Experiments Module.

This module evaluates additional NLP syllabus concepts on Bengali SMS text:
1. Trigram Experiment: Word-level TF-IDF with ngram_range=(1, 3) + Logistic Regression.
2. Character N-gram Experiment: Character-level TF-IDF with ngram_range=(3, 5) + Logistic Regression.
3. Edit Distance Analysis: Levenshtein distance on Bengali & English SMS spelling variations / spam obfuscations.
4. Model Comparison: Comparative evaluation of Baseline vs Trigram vs Character models.

NOTE: This is strictly an offline experimentation module and does not alter production models or API endpoints.
"""

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace root is on sys.path when script is executed directly
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.baseline_model import load_processed_dataset, split_dataset
from training.preprocessor import BengaliTextPreprocessor

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Token extraction pattern aligned with baseline vectorizer
TOKEN_PATTERN_REGEX = re.compile(r"(?u)[\w\u0980-\u09ff]+|<\w+>")

# Empirical benchmarks precomputed from the experimental evaluation
STATIC_MODEL_BENCHMARKS = [
    {
        "model_id": "baseline_word_1_2",
        "name": "Word TF-IDF (1,2) + Logistic Regression",
        "is_production": True,
        "feature_type": "Word Unigrams & Bigrams",
        "ngram_range": "(1, 2)",
        "accuracy": 0.9689,
        "precision": 0.9747,
        "recall": 0.9677,
        "f1_score": 0.9712,
        "vocab_size": 8746,
        "train_time_ms": 120.2,
        "confusion_matrix": {"tn": 229, "fp": 7, "fn": 9, "tp": 270},
        "status_label": "Production Model",
    },
    {
        "model_id": "trigram_word_1_3",
        "name": "Word TF-IDF (1,3) + Logistic Regression",
        "is_production": False,
        "feature_type": "Word Unigrams, Bigrams & Trigrams",
        "ngram_range": "(1, 3)",
        "accuracy": 0.9670,
        "precision": 0.9746,
        "recall": 0.9642,
        "f1_score": 0.9694,
        "vocab_size": 14550,
        "train_time_ms": 187.1,
        "confusion_matrix": {"tn": 229, "fp": 7, "fn": 10, "tp": 269},
        "status_label": "Offline Experiment",
    },
    {
        "model_id": "char_ngram_3_5",
        "name": "Character TF-IDF (3,5) + Logistic Regression",
        "is_production": False,
        "feature_type": "Subword Character N-Grams (char_wb)",
        "ngram_range": "(3, 5)",
        "accuracy": 0.9612,
        "precision": 0.9814,
        "recall": 0.9462,
        "f1_score": 0.9635,
        "vocab_size": 15000,
        "train_time_ms": 446.7,
        "confusion_matrix": {"tn": 231, "fp": 5, "fn": 15, "tp": 264},
        "status_label": "Offline Experiment",
    },
]


def extract_nlp_features(raw_text: str) -> Dict[str, Any]:
    """
    Deconstruct an SMS text into its foundational NLP components:
    - Preprocessed & Unicode normalized text
    - Extracted word tokens (Bengali words & entity tags)
    - Unigrams, Bigrams, and Trigrams
    - Detected entity tokens (<URL>, <PHONE>, <MONEY>, <NUMBER>, <EMAIL>)
    - Subword Character N-grams (3-grams, 4-grams, 5-grams)

    Args:
        raw_text: Original raw input string.

    Returns:
        Structured dictionary containing extracted NLP features.
    """
    if raw_text is None:
        raw_text = ""

    preprocessor = BengaliTextPreprocessor()
    preprocessed_text = preprocessor.preprocess(raw_text)

    # Extract word tokens matching our vocabulary definition
    tokens = TOKEN_PATTERN_REGEX.findall(preprocessed_text)

    # Word N-grams
    unigrams = list(tokens)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]

    # Entity tokens identification
    entity_tokens = [tok for tok in tokens if tok.startswith("<") and tok.endswith(">")]

    # Character N-grams from preprocessed text words
    char_3grams = []
    char_4grams = []
    char_5grams = []

    for word in tokens:
        # Generate character n-grams within word boundaries
        if len(word) >= 3:
            for i in range(len(word) - 2):
                char_3grams.append(word[i:i+3])
        if len(word) >= 4:
            for i in range(len(word) - 3):
                char_4grams.append(word[i:i+4])
        if len(word) >= 5:
            for i in range(len(word) - 4):
                char_5grams.append(word[i:i+5])

    return {
        "original_text": raw_text,
        "preprocessed_text": preprocessed_text,
        "tokens": tokens,
        "token_count": len(tokens),
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "detected_entities": entity_tokens,
        "character_ngrams": {
            "3_grams": char_3grams,
            "4_grams": char_4grams,
            "5_grams": char_5grams,
        },
        "character_counts": {
            "total_chars": len(raw_text),
            "normalized_chars": len(preprocessed_text),
            "char_3grams_count": len(char_3grams),
            "char_4grams_count": len(char_4grams),
            "char_5grams_count": len(char_5grams),
        },
    }


# ==============================================================================
# 1. EDIT DISTANCE UTILITY (LEVENSHTEIN)
# ==============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two strings s1 and s2.

    Levenshtein distance measures the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform s1 into s2.

    Supports Unicode Bengali characters, matras, and Latin scripts.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        Integer minimum edit distance.
    """
    m, n = len(s1), len(s2)

    # Base cases
    if m == 0:
        return n
    if n == 0:
        return m

    # DP matrix of size (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[m][n]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized similarity score based on Levenshtein distance.

    Similarity = 1.0 - (distance / max(len(s1), len(s2)))

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Float similarity score in range [0.0, 1.0].
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return round(1.0 - (dist / max_len), 4)


def run_edit_distance_demonstration(
    custom_pairs: Optional[List[Tuple[str, str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Demonstrate edit distance analysis on spelling variations, vowel alternatives,
    and adversarial obfuscations in Bengali and English SMS spam.

    Args:
        custom_pairs: Optional list of tuples (word1, word2, description).

    Returns:
        List of dictionaries containing comparison details and metrics.
    """
    default_pairs = [
        ("লটারি", "লটারী", "Bengali vowel matra variation (Hroshoi vs Dirghoi)"),
        ("পুরস্কার", "পু র স্কা র", "Bengali adversarial token spacing / bypass"),
        ("ফ্রি", "ফ্রী", "Bengali spelling variant (Free)"),
        ("টাকা", "টাক্কা", "Bengali phonetic / colloquial emphasis (Taka)"),
        ("bKash", "bkashh", "Brand name character elongation / typo"),
        ("bKash", "bK@sh", "Brand name symbol substitution obfuscation"),
        ("FREE", "FR33", "English leetspeak spam substitution"),
        ("BONUS", "B0NUS", "English zero-for-O digit substitution"),
        ("WINNER", "W1NNER", "English one-for-I digit substitution"),
        ("01711000000", "01711-OOO-OOO", "Phone number obfuscation using letter 'O'"),
    ]

    pairs_to_evaluate = custom_pairs if custom_pairs is not None else default_pairs
    results = []

    for word1, word2, category in pairs_to_evaluate:
        dist = levenshtein_distance(word1, word2)
        sim = levenshtein_similarity(word1, word2)
        results.append({
            "string_1": word1,
            "string_2": word2,
            "category": category,
            "levenshtein_distance": dist,
            "similarity_score": sim,
        })

    return results


# ==============================================================================
# 2. VECTORIZER FACTORIES FOR EXPERIMENTS
# ==============================================================================

def build_trigram_vectorizer(
    max_features: Optional[int] = 15000,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 2,
    sublinear_tf: bool = True,
    token_pattern: str = r"(?u)[\w\u0980-\u09ff]+|<\w+>",
) -> TfidfVectorizer:
    """
    Build a word-level TF-IDF vectorizer configured for unigrams, bigrams, and trigrams.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range (1, 3) for unigrams, bigrams, and trigrams.
        min_df: Minimum document frequency threshold.
        sublinear_tf: Apply sublinear tf scaling.
        token_pattern: Regex pattern matching Bengali Unicode words & entity tokens.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern,
    )


def build_char_ngram_vectorizer(
    max_features: Optional[int] = 15000,
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
    sublinear_tf: bool = True,
    analyzer: str = "char_wb",
) -> TfidfVectorizer:
    """
    Build a character-level TF-IDF vectorizer for character n-grams.

    Uses character n-grams inside word boundaries ('char_wb') or across text ('char')
    to capture subword roots, spelling variants, and morphosyntactic affixes.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range (3, 5) for 3-grams to 5-grams.
        min_df: Minimum document frequency threshold.
        sublinear_tf: Apply sublinear tf scaling.
        analyzer: 'char_wb' (within word boundaries) or 'char'.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        analyzer=analyzer,
    )


# ==============================================================================
# 3. EXPERIMENT RUNNERS
# ==============================================================================

def evaluate_model_pipeline(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X_train: Union[pd.Series, List[str]],
    y_train: Union[pd.Series, List[int], np.ndarray],
    X_test: Union[pd.Series, List[str]],
    y_test: Union[pd.Series, List[int], np.ndarray],
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Fit a vectorizer and Logistic Regression classifier, then evaluate on test data.

    Measures Accuracy, Precision, Recall, F1, Confusion Matrix, Feature Count,
    and Fit Duration.

    Args:
        vectorizer: Unfitted TfidfVectorizer instance.
        model: Unfitted LogisticRegression instance.
        X_train: Training text series.
        y_train: Training labels.
        X_test: Testing text series.
        y_test: Testing labels.
        experiment_name: Descriptive name of the experiment.

    Returns:
        Dictionary of evaluation metrics and metadata.
    """
    start_time = time.perf_counter()
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    train_duration_sec = time.perf_counter() - start_time

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    vocab_size = len(vectorizer.vocabulary_)

    return {
        "experiment_name": experiment_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "vocab_size": vocab_size,
        "train_time_ms": round(train_duration_sec * 1000, 2),
        "vectorizer": vectorizer,
        "model": model,
    }


def run_comparative_experiments(
    dataset_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute all three comparative experiments on the standard 80/20 split:
    1. Baseline: Word TF-IDF (1, 2) + Logistic Regression
    2. Trigram: Word TF-IDF (1, 3) + Logistic Regression
    3. Character: Character TF-IDF (3, 5) + Logistic Regression

    Args:
        dataset_path: Path to processed dataset CSV.
        test_size: Test partition fraction (default 0.20).
        random_state: Random seed for stratified splitting (default 42).

    Returns:
        Dictionary containing experiment results and edit distance demonstrations.
    """
    df = load_processed_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(
        df, test_size=test_size, random_state=random_state, stratify=True
    )

    # 1. Baseline: Word TF-IDF (1, 2)
    baseline_vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)[\w\u0980-\u09ff]+|<\w+>",
    )
    baseline_clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs"
    )
    baseline_res = evaluate_model_pipeline(
        baseline_vec, baseline_clf, X_train, y_train, X_test, y_test, "Baseline: Word TF-IDF (1,2)"
    )

    # 2. Trigram Experiment: Word TF-IDF (1, 3)
    trigram_vec = build_trigram_vectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)[\w\u0980-\u09ff]+|<\w+>",
    )
    trigram_clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs"
    )
    trigram_res = evaluate_model_pipeline(
        trigram_vec, trigram_clf, X_train, y_train, X_test, y_test, "Trigram: Word TF-IDF (1,3)"
    )

    # 3. Character N-gram Experiment: Char TF-IDF (3, 5)
    char_vec = build_char_ngram_vectorizer(
        max_features=15000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        analyzer="char_wb",
    )
    char_clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs"
    )
    char_res = evaluate_model_pipeline(
        char_vec, char_clf, X_train, y_train, X_test, y_test, "Character: Char TF-IDF (3,5)"
    )

    # 4. Edit Distance Demonstrations
    edit_dist_res = run_edit_distance_demonstration()

    return {
        "dataset_total": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "experiments": [baseline_res, trigram_res, char_res],
        "edit_distance_demo": edit_dist_res,
    }


def print_experiment_report(results: Dict[str, Any]) -> None:
    """
    Format and print a human-readable comparison report to standard output.
    """
    print("\n" + "=" * 78)
    print("           BANGLAGUARD NLP EXPERIMENTAL EVALUATION REPORT")
    print("=" * 78)
    print(f"Dataset Size: {results['dataset_total']} (Train: {results['train_size']}, Test: {results['test_size']})")
    print(f"Split: Stratified 80/20 (Random State = 42)")
    print("-" * 78)

    header = f"{'Experiment':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8} | {'Vocab':<6} | {'Time (ms)':<9}"
    print(header)
    print("-" * 78)

    for exp in results["experiments"]:
        name = exp["experiment_name"]
        acc = f"{exp['accuracy']:.4f}"
        prec = f"{exp['precision']:.4f}"
        rec = f"{exp['recall']:.4f}"
        f1 = f"{exp['f1_score']:.4f}"
        vocab = str(exp["vocab_size"])
        t_ms = f"{exp['train_time_ms']:.1f}"
        print(f"{name:<28} | {acc:<8} | {prec:<9} | {rec:<6} | {f1:<8} | {vocab:<6} | {t_ms:<9}")

    print("-" * 78)
    print("\nCONFUSION MATRICES (Row=Actual [Ham, Spam], Col=Predicted [Ham, Spam]):")
    for exp in results["experiments"]:
        cm = exp["confusion_matrix"]
        print(f"  * {exp['experiment_name']}: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    print("\n" + "=" * 78)
    print("          LEVENSHTEIN EDIT DISTANCE ANALYSIS ON SMS TEXT")
    print("=" * 78)
    demo_header = f"{'String 1':<16} | {'String 2':<16} | {'Dist':<4} | {'Sim':<6} | {'Description'}"
    print(demo_header)
    print("-" * 78)
    for item in results["edit_distance_demo"]:
        s1 = item["string_1"]
        s2 = item["string_2"]
        dist = item["levenshtein_distance"]
        sim = f"{item['similarity_score']:.2f}"
        desc = item["category"]
        print(f"{s1:<16} | {s2:<16} | {dist:<4} | {sim:<6} | {desc}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    results = run_comparative_experiments()
    print_experiment_report(results)
