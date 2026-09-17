"""
Baseline NLP Text Classification Pipeline for BanglaGuard.

Implements:
1. Processed dataset loading and validation.
2. Stratified 80/20 train/test splitting.
3. TF-IDF vectorization tuned for Bengali SMS text & entity tokens.
4. Logistic Regression classification with balanced weighting.
5. Model evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
6. Model artifact persistence (saving/loading via joblib).
7. Reusable prediction function and BaselinePredictor class for SMS inference.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from training.preprocessor import BengaliTextPreprocessor, preprocess_bengali_text


# Default paths for processed dataset and model directory
DEFAULT_PROCESSED_PATHS = [
    Path("data/processed/processed_dataset.csv"),
    Path("data/processed_dataset.csv"),
]

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
DEFAULT_MODEL_FILENAME = "logistic_regression.joblib"
DEFAULT_METADATA_FILENAME = "baseline_metadata.json"


def locate_processed_dataset(dataset_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locate the processed dataset CSV file.

    Args:
        dataset_path: Explicit path if provided.

    Returns:
        Resolved Path to the processed dataset.

    Raises:
        FileNotFoundError: If no valid dataset file is found.
    """
    if dataset_path:
        p = Path(dataset_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified processed dataset not found at: {dataset_path}")

    for candidate in DEFAULT_PROCESSED_PATHS:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate processed dataset. Looked in: {[str(p) for p in DEFAULT_PROCESSED_PATHS]}"
    )


def load_processed_dataset(dataset_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load and validate the processed dataset.

    Args:
        dataset_path: Path to the processed CSV file.

    Returns:
        DataFrame containing 'processed_text' and 'label' columns.
    """
    resolved_path = locate_processed_dataset(dataset_path)

    try:
        df = pd.read_csv(resolved_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(resolved_path, encoding="utf-8-sig")

    required_cols = {"processed_text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed dataset at {resolved_path} is missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Clean any unexpected NaN values in processed_text or label
    df = df.dropna(subset=["processed_text", "label"]).copy()
    df["processed_text"] = df["processed_text"].astype(str).str.strip()
    df = df[df["processed_text"] != ""].copy()
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets using stratified sampling.

    Args:
        df: Processed DataFrame containing 'processed_text' and 'label'.
        test_size: Fraction for test partition (default 0.20 for 80/20 split).
        random_state: Random seed for reproducibility.
        stratify: Whether to use stratified splitting based on label.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df["processed_text"]
    y = df["label"]

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test


def build_tfidf_vectorizer(
    max_features: Optional[int] = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
    token_pattern: str = r"(?u)[\w\u0980-\u09ff]+|<\w+>",
) -> TfidfVectorizer:
    """
    Create a TF-IDF text vectorizer configured for Bengali SMS text.

    Preserves special entity tokens (<URL>, <PHONE>, <MONEY>, <NUMBER>, <EMAIL>)
    along with full Bengali Unicode word tokens (including matras/diacritics)
    and sublinear term frequency scaling.

    Args:
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Lower and upper boundary of n-grams (unigrams + bigrams).
        min_df: Minimum document frequency for terms.
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        token_pattern: Regex pattern capturing Bengali Unicode words and entity tokens.

    Returns:
        Configured TfidfVectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        token_pattern=token_pattern,
    )
    return vectorizer



def train_baseline_model(
    X_train: Union[pd.Series, List[str]],
    y_train: Union[pd.Series, List[int], np.ndarray],
    vectorizer: Optional[TfidfVectorizer] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[Union[str, dict]] = "balanced",
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Fit the TF-IDF vectorizer on training text and train a Logistic Regression classifier.

    Args:
        X_train: Training SMS text samples.
        y_train: Training labels (0 for ham, 1 for spam).
        vectorizer: Optional existing or custom TfidfVectorizer.
        C: Inverse of regularization strength.
        max_iter: Maximum number of solver iterations.
        class_weight: Weights associated with classes ('balanced' or None).
        random_state: Random state seed.

    Returns:
        Tuple of (fitted_vectorizer, trained_logistic_regression_model).
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)

    return vectorizer, model


def evaluate_baseline_model(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    X_test: Union[pd.Series, List[str]],
    y_test: Union[pd.Series, List[int], np.ndarray],
) -> Dict[str, Any]:
    """
    Evaluate the trained Logistic Regression model on the test dataset.

    Computes:
    - Accuracy
    - Precision (binary spam class and macro/weighted)
    - Recall (binary spam class and macro/weighted)
    - F1-score (binary spam class and macro/weighted)
    - Confusion Matrix (TN, FP, FN, TP)
    - Detailed Scikit-Learn Classification Report

    Args:
        model: Trained LogisticRegression model.
        vectorizer: Fitted TfidfVectorizer.
        X_test: Test text samples.
        y_test: True test labels.

    Returns:
        Dictionary of calculated evaluation metrics and confusion matrix.
    """
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)

    y_test_arr = np.array(y_test)

    # Core binary metrics for Spam (class 1)
    accuracy = float(accuracy_score(y_test_arr, y_pred))
    precision = float(precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0))

    # Multi-class / macro averages
    macro_precision = float(precision_score(y_test_arr, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test_arr, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_arr, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_arr, y_pred, average="weighted", zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Classification report
    report_dict = classification_report(
        y_test_arr,
        y_pred,
        target_names=["Ham (0)", "Spam (1)"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test_arr,
        y_pred,
        target_names=["Ham (0)", "Spam (1)"],
        digits=4,
        zero_division=0,
    )

    metrics = {
        "test_samples": int(len(y_test_arr)),
        "ham_samples": int((y_test_arr == 0).sum()),
        "spam_samples": int((y_test_arr == 1).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }

    return metrics


def print_evaluation_results(metrics: Dict[str, Any], title: str = "Baseline Evaluation Results") -> None:
    """
    Print a cleanly formatted evaluation summary to the console.

    Args:
        metrics: Metrics dictionary produced by evaluate_baseline_model.
        title: Header banner title.
    """
    cm = metrics["confusion_matrix"]

    print("=" * 65)
    print(f"📊 {title.upper()}")
    print("=" * 65)
    print(f"• Total Test Samples: {metrics['test_samples']} (Ham: {metrics['ham_samples']}, Spam: {metrics['spam_samples']})")
    print("-" * 65)
    print("KEY PERFORMANCE METRICS (Spam Detection / Class 1):")
    print(f"  • Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  • Precision:       {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  • Recall:          {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  • F1-Score:        {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print(f"  • Macro F1:        {metrics['macro_f1']:.4f} ({metrics['macro_f1'] * 100:.2f}%)")
    print(f"  • Weighted F1:     {metrics['weighted_f1']:.4f} ({metrics['weighted_f1'] * 100:.2f}%)")
    print("-" * 65)
    print("CONFUSION MATRIX:")
    print(f"                   Predicted HAM (0)    Predicted SPAM (1)")
    print(f"  Actual HAM  (0)  {cm['tn']:<20} {cm['fp']:<20}")
    print(f"  Actual SPAM (1)  {cm['fn']:<20} {cm['tp']:<20}")
    print("-" * 65)
    print("FULL CLASSIFICATION REPORT:")
    print(metrics.get("classification_report_text", "").strip())
    print("=" * 65)


def save_baseline_artifacts(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    metrics: Optional[Dict[str, Any]] = None,
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
) -> Dict[str, Path]:
    """
    Save the trained TF-IDF vectorizer and Logistic Regression model to disk.

    Args:
        vectorizer: Trained TfidfVectorizer.
        model: Trained LogisticRegression model.
        metrics: Optional evaluation metrics dictionary to persist.
        model_dir: Target directory (default: 'models').

    Returns:
        Dictionary mapping artifact names to their saved file Paths.
    """
    dir_path = Path(model_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    vectorizer_path = dir_path / DEFAULT_VECTORIZER_FILENAME
    model_path = dir_path / DEFAULT_MODEL_FILENAME
    metadata_path = dir_path / DEFAULT_METADATA_FILENAME

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "ngram_range": list(vectorizer.ngram_range),
        "model_params": {
            "C": float(model.C),
            "max_iter": int(model.max_iter),
            "class_weight": model.class_weight,
            "solver": model.solver,
        },
        "metrics": metrics if metrics else {},
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "vectorizer": vectorizer_path,
        "model": model_path,
        "metadata": metadata_path,
    }


def load_baseline_artifacts(
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    """
    Load the saved TF-IDF vectorizer, Logistic Regression model, and metadata.

    Args:
        model_dir: Directory containing the saved model artifacts.

    Returns:
        Tuple of (vectorizer, model, metadata_dict).

    Raises:
        FileNotFoundError: If model or vectorizer files are missing.
    """
    dir_path = Path(model_dir)
    vectorizer_path = dir_path / DEFAULT_VECTORIZER_FILENAME
    model_path = dir_path / DEFAULT_MODEL_FILENAME
    metadata_path = dir_path / DEFAULT_METADATA_FILENAME

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return vectorizer, model, metadata


def predict_sms(
    text: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    preprocessor: Optional[BengaliTextPreprocessor] = None,
    is_preprocessed: bool = False,
) -> Dict[str, Any]:
    """
    Reusable prediction function for a single Bengali SMS string.

    Args:
        text: Bengali SMS message (raw or preprocessed).
        vectorizer: Fitted TfidfVectorizer.
        model: Trained LogisticRegression model.
        preprocessor: Optional BengaliTextPreprocessor instance.
        is_preprocessed: If True, skips text preprocessing.

    Returns:
        Dictionary containing:
        - original_text: Original raw input text
        - processed_text: Preprocessed text fed to the vectorizer
        - label: 'spam' or 'ham'
        - label_id: 1 for spam, 0 for ham
        - confidence: Confidence score for the predicted class (0.0 to 1.0)
        - spam_probability: Probability of the SMS being spam
        - ham_probability: Probability of the SMS being ham
    """
    raw_text = "" if text is None else str(text)

    if is_preprocessed:
        proc_text = raw_text
    else:
        proc = preprocessor or BengaliTextPreprocessor()
        proc_text = proc.preprocess(raw_text)

    # Transform text to TF-IDF feature vector
    # If text became empty after preprocessing, use raw_text fallback to avoid empty vector
    text_to_vectorize = proc_text if proc_text.strip() != "" else raw_text
    vec = vectorizer.transform([text_to_vectorize])

    # Predict class & probabilities
    pred_label_id = int(model.predict(vec)[0])
    probs = model.predict_proba(vec)[0]

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    label_str = "spam" if pred_label_id == 1 else "ham"
    confidence = spam_prob if pred_label_id == 1 else ham_prob

    return {
        "original_text": raw_text,
        "processed_text": proc_text,
        "label": label_str,
        "label_id": pred_label_id,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
    }


class BaselinePredictor:
    """
    Modular object-oriented interface for Bengali SMS Spam prediction.
    """

    def __init__(
        self,
        vectorizer: Optional[TfidfVectorizer] = None,
        model: Optional[LogisticRegression] = None,
        preprocessor: Optional[BengaliTextPreprocessor] = None,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    ):
        """
        Initialize the baseline predictor. If vectorizer or model are omitted,
        they will be loaded from model_dir.
        """
        if vectorizer is None or model is None:
            loaded_vec, loaded_model, _ = load_baseline_artifacts(model_dir)
            self.vectorizer = vectorizer or loaded_vec
            self.model = model or loaded_model
        else:
            self.vectorizer = vectorizer
            self.model = model

        self.preprocessor = preprocessor or BengaliTextPreprocessor()

    @classmethod
    def from_saved(cls, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> "BaselinePredictor":
        """Load predictor from saved model directory."""
        return cls(model_dir=model_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict spam/ham status for a single Bengali SMS message.
        """
        return predict_sms(
            text=text,
            vectorizer=self.vectorizer,
            model=self.model,
            preprocessor=self.preprocessor,
        )

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict spam/ham status for a batch of Bengali SMS messages.
        """
        return [self.predict(t) for t in texts]


def train_and_evaluate_pipeline(
    dataset_path: Optional[Union[str, Path]] = None,
    model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    save_models: bool = True,
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    """
    End-to-end execution of baseline model training and evaluation:
    1. Load processed dataset.
    2. Stratified 80/20 train/test split.
    3. Construct Bengali TF-IDF vectorizer and fit on train data.
    4. Train Logistic Regression classifier.
    5. Evaluate on test set.
    6. Save models and metadata to model_dir.
    7. Print comprehensive evaluation report.

    Args:
        dataset_path: Optional path to processed dataset CSV.
        model_dir: Target directory to save trained model artifacts.
        test_size: Fraction of test split (default 0.20).
        random_state: Random state seed.
        max_features: Max features for TF-IDF.
        ngram_range: N-gram range for TF-IDF.
        save_models: Whether to save artifacts to disk.

    Returns:
        Tuple of (vectorizer, model, metrics_dict).
    """
    print("📁 Loading processed dataset...")
    df = load_processed_dataset(dataset_path)
    total_samples = len(df)
    spam_samples = int((df["label"] == 1).sum())
    ham_samples = int((df["label"] == 0).sum())
    print(f"  * Total Samples: {total_samples:,}")
    print(f"  * Ham (0): {ham_samples:,} ({ham_samples/total_samples*100:.1f}%)")
    print(f"  * Spam (1): {spam_samples:,} ({spam_samples/total_samples*100:.1f}%)")

    # Stratified split
    print(f"\n✂️ Performing stratified train/test split (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = split_dataset(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    print(f"  * Training set: {len(X_train):,} samples")
    print(f"  * Test set:     {len(X_test):,} samples")

    # Vectorizer and Model training
    print("\n⚙️ Vectorizing with TF-IDF and training Logistic Regression classifier...")
    vec = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    vectorizer, model = train_baseline_model(
        X_train=X_train,
        y_train=y_train,
        vectorizer=vec,
        random_state=random_state,
    )
    print(f"  * TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,} features")
    print("  * Logistic Regression converged successfully.")

    # Evaluation
    print("\n📈 Evaluating baseline model on test dataset...")
    metrics = evaluate_baseline_model(
        model=model,
        vectorizer=vectorizer,
        X_test=X_test,
        y_test=y_test,
    )

    print_evaluation_results(metrics, title="BanglaGuard Baseline NLP Model Evaluation (TF-IDF + Logistic Regression)")

    # Save artifacts
    if save_models:
        saved_paths = save_baseline_artifacts(
            vectorizer=vectorizer,
            model=model,
            metrics=metrics,
            model_dir=model_dir,
        )
        print(f"\n💾 Model artifacts saved successfully inside '{model_dir}':")
        for k, v in saved_paths.items():
            print(f"  * {k:<12}: {v.resolve()}")

    return vectorizer, model, metrics
