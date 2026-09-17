"""
Training package for BanglaGuard.

Contains NLP text preprocessors, data preparation pipelines, and model training utilities.
"""

from training.preprocessor import BengaliTextPreprocessor, preprocess_bengali_text
from training.data_prep import (
    locate_dataset,
    load_raw_dataset,
    analyze_dataset,
    clean_dataset,
    preprocess_dataset,
    prepare_and_preprocess_pipeline,
)
from training.baseline_model import (
    locate_processed_dataset,
    load_processed_dataset,
    split_dataset,
    build_tfidf_vectorizer,
    train_baseline_model,
    evaluate_baseline_model,
    print_evaluation_results,
    save_baseline_artifacts,
    load_baseline_artifacts,
    predict_sms,
    BaselinePredictor,
    train_and_evaluate_pipeline,
)

__all__ = [
    "BengaliTextPreprocessor",
    "preprocess_bengali_text",
    "locate_dataset",
    "load_raw_dataset",
    "analyze_dataset",
    "clean_dataset",
    "preprocess_dataset",
    "prepare_and_preprocess_pipeline",
    "locate_processed_dataset",
    "load_processed_dataset",
    "split_dataset",
    "build_tfidf_vectorizer",
    "train_baseline_model",
    "evaluate_baseline_model",
    "print_evaluation_results",
    "save_baseline_artifacts",
    "load_baseline_artifacts",
    "predict_sms",
    "BaselinePredictor",
    "train_and_evaluate_pipeline",
]

