"""
Data Preparation and Cleaning Module for BanglaGuard.

Handles:
1. Dataset loading and automatic path resolution.
2. Comprehensive exploratory dataset analysis (counts, missing, duplicates, text lengths).
3. Data cleaning (null removal, deduplication, label normalization/encoding).
4. Full text preprocessing integration with BengaliTextPreprocessor.
5. Exporting processed datasets to the data directory.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd

from training.preprocessor import BengaliTextPreprocessor, preprocess_bengali_text


# Default search locations for raw dataset
CANDIDATE_DATASET_PATHS = [
    Path("bangla_dataset.csv"),
    Path("dataset.csv"),
    Path("data/bangla_dataset.csv"),
    Path("data/dataset.csv"),
    Path("data/raw/bangla_dataset.csv"),
    Path("data/raw/dataset.csv"),
]

# Default output locations for processed dataset
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PROCESSED_FILE = DEFAULT_PROCESSED_DIR / "processed_dataset.csv"
FALLBACK_PROCESSED_FILE = Path("data/processed_dataset.csv")


def locate_dataset(dataset_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locate the raw dataset file, checking user-supplied path or common defaults.

    Args:
        dataset_path: Optional explicit path to the dataset.

    Returns:
        Resolved Path to existing dataset file.

    Raises:
        FileNotFoundError: If the dataset file cannot be found.
    """
    if dataset_path:
        p = Path(dataset_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset path not found: {dataset_path}")

    # Check candidates
    for candidate in CANDIDATE_DATASET_PATHS:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate dataset file. Checked: {[str(c) for c in CANDIDATE_DATASET_PATHS]}"
    )


def load_raw_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the raw CSV dataset with robust UTF-8 encoding.

    Args:
        file_path: Path to raw CSV dataset.

    Returns:
        Loaded DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Attempt utf-8, fallback to utf-8-sig if BOM present
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig")

    return df


def analyze_dataset(df: pd.DataFrame, title: str = "Dataset Analysis") -> Dict:
    """
    Perform statistical and structural analysis of the dataset.

    Args:
        df: DataFrame to analyze.
        title: Header title for the analysis output.

    Returns:
        Dictionary containing extracted statistical metrics.
    """
    total_records = len(df)
    missing_type = int(df["type"].isna().sum()) if "type" in df.columns else 0
    missing_text = int(df["text"].isna().sum()) if "text" in df.columns else 0
    total_missing_rows = int(df.isna().any(axis=1).sum())

    # Check duplicates in text column
    duplicate_texts = int(df["text"].dropna().duplicated().sum()) if "text" in df.columns else 0
    total_duplicate_rows = int(df.duplicated().sum())

    # Class breakdown
    type_counts = {}
    if "type" in df.columns:
        norm_types = df["type"].astype(str).str.strip().str.lower()
        type_counts = norm_types.value_counts().to_dict()

    # Text length statistics
    text_stats = {}
    if "text" in df.columns:
        valid_texts = df["text"].dropna().astype(str)
        if not valid_texts.empty:
            char_lens = valid_texts.str.len()
            word_lens = valid_texts.str.split().str.len()

            text_stats = {
                "char_length": {
                    "min": int(char_lens.min()),
                    "max": int(char_lens.max()),
                    "mean": float(round(char_lens.mean(), 2)),
                    "median": float(round(char_lens.median(), 2)),
                },
                "word_count": {
                    "min": int(word_lens.min()),
                    "max": int(word_lens.max()),
                    "mean": float(round(word_lens.mean(), 2)),
                    "median": float(round(word_lens.median(), 2)),
                },
            }

    stats = {
        "title": title,
        "total_records": total_records,
        "missing_type": missing_type,
        "missing_text": missing_text,
        "total_missing_rows": total_missing_rows,
        "duplicate_texts": duplicate_texts,
        "total_duplicate_rows": total_duplicate_rows,
        "class_counts": type_counts,
        "text_stats": text_stats,
    }

    return stats


def print_dataset_analysis(stats: Dict) -> None:
    """Print a nicely formatted analysis report to console."""
    print("=" * 60)
    print(f"[DATASET ANALYSIS] {stats.get('title', 'Dataset Analysis')}")
    print("=" * 60)
    print(f"- Total Records:          {stats['total_records']}")
    print(f"- Missing Type Values:    {stats['missing_type']}")
    print(f"- Missing Text Values:    {stats['missing_text']}")
    print(f"- Total Missing Rows:     {stats['total_missing_rows']}")
    print(f"- Duplicate SMS Texts:    {stats['duplicate_texts']}")
    print(f"- Total Exact Duplicates: {stats['total_duplicate_rows']}")

    print("\n[CLASS DISTRIBUTION]")
    for label, count in stats.get("class_counts", {}).items():
        pct = (count / stats["total_records"] * 100) if stats["total_records"] > 0 else 0
        print(f"  * {label:<8}: {count:>5} ({pct:.2f}%)")

    tstats = stats.get("text_stats", {})
    if tstats:
        print("\n[SMS TEXT LENGTH STATISTICS]")
        c_stat = tstats.get("char_length", {})
        w_stat = tstats.get("word_count", {})
        print(f"  * Character Length: Min={c_stat.get('min')}, Max={c_stat.get('max')}, Mean={c_stat.get('mean')}, Median={c_stat.get('median')}")
        print(f"  * Word Count:       Min={w_stat.get('min')}, Max={w_stat.get('max')}, Mean={w_stat.get('mean')}, Median={w_stat.get('median')}")
    print("=" * 60)


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the dataset:
    - Drop missing or empty SMS texts and types.
    - Remove duplicate SMS messages.
    - Normalize type column (strip, lower).
    - Map spam -> 1 and ham -> 0.

    Args:
        df: Raw DataFrame with 'type' and 'text' columns.

    Returns:
        Tuple of (cleaned_df, cleaning_metrics_dict).
    """
    initial_count = len(df)

    # Make working copy
    cleaned = df.copy()

    # Ensure required columns exist
    if "type" not in cleaned.columns or "text" not in cleaned.columns:
        raise ValueError(f"Dataset must contain 'type' and 'text' columns. Found: {list(cleaned.columns)}")

    # 1. Remove rows with missing or whitespace-only text or type
    cleaned = cleaned.dropna(subset=["text", "type"])
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["type"] = cleaned["type"].astype(str).str.strip().str.lower()

    # Drop empty strings after strip
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned[cleaned["type"] != ""]
    missing_removed = initial_count - len(cleaned)

    # 2. Remove duplicate SMS messages based on text content
    count_before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["text"]).reset_index(drop=True)
    duplicates_removed = count_before_dedup - len(cleaned)

    # 3. Filter valid types and encode label (spam -> 1, ham -> 0)
    cleaned = cleaned[cleaned["type"].isin(["spam", "ham"])].copy()

    label_mapping = {"ham": 0, "spam": 1}
    cleaned["label"] = cleaned["type"].map(label_mapping).astype(int)

    # Rename text to original_text for clarity
    cleaned["original_text"] = cleaned["text"]

    metrics = {
        "initial_count": initial_count,
        "missing_removed": missing_removed,
        "duplicates_removed": duplicates_removed,
        "cleaned_count": len(cleaned),
        "spam_count": int((cleaned["label"] == 1).sum()),
        "ham_count": int((cleaned["label"] == 0).sum()),
    }

    return cleaned, metrics


def preprocess_dataset(
    df: pd.DataFrame,
    preprocessor: Optional[BengaliTextPreprocessor] = None,
) -> pd.DataFrame:
    """
    Apply NLP preprocessing to the cleaned DataFrame.

    Preserves:
    - 'original_text'
    - 'processed_text'
    - 'type'
    - 'label'

    Args:
        df: Cleaned DataFrame.
        preprocessor: Optional BengaliTextPreprocessor instance.

    Returns:
        Processed DataFrame with 'processed_text' column added.
    """
    proc = preprocessor if preprocessor is not None else BengaliTextPreprocessor()

    df_proc = df.copy()
    df_proc["processed_text"] = df_proc["original_text"].apply(proc.preprocess)

    # Remove any rows where processed_text became empty (if any)
    df_proc = df_proc[df_proc["processed_text"].str.strip() != ""].reset_index(drop=True)

    # Reorder columns logically for downstream modeling
    cols = ["type", "label", "original_text", "processed_text"]
    other_cols = [c for c in df_proc.columns if c not in cols and c != "text"]
    df_proc = df_proc[cols + other_cols]

    return df_proc


def prepare_and_preprocess_pipeline(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    preprocessor: Optional[BengaliTextPreprocessor] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the end-to-end dataset preparation & preprocessing pipeline:
    1. Locate and load raw dataset.
    2. Perform and display initial exploratory analysis.
    3. Clean data (handle nulls, duplicates, label encoding).
    4. Apply Bengali NLP preprocessing.
    5. Save processed dataset to output path (defaulting to data/processed/).
    6. Print comprehensive summary report.

    Args:
        input_path: Optional path to raw dataset CSV.
        output_path: Optional path where processed CSV will be saved.
        preprocessor: Optional custom preprocessor instance.

    Returns:
        Tuple of (processed_df, summary_metrics).
    """
    # 1. Locate dataset
    resolved_input_path = locate_dataset(input_path)
    print(f"[INFO] Loading raw dataset from: {resolved_input_path.resolve()}")
    raw_df = load_raw_dataset(resolved_input_path)

    # 2. Initial analysis
    initial_stats = analyze_dataset(raw_df, title="Raw Dataset Initial Analysis")
    print_dataset_analysis(initial_stats)

    # 3. Clean dataset
    print("\n[INFO] Cleaning dataset...")
    cleaned_df, clean_metrics = clean_dataset(raw_df)
    print(f"  * Removed {clean_metrics['missing_removed']} missing/empty rows")
    print(f"  * Removed {clean_metrics['duplicates_removed']} duplicate SMS messages")

    # 4. NLP Preprocessing
    print("\n[INFO] Running Bengali NLP Preprocessing (entity tokens, unicode normalization)...")
    proc = preprocessor or BengaliTextPreprocessor()
    processed_df = preprocess_dataset(cleaned_df, preprocessor=proc)
    print(f"  * Preprocessed {len(processed_df)} messages")

    # 5. Save output
    target_out = Path(output_path) if output_path else DEFAULT_PROCESSED_FILE
    target_out.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(target_out, index=False, encoding="utf-8")
    print(f"\n[INFO] Saved processed dataset to: {target_out.resolve()}")

    # Also save a copy to FALLBACK_PROCESSED_FILE if distinct
    if target_out.resolve() != FALLBACK_PROCESSED_FILE.resolve():
        FALLBACK_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(FALLBACK_PROCESSED_FILE, index=False, encoding="utf-8")
        print(f"[INFO] Saved duplicate copy to: {FALLBACK_PROCESSED_FILE.resolve()}")

    # 6. Final summary metrics
    summary = {
        "input_file": str(resolved_input_path),
        "output_file": str(target_out),
        "original_dataset_size": initial_stats["total_records"],
        "final_dataset_size": len(processed_df),
        "spam_messages": int((processed_df["label"] == 1).sum()),
        "ham_messages": int((processed_df["label"] == 0).sum()),
        "duplicates_removed": clean_metrics["duplicates_removed"],
        "missing_rows_removed": clean_metrics["missing_removed"],
    }

    print_final_summary(summary)
    return processed_df, summary


def print_final_summary(summary: Dict) -> None:
    """Print the final pipeline execution summary."""
    print("\n" + "=" * 60)
    print("BANGLAGUARD DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"- Input Dataset:          {summary['input_file']}")
    print(f"- Output Dataset:         {summary['output_file']}")
    print(f"- Original Dataset Size:  {summary['original_dataset_size']:,} records")
    print(f"- Missing Rows Removed:   {summary['missing_rows_removed']:,}")
    print(f"- Duplicates Removed:     {summary['duplicates_removed']:,}")
    print(f"- Final Dataset Size:     {summary['final_dataset_size']:,} records")
    print(f"  * Spam Messages (1):    {summary['spam_messages']:,} ({summary['spam_messages']/summary['final_dataset_size']*100:.2f}%)")
    print(f"  * Ham Messages (0):     {summary['ham_messages']:,} ({summary['ham_messages']/summary['final_dataset_size']*100:.2f}%)")
    print("=" * 60)
    print("[SUCCESS] Bengali NLP data preparation and preprocessing complete!\n")
