import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Set UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from training.baseline_model import BaselinePredictor, load_baseline_artifacts
from training.preprocessor import BengaliTextPreprocessor



def analyze_target_message(text: str):
    print("=" * 70)
    print("1. INFERENCE & FEATURE CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    preprocessor = BengaliTextPreprocessor()
    proc_text = preprocessor.preprocess(text)
    print(f"• Original SMS:     '{text}'")
    print(f"• Preprocessed SMS: '{proc_text}'")

    predictor = BaselinePredictor.from_saved("models")
    res = predictor.predict(text)
    
    print("\n--- Model Prediction Output ---")
    print(f"• Predicted Label:   {res['label'].upper()} (Label ID: {res['label_id']})")
    print(f"• Confidence Score:  {res['confidence'] * 100:.2f}%")
    print(f"• Spam Probability:  {res['spam_probability'] * 100:.2f}%")
    print(f"• Ham Probability:   {res['ham_probability'] * 100:.2f}%")

    vectorizer, model, metadata = load_baseline_artifacts("models")
    vec = vectorizer.transform([proc_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    print(f"\n• Model Intercept (Base Bias): {intercept:.4f}")

    # Extract non-zero features
    _, cols = vec.nonzero()
    rows = []
    for c in cols:
        feat_name = feature_names[c]
        tfidf_val = float(vec[0, c])
        coef = float(coefs[c])
        contrib = float(tfidf_val * coef)
        rows.append({
            "Feature": feat_name,
            "TF-IDF Value": round(tfidf_val, 4),
            "Coefficient": round(coef, 4),
            "Contribution": round(contrib, 4),
            "Direction": "SPAM (Positive)" if contrib > 0 else "HAM (Negative)"
        })

    df_feats = pd.DataFrame(rows)
    if not df_feats.empty:
        df_feats = df_feats.sort_values(by="Contribution", ascending=False)
        print("\n--- Present Features & Contributions (Ranked) ---")
        print(df_feats.to_string(index=False))

    total_contrib = df_feats["Contribution"].sum() if not df_feats.empty else 0.0
    linear_score = intercept + total_contrib
    manual_prob = 1.0 / (1.0 + np.exp(-linear_score))

    print(f"\n• Total Linear Logit: {intercept:.4f} + {total_contrib:.4f} = {linear_score:.4f}")
    print(f"• Calculated Probability: 1 / (1 + exp(-{linear_score:.4f})) = {manual_prob:.4f}")

    return proc_text, df_feats


def analyze_dataset_patterns():
    print("\n" + "=" * 70)
    print("2. DATASET PATTERN & KEYWORD DISTRIBUTION ANALYSIS")
    print("=" * 70)

    dataset_path = Path("data/processed/processed_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/processed_dataset.csv")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"• Total Processed Dataset Size: {len(df):,} messages")
    print(f"• Class Breakdown: Ham = {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean()*100:.1f}%), Spam = {(df['label'] == 1).sum():,} ({(df['label'] == 1).mean()*100:.1f}%)")

    # Keywords to analyze
    target_keywords = [
        "recharge", "রিচার্জ", "dth", "tv", "dear", "customer", 
        "success", "rs", "টাকা", "<money>", "<number>"
    ]

    kw_stats = []
    for kw in target_keywords:
        match = df["processed_text"].str.contains(kw, case=False, na=False) | df["original_text"].str.contains(kw, case=False, na=False)
        sub = df[match]
        total_m = len(sub)
        spam_m = int((sub["label"] == 1).sum())
        ham_m = int((sub["label"] == 0).sum())
        spam_pct = (spam_m / total_m * 100) if total_m > 0 else 0.0
        kw_stats.append({
            "Keyword": kw,
            "Total In Dataset": total_m,
            "Spam Count": spam_m,
            "Ham Count": ham_m,
            "Spam %": f"{spam_pct:.1f}%"
        })

    df_kw = pd.DataFrame(kw_stats)
    print("\n--- Keyword Class Distribution in Training Data ---")
    print(df_kw.to_string(index=False))

    # Examine exact or close matches to recharge/transactional confirmations
    print("\n--- Examples of Similar Recharge / Transaction Messages in Dataset ---")
    similar_mask = df["original_text"].str.contains("recharge|রিচার্জ|dth|টাকা", case=False, na=False)
    similar_samples = df[similar_mask].head(10)
    for idx, row in similar_samples.iterrows():
        print(f"[{row['type'].upper():<4}] {row['original_text']}")


if __name__ == "__main__":
    target_msg = "Dear Customer You DTH tV recharge is successfull Rs 322"
    analyze_target_message(target_msg)
    analyze_dataset_patterns()
