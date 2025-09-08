# CREDIT DEFAULT RISK — END‑TO‑END (2 PARTS IN ONE FILE)
# -----------------------------------------------------
# This script keeps ALL your functionality but enforces ML hygiene:
#   • PART 1  = EDA + cleaning + safe conversions + reports (NO fitting on full data)
#   • PART 2  = Split → fit preprocessors/rare-bins on TRAIN only → train models → SHAP
# You can run cells top‑to‑bottom, or run PART 1, inspect artifacts, then PART 2.
# (Inspired by Aytug Akarlar's approach: XGBoost + SHAP for explainability.)

# =====================
# Common dependencies
# =====================
import os
import json
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from sklearn.ensemble import RandomForestClassifier

# Optional (XGBoost + SHAP)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

import matplotlib.pyplot as plt

RAW_CSV = 'Case_Data.csv'
ART_DIR = 'artifacts'
OUT_DIR = 'processed_data'
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'TARGET'
MISSING_STRINGS = ["NA", "N/A", "na", "n/a", "NaN", "nan", "", " "]
RARE_THRESHOLD = 0.02
HIGH_CARDINALITY = 8

# =====================
# Utilities (reused)
# =====================

def fit_rare_maps(dframe: pd.DataFrame, cols: list, thresh: float = RARE_THRESHOLD):
    """Return dict[col] -> list of kept categories (>= threshold)."""
    maps = {}
    for c in cols:
        freq = dframe[c].value_counts(normalize=True, dropna=False)
        keep = [val for val, p in freq.items() if p >= thresh]
        maps[c] = keep
    return maps


def apply_rare_maps(dframe: pd.DataFrame, maps: dict):
    for c, keep in maps.items():
        dframe[c] = np.where(dframe[c].isin(keep), dframe[c], 'Other')
    return dframe


def build_preprocessor(numerical_features, categorical_features):
    """ColumnTransformer with train‑only fit behavior handled by caller."""
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# =================================================
# PART 1 — EDA, CLEANING, TEMPORAL CONVERSIONS ONLY
# =================================================
print('\n=== PART 1: EDA & CLEANING (no model fitting) ===')

df = pd.read_csv(RAW_CSV)
print('Raw shape:', df.shape)

# 1) Deduplicate & standardize obvious stringy-missings
working_dataframe = df.drop_duplicates().reset_index(drop=True).copy()
working_dataframe.replace(MISSING_STRINGS, np.nan, inplace=True)

# 2) DAYS_* → numeric + sentinel handling + YEARS
DAYS_COLS = [c for c in working_dataframe.columns if c.startswith('DAYS_')]
for c in DAYS_COLS:
    working_dataframe[c] = pd.to_numeric(working_dataframe[c], errors='coerce')

if 'DAYS_EMPLOYED' in working_dataframe.columns:
    EMP_SENTINEL = 365243
    emp_mask = working_dataframe['DAYS_EMPLOYED'].abs() == EMP_SENTINEL
    working_dataframe['EMPLOYMENT_SENTINEL_FLAG'] = emp_mask.astype(int)
    if emp_mask.any():
        print(f"Converting DAYS_EMPLOYED sentinel {EMP_SENTINEL} to NaN on {int(emp_mask.sum())} rows")
        working_dataframe.loc[emp_mask, 'DAYS_EMPLOYED'] = np.nan

# Convert temporal columns to interpretable years
if 'DAYS_BIRTH' in working_dataframe.columns:
    working_dataframe['AGE_YEARS'] = (-working_dataframe['DAYS_BIRTH'] / 365.25).round(3)
if 'DAYS_EMPLOYED' in working_dataframe.columns:
    working_dataframe['EMPLOYMENT_YEARS'] = (-working_dataframe['DAYS_EMPLOYED'] / 365.25).round(3)
if 'DAYS_ID_PUBLISH' in working_dataframe.columns:
    working_dataframe['ID_YEARS'] = (-working_dataframe['DAYS_ID_PUBLISH'] / 365.25).round(3)
if 'DAYS_LAST_PHONE_CHANGE' in working_dataframe.columns:
    working_dataframe['PHONE_CHANGE_YEARS'] = (-working_dataframe['DAYS_LAST_PHONE_CHANGE'] / 365.25).round(3)
if 'DAYS_REGISTRATION' in working_dataframe.columns:
    working_dataframe['REGISTRATION_YEARS'] = (-working_dataframe['DAYS_REGISTRATION'] / 365.25).round(3)

# Optional missingness indicators (these are allowed before split — they are deterministic flags)
for col in ['AGE_YEARS','EMPLOYMENT_YEARS','ID_YEARS','PHONE_CHANGE_YEARS','REGISTRATION_YEARS']:
    if col in working_dataframe.columns:
        working_dataframe[f'{col}_MISSING'] = working_dataframe[col].isna().astype(int)

# 3) Drop original DAYS_* to avoid confusion
working_dataframe.drop(columns=[c for c in DAYS_COLS if c in working_dataframe.columns], errors='ignore', inplace=True)

# 4) Replace any Infs from pre-existing ratios
working_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

# 5) Types & simple EDA snapshots
categorical_features = working_dataframe.select_dtypes(include=['object','category']).columns.tolist()
numerical_features   = working_dataframe.select_dtypes(include=['number']).columns.tolist()
if TARGET in categorical_features: categorical_features.remove(TARGET)
if TARGET in numerical_features:   numerical_features.remove(TARGET)
print(f"Categorical: {len(categorical_features)} | Numeric: {len(numerical_features)}")

# Missingness BEFORE imputation (expected > 0)
print('Nulls BEFORE imputation:', int(working_dataframe.isna().sum().sum()))

# Cardinality snapshot for planning (no fitting yet)
cat_unique_counts_df = (
    pd.DataFrame({
        'column': categorical_features,
        'unique_count': [working_dataframe[c].nunique(dropna=False) for c in categorical_features]
    })
    .sort_values('unique_count', ascending=False)
    .reset_index(drop=True)
)
print('\nTop 15 categorical by cardinality:')
print(cat_unique_counts_df.head(15))

# Identify high-cardinality candidates for rare-binning (fit later on TRAIN)
high_card_cols = cat_unique_counts_df.query('unique_count > @HIGH_CARDINALITY')['column'].tolist()
print('High-cardinality columns (plan rare-binning in Part 2):', high_card_cols)

# 6) Save the CLEANED but UN-FITTED frame for transparency & reproducibility
clean_path = os.path.join(OUT_DIR, 'cleaned_for_split.csv')
working_dataframe.to_csv(clean_path, index=False)
print('Saved cleaned (no imputation/encoding):', clean_path, '| Shape:', working_dataframe.shape)

# Save metadata for Part 2
meta = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'high_card_cols': high_card_cols,
    'rare_threshold': RARE_THRESHOLD,
    'target': TARGET,
}
with open(os.path.join(OUT_DIR, 'part1_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print('Saved Part 1 metadata → processed_data/part1_meta.json')

print('\n=== PART 1 COMPLETE — proceed to PART 2 when ready. ===')

# ============================================================
# PART 2 — SPLIT, TRAIN‑ONLY FIT, TRANSFORM, MODEL, EXPLAIN
# ============================================================
print('\n=== PART 2 — SPLIT, TRAIN‑ONLY FIT, TRANSFORM, MODEL, EXPLAIN ===')

# 0) Load artifacts from Part 1
part1_df = pd.read_csv(clean_path)
with open(os.path.join(OUT_DIR, 'part1_meta.json'), 'r') as f:
    meta = json.load(f)

categorical_features = meta['categorical_features']
numerical_features = meta['numerical_features']
high_card_cols = meta['high_card_cols']

# 1) Split FIRST (to avoid leakage)
X_full = part1_df.drop(columns=[c for c in [TARGET] if c in part1_df.columns])
y_full = part1_df[TARGET].copy() if TARGET in part1_df.columns else None

if y_full is None:
    raise ValueError('TARGET column not found. Ensure your raw data includes TARGET.')

X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print('Splits:', X_train.shape, X_valid.shape, X_test.shape)

# 2) Fit rare-binning maps on TRAIN only, then apply to valid/test
rare_maps = fit_rare_maps(pd.concat([X_train[categorical_features], y_train], axis=1), high_card_cols, RARE_THRESHOLD) if high_card_cols else {}
if rare_maps:
    X_train = apply_rare_maps(X_train, rare_maps)
    X_valid = apply_rare_maps(X_valid, rare_maps)
    X_test  = apply_rare_maps(X_test,  rare_maps)
    with open(os.path.join(OUT_DIR, 'rare_maps.json'), 'w') as f:
        json.dump({k: list(v) for k, v in rare_maps.items()}, f, indent=2)
    print('Fitted rare maps on TRAIN and applied to VALID/TEST.')

# 3) Build & FIT preprocessors on TRAIN only, then transform all
preprocessor = build_preprocessor(numerical_features, categorical_features)

Xtr_enc = preprocessor.fit_transform(X_train)
Xva_enc = preprocessor.transform(X_valid)
Xte_enc = preprocessor.transform(X_test)

# Build feature names for inspection
try:
    num_names_out = preprocessor.named_transformers_["num"].named_steps["imputer"].get_feature_names_out(numerical_features).tolist()
except Exception:
    num_names_out = numerical_features
cat_names_out = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features).tolist()
feature_names = num_names_out + cat_names_out

# Sanity checks
for name, mat in [('TRAIN', Xtr_enc), ('VALID', Xva_enc), ('TEST', Xte_enc)]:
    n_nans = int(np.isnan(mat).sum())
    n_infs = int(np.isinf(mat).sum())
    print(f"{name} → Nulls: {n_nans} | Infs: {n_infs}")
    assert n_nans == 0 and n_infs == 0, f"{name} has NaNs/Infs after preprocessing."

# 4) Train a strong baseline (XGBoost if available, else RandomForest)
if xgb is not None:
    clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        eval_metric='auc'
    )
else:
    clf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)

clf.fit(Xtr_enc, y_train)

# 5) Evaluate
proba_valid = clf.predict_proba(Xva_enc)[:, 1]
proba_test  = clf.predict_proba(Xte_enc)[:, 1]

print('\nVALID AUC:', roc_auc_score(y_valid, proba_valid))
print('VALID AUPRC:', average_precision_score(y_valid, proba_valid))
print('\nTEST AUC:', roc_auc_score(y_test, proba_test))
print('TEST AUPRC:', average_precision_score(y_test, proba_test))

# Thresholded metrics (0.5 as a placeholder)
th_valid = (proba_valid >= 0.5).astype(int)
print('\nVALID classification report (thr=0.5):')
print(classification_report(y_valid, th_valid, digits=3))

# 6) SHAP explainability (sample to keep plots readable)
if shap is not None:
    try:
        explainer = None
        if xgb is not None and isinstance(clf, xgb.XGBClassifier):
            # TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(clf, feature_names=feature_names)
        else:
            # Model-agnostic KernelExplainer (slow) — sample 200
            bg_idx = np.random.choice(Xtr_enc.shape[0], size=min(200, Xtr_enc.shape[0]), replace=False)
            explainer = shap.KernelExplainer(lambda d: clf.predict_proba(d)[:,1], Xtr_enc[bg_idx, :])

        samp = min(2000, Xva_enc.shape[0])
        idx  = np.random.choice(Xva_enc.shape[0], size=samp, replace=False)
        X_shap = Xva_enc[idx, :]

        shap_values = explainer(X_shap)

        # Summary plots
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features=X_shap, feature_names=feature_names, plot_type='dot', show=False, max_display=25)
        plt.title('SHAP Summary — Top Features (VALID sample)')
        plt.tight_layout(); plt.savefig(os.path.join(ART_DIR, 'shap_summary.png'), dpi=150); plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features=X_shap, feature_names=feature_names, plot_type='bar', show=False, max_display=25)
        plt.title('Mean |SHAP| by Feature (VALID sample)')
        plt.tight_layout(); plt.savefig(os.path.join(ART_DIR, 'shap_bar.png'), dpi=150); plt.close()

        print('Saved SHAP plots → artifacts/shap_summary.png, artifacts/shap_bar.png')
    except Exception as e:
        print('SHAP step skipped due to error:', e)
else:
    print('SHAP not installed — skipping explainability step.')

# 7) Persist processed splits for downstream use
# Encoded matrices are numpy; wrap to DataFrame with names for convenience
Xtr_df = pd.DataFrame(Xtr_enc, columns=feature_names, index=X_train.index)
Xva_df = pd.DataFrame(Xva_enc, columns=feature_names, index=X_valid.index)
Xte_df = pd.DataFrame(Xte_enc, columns=feature_names, index=X_test.index)

Xtr_df[TARGET] = y_train.values
Xva_df[TARGET] = y_valid.values
Xte_df[TARGET] = y_test.values

Xtr_df.to_csv(os.path.join(OUT_DIR, 'train_encoded.csv'), index=False)
Xva_df.to_csv(os.path.join(OUT_DIR, 'valid_encoded.csv'), index=False)
Xte_df.to_csv(os.path.join(OUT_DIR, 'test_encoded.csv'), index=False)
print('Saved encoded splits → train/valid/test CSVs in processed_data/.')

print('\n=== PART 2 COMPLETE. ===')
