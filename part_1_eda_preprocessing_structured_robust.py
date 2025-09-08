# PART 1 — EDA & PREPROCESSING (ROBUST & REPRODUCIBLE)
# ---------------------------------------------------
# This file standardizes the order of operations you outlined and
# fixes common edge cases (sentinels, stringy NaNs, rare bins, etc.).
# It STOPS before splitting/training/SHAP (that will be Part 2).

import os
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# 0) Load & basic hygiene
# -------------------------
RAW_CSV = 'Case_Data.csv'
OUT_DIR = 'processed_data'
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RAW_CSV)
print('Raw shape:', df.shape)

# Deduplicate (exact)
working_dataframe = df.drop_duplicates().reset_index(drop=True).copy()
print('Rows after de-dupe:', working_dataframe.shape[0])

# Standardize obvious stringy-missings EARLY (affects num/cat)
MISSING_STRINGS = ["NA", "N/A", "na", "n/a", "NaN", "nan", "", " "]
working_dataframe.replace(MISSING_STRINGS, np.nan, inplace=True)

# ---------------------------------------
# 1) Temporal cleanup (DAYS_* → years)
#    + sentinel handling & indicators
# ---------------------------------------
DAYS_COLS = [c for c in working_dataframe.columns if c.startswith('DAYS_')]
for c in DAYS_COLS:
    working_dataframe[c] = pd.to_numeric(working_dataframe[c], errors='coerce')

# Home Credit sentinel for employment
if 'DAYS_EMPLOYED' in working_dataframe.columns:
    EMP_SENTINEL = 365243
    emp_mask = working_dataframe['DAYS_EMPLOYED'].abs() == EMP_SENTINEL
    working_dataframe['EMPLOYMENT_SENTINEL_FLAG'] = emp_mask.astype(int)
    if emp_mask.any():
        print(f"Converting DAYS_EMPLOYED sentinel {EMP_SENTINEL} to NaN on {int(emp_mask.sum())} rows")
        working_dataframe.loc[emp_mask, 'DAYS_EMPLOYED'] = np.nan

# Convert to positive years (use 365.25 for leap year averaging)
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

# Optional missingness indicators (helpful for linear models)
for col in ['AGE_YEARS','EMPLOYMENT_YEARS','ID_YEARS','PHONE_CHANGE_YEARS','REGISTRATION_YEARS']:
    if col in working_dataframe.columns:
        working_dataframe[f'{col}_MISSING'] = working_dataframe[col].isna().astype(int)

# Quick visual check for temporals if present
cols_to_show = [c for c in [
    'DAYS_BIRTH','AGE_YEARS',
    'DAYS_EMPLOYED','EMPLOYMENT_YEARS','EMPLOYMENT_SENTINEL_FLAG',
    'DAYS_ID_PUBLISH','ID_YEARS',
    'DAYS_LAST_PHONE_CHANGE','PHONE_CHANGE_YEARS',
    'DAYS_REGISTRATION','REGISTRATION_YEARS'
] if c in working_dataframe.columns]
if cols_to_show:
    display(working_dataframe[cols_to_show].head())

# Drop original DAYS_* to avoid leakage/confusion downstream
working_dataframe.drop(columns=[c for c in DAYS_COLS if c in working_dataframe.columns], errors='ignore', inplace=True)

# -------------------------------------------------------
# 2) Type detection (exclude TARGET) + numeric sanitizing
# -------------------------------------------------------
TARGET = 'TARGET'

# Replace infs from any ratios that might exist
working_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

categorical_features = working_dataframe.select_dtypes(include=['object','category']).columns.tolist()
numerical_features   = working_dataframe.select_dtypes(include=['number']).columns.tolist()
if TARGET in categorical_features: categorical_features.remove(TARGET)
if TARGET in numerical_features:   numerical_features.remove(TARGET)
print(f"Categorical: {len(categorical_features)} | Numeric: {len(numerical_features)}")

# -------------------------------------------------
# 3) EDA helpers — categorical freq & cardinality
# -------------------------------------------------

def categorical_summary(dframe: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    categorical_columns = dframe.select_dtypes(include=['object','category'])
    rows = []
    for column in categorical_columns.columns:
        vc = categorical_columns[column].value_counts(dropna=False).head(top_n)
        for k, v in vc.items():
            rows.append({'column': column, 'value': k, 'count': int(v)})
    return pd.DataFrame(rows)

cat_summary = categorical_summary(working_dataframe, top_n=8)
# display(cat_summary.head(40))  # uncomment to preview

# Cardinality table
cat_unique_counts_df = (
    pd.DataFrame({
        'column': categorical_features,
        'unique_count': [working_dataframe[c].nunique(dropna=False) for c in categorical_features]
    })
    .sort_values('unique_count', ascending=False)
    .reset_index(drop=True)
)
print('\nTop categorical by cardinality:')
print(cat_unique_counts_df.head(10))

# Rare-binning candidates (only high-cardinality cats)
RARE_THRESHOLD = 0.02
HIGH_CARDINALITY = 8
high_card_cols = cat_unique_counts_df.query('unique_count > @HIGH_CARDINALITY')['column'].tolist()

# Fit/apply rare maps (NOTE: for true ML hygiene, this should be fit on TRAIN ONLY in Part 2)

def fit_rare_maps(dframe: pd.DataFrame, cols: list, thresh: float = RARE_THRESHOLD):
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

if high_card_cols:
    rare_maps = fit_rare_maps(working_dataframe, high_card_cols, RARE_THRESHOLD)
    working_dataframe = apply_rare_maps(working_dataframe, rare_maps)
    print(f"Applied rare binning (threshold={RARE_THRESHOLD}) to: {high_card_cols}")

# Recompute types (in case rare binning changed dtypes)
categorical_features = working_dataframe.select_dtypes(include=['object','category']).columns.tolist()
numerical_features   = working_dataframe.select_dtypes(include=['number']).columns.tolist()
if TARGET in categorical_features: categorical_features.remove(TARGET)
if TARGET in numerical_features:   numerical_features.remove(TARGET)

print(f"\nNulls BEFORE imputation: {int(working_dataframe.isna().sum().sum())}")

# ------------------------------------------------------
# 4) Preprocessor (impute+encode/scale) with hard checks
# ------------------------------------------------------
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

# Separate X/y
X = working_dataframe.drop(columns=[c for c in [TARGET] if c in working_dataframe.columns])
y = working_dataframe[TARGET] if TARGET in working_dataframe.columns else None

# Fit-transform
X_enc = preprocessor.fit_transform(X)

# Build output feature names
try:
    num_names_out = preprocessor.named_transformers_["num"].named_steps["imputer"].get_feature_names_out(numerical_features).tolist()
except Exception:
    num_names_out = numerical_features
cat_names_out = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features).tolist()
feature_names = num_names_out + cat_names_out

# Wrap for inspection and HARD sanity checks
X_enc_df = pd.DataFrame(X_enc, columns=feature_names, index=X.index)
print(f"AFTER transform → Nulls: {int(np.isnan(X_enc_df.values).sum())} | Infs: {int(np.isinf(X_enc_df.values).sum())}")
assert not np.isnan(X_enc_df.values).any(), "Preprocessor produced NaNs!"
assert not np.isinf(X_enc_df.values).any(), "Preprocessor produced Infs!"

# Save encoded matrix for Part 2 (optional but handy)
X_save = X_enc_df.copy()
if y is not None:
    X_save[TARGET] = y.values
encoded_path = os.path.join(OUT_DIR, 'encoded_for_model.csv')
X_save.to_csv(encoded_path, index=False)
print('Saved:', encoded_path, '| Shape:', X_save.shape)

# ---------------------
# 5) EDA snapshots out
# ---------------------
# Top missing columns BEFORE imputation (from original working_dataframe)
null_counts = df.replace(MISSING_STRINGS, np.nan).isna().sum().sort_values(ascending=False)
null_report = null_counts[null_counts > 0].head(20)
print('\nTop raw missing columns:')
print(null_report)

# Cardinality snapshot
print('\nCategorical cardinality (top 15):')
print(cat_unique_counts_df.head(15))

print('\nPART 1 complete — data is cleaned, transformed, encoded, and saved.\n(Proceed to PART 2 for split/train/eval/SHAP.)')
