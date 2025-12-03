#!/usr/bin/env python3
"""
GeorgiaRealEstateModel.py

Train a tuned HistGradientBoostingRegressor to predict log(sale_price)
from Georgia property data (processed CSVs).

Key points in this version:
- Uses BOTH numeric and categorical features.
- Trims extreme sale_price outliers (top OUTLIER_QUANTILE).
- Runs a light RandomizedSearchCV to tune HistGradientBoostingRegressor without
  blowing up Windows resources:
    * Subsamples training data for tuning.
    * Uses n_jobs=1 (sequential).
    * Then refits best model on FULL training data.
- Categorical columns are cast to string in the DataFrame before encoding
  (so OneHotEncoder never sees mixed float/str columns).
- Saves the entire pipeline (preprocessing + model) to:
      georgia_gbdt_model.joblib
- Shows:
  - Histogram of sale_price (after trimming)
  - Actual vs Predicted price (test set)
  - Residuals vs Predicted price
  - Permutation-based feature importances
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from joblib import dump

import matplotlib.pyplot as plt

# ===================== CONFIG =====================

DO_TUNING = True              # set to False if you want faster runs
OUTLIER_QUANTILE = 0.995      # drop top 0.5% of sale_price
N_RANDOM_SEARCH_ITER = 8      # number of random hyperparam combos

# ==================================================


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_processed_directory() -> str:
    """
    Returns the path to the "Georgia Real Estate Data Processed" folder,
    assuming this script is inside "Georgia Real Estate ML Model"
    in the same parent directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(parent_dir, "Georgia Real Estate Data Processed")
    return processed_dir


def list_processed_csvs(processed_dir: str) -> List[str]:
    """
    Return a sorted list of processed CSV filenames ending in '_processed.csv'
    in the given directory.
    """
    if not os.path.isdir(processed_dir):
        print(f"[ERROR] Processed folder not found: {processed_dir}")
        return []

    files = [
        name
        for name in os.listdir(processed_dir)
        if name.lower().endswith("_processed.csv")
    ]
    files.sort()
    return files


def choose_files(csv_files: List[str], processed_dir: str) -> List[str]:
    """
    Allow the user to choose which processed CSV files to use.

    Options:
    - Command-line arguments:
        python GeorgiaRealEstateModel.py all
        python GeorgiaRealEstateModel.py 1 3
        python GeorgiaRealEstateModel.py Athens_PropertyData_processed.csv
    - Interactive prompt:
        Type 'all' or a comma-separated list of indices (e.g., 1,3).
    """
    if not csv_files:
        print("[ERROR] No *_processed.csv files found in processed folder.")
        sys.exit(1)

    print("\nAvailable processed CSV files in:")
    print(f"  {processed_dir}\n")
    for idx, name in enumerate(csv_files, start=1):
        print(f"  {idx}. {name}")

    # Command-line override
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        print(f"\n[INFO] Using command-line arguments: {args}")

        # Single argument: "all" or "*"
        if len(args) == 1 and args[0].lower() in ("all", "*"):
            return csv_files

        chosen: List[str] = []
        for arg in args:
            if arg.isdigit():
                idx_int = int(arg)
                if 1 <= idx_int <= len(csv_files):
                    chosen.append(csv_files[idx_int - 1])
                else:
                    print(f"[ERROR] Index out of range: {arg}")
                    sys.exit(1)
            else:
                candidate = arg
                if not candidate.lower().endswith(".csv"):
                    candidate = candidate + ".csv"
                if candidate in csv_files:
                    chosen.append(candidate)
                else:
                    print(f"[ERROR] File not found in list: {candidate}")
                    sys.exit(1)

        if not chosen:
            print("[ERROR] No valid files from command line arguments.")
            sys.exit(1)

        return chosen

    # Interactive input
    user_input = input(
        "\nType 'all' to use all files, or a comma-separated list of indices "
        "(e.g., 1,3): "
    ).strip()

    if user_input.lower() in ("all", "*", ""):
        return csv_files

    indices: List[int] = []
    for part in user_input.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            print(f"[ERROR] Invalid index '{part}'")
            sys.exit(1)
        idx_int = int(part)
        if 1 <= idx_int <= len(csv_files):
            indices.append(idx_int)
        else:
            print(f"[ERROR] Index out of range: {idx_int}")
            sys.exit(1)

    chosen_files = [csv_files[i - 1] for i in indices]
    return chosen_files


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_combine(processed_dir: str, chosen_files: List[str]) -> pd.DataFrame:
    """
    Load each chosen CSV from the processed directory and concatenate them
    into a single DataFrame.
    """
    dfs = []
    for name in chosen_files:
        path = os.path.join(processed_dir, name)
        print(f"[INFO] Loading {path}")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[INFO] Combined rows from all chosen files: {len(combined)}")
    return combined


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sale_date to datetime and engineer simple time-based features.

    - We do NOT drop rows with invalid sale_date; instead we keep them as NaT
      and the derived time features will be NaN. HistGradientBoostingRegressor
      can handle NaNs in numeric features.
    """
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df.get("sale_date"), errors="coerce")
    missing_dates = df["sale_date"].isna().sum()

    if missing_dates > 0:
        print(
            f"[WARN] {missing_dates} rows have invalid sale_date after parsing; "
            f"keeping them with NaN time features."
        )

    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_quarter"] = df["sale_date"].dt.quarter
    df["sale_year_month"] = df["sale_year"] * 12 + df["sale_month"]

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "sale_price",
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], pd.DataFrame]:
    """
    Prepare X (features) and y (log target) for modeling.

    Steps:
    - Ensure target is numeric and > 0.
    - Engineer time-based features.
    - Trim extreme outliers (top OUTLIER_QUANTILE by sale_price).
    - Log-transform sale_price into log_sale_price.
    - Split columns into:
        - numeric_cols (excluding ID/loan fields and the target/log)
        - categorical_cols (low-cardinality object columns like city,
          county, property_type, etc.; addresses/names are excluded).
    - Cast all categorical columns to string so OneHotEncoder never
      sees mixed types.
    """

    df = df.copy()

    # Clean target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    before_rows = len(df)
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0]
    print(f"[INFO] Dropped {before_rows - len(df)} rows with missing/invalid {target_col}")

    # Time features
    df = engineer_time_features(df)

    # Trim extreme outliers in price
    upper = df[target_col].quantile(OUTLIER_QUANTILE)
    before_trim = len(df)
    df = df[df[target_col] <= upper]
    print(
        f"[INFO] Trimmed outliers above {OUTLIER_QUANTILE*100:.1f}th percentile "
        f"({upper:,.0f}). Rows removed: {before_trim - len(df)}"
    )

    # Plot histogram of sale_price AFTER trimming
    plt.figure(figsize=(8, 4))
    df[target_col].hist(bins=50)
    plt.xlabel("Sale price ($)")
    plt.ylabel("Count")
    plt.title("Distribution of sale_price (after outlier trimming)")
    plt.tight_layout()

    # Log-transform target
    df["log_sale_price"] = np.log(df[target_col])

    # Numeric features
    numeric_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = set([target_col, "log_sale_price"])
    numeric_cols: List[str] = [
        col
        for col in numeric_cols_all
        if col not in drop_cols
        and "id" not in col.lower()
        and "loan" not in col.lower()
    ]

    # Categorical features (object dtype)
    cat_candidates = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols: List[str] = []

    for col in cat_candidates:
        col_lower = col.lower()

        # Skip addresses and names (too high-cardinality / not generalizable)
        if any(
            kw in col_lower
            for kw in ["address", "name", "buyer", "seller", "owner"]
        ):
            continue

        uniq = df[col].nunique(dropna=True)
        if uniq == 0:
            continue

        # Keep if relatively low-cardinality or obviously location/type-related
        if (
            uniq <= 50
            or any(
                kw in col_lower
                for kw in [
                    "city",
                    "county",
                    "state",
                    "zip",
                    "geography",
                    "neighborhood",
                    "subdivision",
                    "property_type",
                    "home_type",
                    "land_use",
                ]
            )
        ):
            categorical_cols.append(col)

    print(
        f"[INFO] Using {len(numeric_cols)} numeric features and "
        f"{len(categorical_cols)} categorical features."
    )

    # Cast categorical columns to string to avoid mixed-type errors in OneHotEncoder
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].astype(str)

    X = df[numeric_cols + categorical_cols]
    y = df["log_sale_price"].values

    return X, y, numeric_cols, categorical_cols, df


# ---------------------------------------------------------------------------
# Modeling & visualization
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
):
    """
    Train (and optionally tune) a HistGradientBoostingRegressor pipeline,
    then evaluate it and show plots.

    To avoid Windows resource issues:
    - We subsample the training set to at most 20k rows for tuning.
    - We run RandomizedSearchCV with n_jobs=1 (sequential).
    - After tuning, we refit the best pipeline on the FULL training set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"[INFO] Train size: {X_train.shape[0]} rows")
    print(f"[INFO] Test  size: {X_test.shape[0]} rows")

    # ----- Preprocessor -----
    # Numeric passthrough, categorical one-hot
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    base_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    pipeline = base_pipeline

    # ----- Hyperparameter tuning (lightweight) -----
    if DO_TUNING:
        print("\n[INFO] Running RandomizedSearchCV for HistGradientBoostingRegressor...")

        # Subsample training data for tuning to reduce memory / time
        X_tune, y_tune = X_train, y_train
        max_tune_rows = 20000
        if X_train.shape[0] > max_tune_rows:
            X_tune, _, y_tune, _ = train_test_split(
                X_train,
                y_train,
                train_size=max_tune_rows,
                random_state=42,
            )
            print(f"[INFO] Using {X_tune.shape[0]} rows for hyperparameter tuning.")

        param_distributions = {
            "model__max_iter": [200, 300, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [None, 3, 5, 8],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }

        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=param_distributions,
            n_iter=N_RANDOM_SEARCH_ITER,
            cv=3,
            n_jobs=1,  # IMPORTANT: sequential to avoid WinError 1450
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=2,
        )

        search.fit(X_tune, y_tune)

        print("\n[INFO] Best hyperparameters found:")
        print(search.best_params_)

        # Take best pipeline and refit on the FULL training set
        pipeline = search.best_estimator_
        print("\n[INFO] Re-fitting best pipeline on full training data...")
        pipeline.fit(X_train, y_train)

    else:
        print("\n[INFO] Training base pipeline without tuning...")
        pipeline.fit(X_train, y_train)

    # ----- Evaluation on test set -----
    y_test_pred = pipeline.predict(X_test)

    # Metrics in log space
    rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_log = mean_absolute_error(y_test, y_test_pred)
    r2_log = r2_score(y_test, y_test_pred)

    print("\n[METRICS - log(price)]")
    print(f"  RMSE (log) : {rmse_log:.4f}")
    print(f"  MAE  (log) : {mae_log:.4f}")
    print(f"  R^2  (log) : {r2_log:.4f}")

    # Metrics in price space
    y_test_price = np.exp(y_test)
    y_test_pred_price = np.exp(y_test_pred)

    rmse_price = np.sqrt(mean_squared_error(y_test_price, y_test_pred_price))
    mae_price = mean_absolute_error(y_test_price, y_test_pred_price)
    r2_price = r2_score(y_test_price, y_test_pred_price)

    print("\n[METRICS - price]")
    print(f"  RMSE ($)   : {rmse_price:,.2f}")
    print(f"  MAE  ($)   : {mae_price:,.2f}")
    print(f"  R^2        : {r2_price:.4f}")

    # ----- Plots -----

    # Actual vs Predicted price
    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_test_price,
        y_test_pred_price,
        alpha=0.3,
        edgecolor="none",
    )
    min_val = min(y_test_price.min(), y_test_pred_price.min())
    max_val = max(y_test_price.max(), y_test_pred_price.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)
    plt.xlabel("Actual price ($)")
    plt.ylabel("Predicted price ($)")
    plt.title("Test set: Actual vs Predicted sale_price")
    plt.tight_layout()

    # Residuals vs Predicted price
    residuals = y_test_price - y_test_pred_price
    plt.figure(figsize=(8, 4))
    plt.scatter(
        y_test_pred_price,
        residuals,
        alpha=0.3,
        edgecolor="none",
    )
    plt.axhline(0, color="k", linestyle="--", linewidth=2)
    plt.xlabel("Predicted price ($)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs Predicted price (test set)")
    plt.tight_layout()

    # Feature importances via permutation importance
    print("\n[INFO] Computing permutation importances (this may take a bit)...")
    max_samples = 5000
    if X_test.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_test.shape[0], size=max_samples, replace=False)
        X_perm = X_test.iloc[idx]
        y_perm = y_test[idx]
    else:
        X_perm = X_test
        y_perm = y_test

    perm_result = permutation_importance(
        pipeline,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importances = perm_result.importances_mean

    # This now works because our ColumnTransformer has plain OneHotEncoder
    feature_names_out = pipeline.named_steps["preprocess"].get_feature_names_out()
    idx_sorted = np.argsort(importances)[::-1]

    top_k = min(20, len(idx_sorted))
    top_idx = idx_sorted[:top_k]

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_k), importances[top_idx])
    plt.xticks(
        range(top_k),
        [feature_names_out[i] for i in top_idx],
        rotation=60,
        ha="right",
    )
    plt.ylabel("Permutation importance (mean Δscore)")
    plt.title("Top feature importances (HistGradientBoostingRegressor)")
    plt.tight_layout()

    plt.show()

    return pipeline, y_test, y_test_pred


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    processed_dir = get_processed_directory()
    print(f"[INFO] Processed data folder: {processed_dir}")

    csv_files = list_processed_csvs(processed_dir)
    chosen_files = choose_files(csv_files, processed_dir)

    print("\n[INFO] Using these processed files:")
    for name in chosen_files:
        print(f"  - {name}")

    df = load_and_combine(processed_dir, chosen_files)
    X, y, numeric_cols, categorical_cols, df = prepare_features_and_target(
        df,
        target_col="sale_price",
    )

    pipeline, y_test, y_test_pred = train_and_evaluate(
        X,
        y,
        numeric_cols,
        categorical_cols,
    )

    # Save trained pipeline + feature lists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "georgia_gbdt_model.joblib")

    dump(
        {
            "pipeline": pipeline,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        },
        model_path,
    )

    print(f"\n[INFO] Saved trained model bundle to:\n  {model_path}")


if __name__ == "__main__":
    main()
