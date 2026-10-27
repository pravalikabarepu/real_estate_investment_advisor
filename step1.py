"""STEP 1 - Clean the raw data, engineer features, create both target variables.
Run:  python -m src.preprocess
Output: data/cleaned_housing.csv  and  models/city_stats.json
"""
import json

import numpy as np
import pandas as pd

from src.common import (CLEAN_CSV, DATA_DIR, MODEL_DIR, RAW_CSV, RAW_NUM,
                        add_features, to_number)

BASE_GROWTH = 0.08          # fallback yearly growth rate (8%)
YEARS = 5


def cap_outliers(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Clip extreme values using the IQR rule (k=3 keeps only *extreme* outliers)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - k * iqr, q3 + k * iqr)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # ---- 1. load -------------------------------------------------------
    df = pd.read_csv(r"C:\Users\Pravalika.b\Downloads\india_housing_prices.csv")
    df.columns = df.columns.str.strip()
    print("Raw shape:", df.shape)

    # ---- 2. duplicates and useless columns -----------------------------
    df = df.drop_duplicates()
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    print("After removing duplicates:", df.shape)

    # ---- 3. data types + missing values --------------------------------
    for c in RAW_NUM:
        if c in df.columns:
            df[c] = to_number(df[c])
    if "Amenities" in df.columns:
        df["Amenities"] = df["Amenities"].fillna("None")
    print("Missing values before imputation:\n", df.isna().sum()[df.isna().sum() > 0])
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())      # numeric -> median
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])  # categorical -> mode

    # ---- 4. remove impossible rows, cap extreme outliers ---------------
    df = df[(df["Size_in_SqFt"] > 0) & (df["Price_in_Lakhs"] > 0)]
    for c in ["Size_in_SqFt", "Price_in_Lakhs"]:
        df[c] = cap_outliers(df[c])

    # ---- 5. feature engineering ----------------------------------------
    df = add_features(df)
    df["Price_per_SqFt"] = cap_outliers(df["Price_per_SqFt"])

    # ---- 6. classification label: Good_Investment (multi-factor score) -
    df["City_Median_PPSF"] = df.groupby("City")["Price_per_SqFt"].transform("median")
    infra_median = float(df["Infrastructure_Score"].median())
    df["Investment_Score"] = (
        (df["Price_per_SqFt"] <= df["City_Median_PPSF"]).astype(int)   # cheaper than city median
        + (df["BHK"] >= 3).astype(int)                                  # family-size home
        + df["Is_Ready"]                                                # ready to move in
        + (df["Infrastructure_Score"] >= infra_median).astype(int)      # good surroundings
    )
    df["Good_Investment"] = (df["Investment_Score"] >= 3).astype(int)
    print("\nGood_Investment balance:\n", df["Good_Investment"].value_counts(normalize=True).round(3))

    # ---- 7. regression target: price after 5 years ---------------------
    # City growth rate between 6% and 10%, based on how expensive the city is
    city_med = df.groupby("City")["Price_per_SqFt"].median()
    city_rate = 0.06 + 0.04 * city_med.rank(pct=True)
    df["Growth_Rate"] = df["City"].map(city_rate).fillna(BASE_GROWTH) + 0.01 * df["Good_Investment"]
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** YEARS

    # ---- 8. save --------------------------------------------------------
    df.to_csv(CLEAN_CSV, index=False)
    with open(MODEL_DIR / "city_stats.json", "w") as f:
        json.dump({"city_median_ppsf": city_med.to_dict(),
                   "city_growth_rate": city_rate.to_dict(),
                   "infra_median": infra_median}, f, indent=2)
    print("\nSaved", CLEAN_CSV, "with shape", df.shape)


if __name__ == "__main__":
    main()
