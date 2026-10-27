"""Shared settings and helper functions used by every script and by the Streamlit app."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EDA_DIR = BASE_DIR / "eda_outputs"
RAW_CSV = Path(r"C:\Users\Pravalika.b\Downloads\india_housing_prices.csv")
CLEAN_CSV = DATA_DIR / "cleaned_housing_sample.csv"

CURRENT_YEAR = 2026
RANDOM_STATE = 42

# ---------------------------------------------------------------- columns
# Raw numeric columns of the dataset (used for type conversion)
RAW_NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
           "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
           "Nearby_Hospitals", "Parking_Space"]

# Columns the models learn from (Locality is dropped: too many unique values)
NUM_COLS = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Floor_No", "Total_Floors",
            "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space",
            "Floor_Ratio", "Infrastructure_Score", "Amenity_Count"]
CAT_COLS = ["State", "City", "Property_Type", "Furnished_Status",
            "Public_Transport_Accessibility", "Security", "Facing",
            "Owner_Type", "Availability_Status"]

TRANSPORT_MAP = {"low": 1, "poor": 1, "medium": 2, "moderate": 2, "average": 2,
                 "high": 3, "good": 3, "very high": 4, "excellent": 4}


# ---------------------------------------------------------------- helpers
def to_number(s: pd.Series) -> pd.Series:
    """Convert a column to numbers; also understands Yes/No style values."""
    num = pd.to_numeric(s, errors="coerce")
    if s.dtype == object:
        yn = s.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0})
        num = num.fillna(yn)
    return num


def transport_score(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    mapped = s.astype(str).str.strip().str.lower().map(TRANSPORT_MAP)
    return num.fillna(mapped).fillna(2)


def amenity_count(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).apply(
        lambda x: len([a for a in re.split(r"[,;|]", x) if a.strip()]))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features. Works on the full dataset AND on a single
    row coming from the Streamlit form, so training and prediction match."""
    df = df.copy()
    if "Year_Built" in df.columns:
        df["Age_of_Property"] = CURRENT_YEAR - pd.to_numeric(df["Year_Built"], errors="coerce")
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"].replace(0, np.nan)
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].replace(0, np.nan)).fillna(0)
    df["Transport_Score"] = transport_score(df["Public_Transport_Accessibility"])
    df["Infrastructure_Score"] = df["Nearby_Schools"] + df["Nearby_Hospitals"] + df["Transport_Score"]
    df["Amenity_Count"] = amenity_count(df["Amenities"])
    df["Is_Ready"] = df["Availability_Status"].astype(str).str.lower().str.contains("ready|available").astype(int)
    return df


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)
