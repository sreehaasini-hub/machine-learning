"""Dataset loading, validation and all dataset-derived statistics."""
import os
import numpy as np
import pandas as pd

REQUIRED = ["Crop", "Crop_Year", "Season", "State", "Area", "Production",
            "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
CAT = ["Crop", "Season", "State"]
NUM = ["Crop_Year", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
CORR_COLS = ["Annual_Rainfall", "Area", "Production", "Fertilizer", "Pesticide", "Yield"]


class DataError(Exception):
    """Raised when the dataset is missing or malformed."""


def load_data(path):
    if not os.path.exists(path):
        raise DataError(f"Dataset not found at: {path}. Place crop_yield.csv inside the data/ folder.")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"Required column(s) missing from crop_yield.csv: {', '.join(missing)}")
    for c in CAT:  # raw file has trailing spaces e.g. 'Kharif     ', 'Coconut '
        df[c] = df[c].astype(str).str.strip()
    return df


def overview_stats(df):
    return {
        "records": len(df), "states": df.State.nunique(), "crops": df.Crop.nunique(),
        "seasons": df.Season.nunique(), "year_min": int(df.Crop_Year.min()),
        "year_max": int(df.Crop_Year.max()), "avg_rain": df.Annual_Rainfall.mean(),
        "avg_yield": df.Yield.mean(), "median_yield": df.Yield.median(),
    }


def quality_report(df, raw_dtypes=None):
    miss = df.isna().sum()
    rows = [{"column": c, "dtype": str(df[c].dtype), "missing": int(miss[c]),
             "unique": int(df[c].nunique()),
             "kind": "Categorical" if c in CAT else "Numerical"} for c in df.columns]
    return {
        "shape": df.shape, "rows": rows, "missing_total": int(miss.sum()),
        "duplicates": int(df.duplicated().sum()),
        "complete": int((~df.isna().any(axis=1)).sum()),
        "n_num": sum(r["kind"] == "Numerical" for r in rows),
        "n_cat": sum(r["kind"] == "Categorical" for r in rows),
        "zero_yield": int((df.Yield == 0).sum()), "zero_prod": int((df.Production == 0).sum()),
        "seasons": sorted(df.Season.unique()),
    }


def correlations(df):
    return (df[CORR_COLS].corr(method="pearson"), df[CORR_COLS].corr(method="spearman"))


def filter_df(df, state="", crop="", season=""):
    if state: df = df[df.State == state]
    if crop: df = df[df.Crop == crop]
    if season: df = df[df.Season == season]
    return df


def yearly(df):
    g = df.groupby("Crop_Year")
    return pd.DataFrame({"rain": g.Annual_Rainfall.mean(), "yield": g.Yield.mean(),
                         "prod": g.Production.sum(), "crops": g.Crop.nunique(),
                         "records": g.size()}).reset_index()


def trend_slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0


def key_findings(df):
    rain = df.groupby("State").Annual_Rainfall.mean()
    yc = df.groupby("Crop").Yield.mean()
    pc = df.groupby("Crop").Production.sum()
    ys = df.groupby("Crop_Year").Yield.mean()
    rs = df.groupby("Crop_Year").Annual_Rainfall.mean()
    p, _ = correlations(df)
    yr = p["Yield"].drop("Yield")
    return {
        "rain_hi": (rain.idxmax(), rain.max()), "rain_lo": (rain.idxmin(), rain.min()),
        "yield_crop": (yc.idxmax(), yc.max()), "prod_crop": (pc.idxmax(), pc.max()),
        "corr": (yr.abs().idxmax(), yr[yr.abs().idxmax()]),
        "year_yield": (int(ys.idxmax()), ys.max()), "year_rain": (int(rs.idxmax()), rs.max()),
        "rain_corr": yr["Annual_Rainfall"],
    }
