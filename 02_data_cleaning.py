"""
02_data_cleaning.py
--------------------
Cleans raw data, handles missing values, engineers features like:
- 7-day rolling averages
- Case Fatality Rate (CFR)
- Wave classification (Wave 1 / 2 / 3)
- Vaccination coverage %
Saves cleaned data to /data/cleaned_statewise.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    statewise = pd.read_csv(
        os.path.join(DATA_DIR, "statewise_covid.csv"), parse_dates=["date"]
    )
    demographics = pd.read_csv(os.path.join(DATA_DIR, "demographics.csv"))
    return statewise, demographics


def clean_statewise(df):
    print("Cleaning state-wise COVID data...")

    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # Fill negatives (data corrections in raw source) with 0
    for col in ["new_cases", "new_deaths", "new_vaccinations"]:
        df[col] = df[col].clip(lower=0)

    # 7-day rolling averages per state
    df["cases_7day_avg"] = (
        df.groupby("state")["new_cases"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .round(1)
    )
    df["deaths_7day_avg"] = (
        df.groupby("state")["new_deaths"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .round(2)
    )

    # Case Fatality Rate (rolling 14-day)
    df["cfr_14day"] = (
        df.groupby("state")["new_deaths"]
        .transform(lambda x: x.rolling(14, min_periods=1).sum())
        /
        df.groupby("state")["new_cases"]
        .transform(lambda x: x.rolling(14, min_periods=1).sum())
        .replace(0, np.nan)
        * 100
    ).round(3)

    # Wave classification based on date ranges (India-specific)
    def classify_wave(date):
        if pd.Timestamp("2020-03-01") <= date <= pd.Timestamp("2021-02-28"):
            return "Wave 1"
        elif pd.Timestamp("2021-03-01") <= date <= pd.Timestamp("2021-07-31"):
            return "Wave 2 (Delta)"
        elif pd.Timestamp("2021-12-01") <= date <= pd.Timestamp("2022-03-31"):
            return "Wave 3 (Omicron)"
        else:
            return "Inter-wave"

    df["wave"] = df["date"].apply(classify_wave)

    print(f"  Rows after cleaning: {len(df)}")
    return df


def merge_demographics(covid_df, demo_df):
    print("Merging demographic features...")

    # Aggregate COVID data to state-level totals
    state_totals = covid_df.groupby("state").agg(
        total_cases=("new_cases", "sum"),
        total_deaths=("new_deaths", "sum"),
        total_vaccinations=("new_vaccinations", "sum"),
        peak_daily_cases=("new_cases", "max"),
        peak_daily_deaths=("new_deaths", "max"),
    ).reset_index()

    # Wave-wise breakdown
    wave_totals = covid_df.groupby(["state", "wave"]).agg(
        wave_cases=("new_cases", "sum"),
        wave_deaths=("new_deaths", "sum"),
    ).reset_index()

    # Merge with demographics
    merged = state_totals.merge(demo_df, on="state", how="inner")

    # Derived features
    merged["cases_per_million"] = (
        merged["total_cases"] / merged["population_millions"]
    ).round(1)
    merged["deaths_per_million"] = (
        merged["total_deaths"] / merged["population_millions"]
    ).round(1)
    merged["overall_cfr"] = (
        merged["total_deaths"] / merged["total_cases"].replace(0, np.nan) * 100
    ).round(3)
    merged["vaccination_coverage_pct"] = (
        merged["total_vaccinations"] / (merged["population_millions"] * 1e6) * 100
    ).round(2)

    return merged, wave_totals


def save_cleaned(covid_df, merged_df, wave_df):
    covid_df.to_csv(os.path.join(DATA_DIR, "cleaned_statewise.csv"), index=False)
    merged_df.to_csv(os.path.join(DATA_DIR, "state_summary.csv"), index=False)
    wave_df.to_csv(os.path.join(DATA_DIR, "wave_breakdown.csv"), index=False)
    print("  Saved: cleaned_statewise.csv, state_summary.csv, wave_breakdown.csv")


if __name__ == "__main__":
    statewise, demographics = load_data()
    cleaned = clean_statewise(statewise)
    summary, wave_df = merge_demographics(cleaned, demographics)
    save_cleaned(cleaned, summary, wave_df)
    print("\n✅ Cleaning complete.")
    print(summary[["state", "total_cases", "overall_cfr", "vaccination_coverage_pct"]].to_string(index=False))
