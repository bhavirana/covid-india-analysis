"""
03_eda_visualizations.py
-------------------------
Generates all exploratory visualizations:
  1. National daily cases + wave bands (time series)
  2. Top 10 states by total cases (bar chart)
  3. Wave-wise case distribution by state (grouped bar)
  4. Case Fatality Rate comparison across states
  5. Correlation heatmap: demographics vs COVID outcomes
  6. Vaccination coverage vs death rate scatter
  7. Urban population % vs cases per million
  8. Hospital beds vs CFR scatter

All plots saved to /outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
WAVE_COLORS = {
    "Wave 1":          "#4C72B0",
    "Wave 2 (Delta)":  "#DD8452",
    "Wave 3 (Omicron)":"#55A868",
    "Inter-wave":      "#C0C0C0",
}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def load():
    cleaned   = pd.read_csv(os.path.join(DATA_DIR, "cleaned_statewise.csv"),  parse_dates=["date"])
    summary   = pd.read_csv(os.path.join(DATA_DIR, "state_summary.csv"))
    wave_df   = pd.read_csv(os.path.join(DATA_DIR, "wave_breakdown.csv"))
    return cleaned, summary, wave_df


# ── Plot 1: National Daily Cases with Wave Bands ─────────────────────────────
def plot_national_timeseries(cleaned):
    print("  Plot 1: National daily cases time series...")
    national = (
        cleaned.groupby("date")[["new_cases", "cases_7day_avg"]]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    wave_bands = [
        ("2020-03-01", "2021-02-28", "Wave 1",          "#4C72B0"),
        ("2021-03-01", "2021-07-31", "Wave 2 (Delta)",  "#DD8452"),
        ("2021-12-01", "2022-03-31", "Wave 3 (Omicron)","#55A868"),
    ]
    for start, end, label, color in wave_bands:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.12, color=color, label=label)

    ax.bar(national["date"], national["new_cases"],
           color="#A8C5E8", alpha=0.5, width=1, label="Daily cases")
    ax.plot(national["date"], national["cases_7day_avg"],
            color="#1A3A6B", linewidth=2, label="7-day avg")

    ax.set_title("India COVID-19 Daily Cases — Three Waves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("New Cases")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e5:.0f}L"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_national_timeseries.png"))
    plt.close()


# ── Plot 2: Top 10 States by Total Cases ─────────────────────────────────────
def plot_top_states(summary):
    print("  Plot 2: Top 10 states by total cases...")
    top10 = summary.nlargest(10, "total_cases")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top10["state"], top10["total_cases"] / 1e6,
                   color=sns.color_palette("Blues_d", 10))
    ax.set_xlabel("Total Cases (Millions)")
    ax.set_title("Top 10 States — Total COVID-19 Cases", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, top10["total_cases"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val/1e6:.2f}M", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_top10_states_cases.png"))
    plt.close()


# ── Plot 3: Wave-wise Case Distribution ──────────────────────────────────────
def plot_wave_breakdown(wave_df):
    print("  Plot 3: Wave-wise case distribution...")
    top_states = ["Maharashtra", "Kerala", "Karnataka", "Tamil Nadu",
                  "Delhi", "Uttar Pradesh", "West Bengal", "Gujarat",
                  "Andhra Pradesh", "Rajasthan"]
    df = wave_df[
        wave_df["state"].isin(top_states) &
        wave_df["wave"].isin(["Wave 1", "Wave 2 (Delta)", "Wave 3 (Omicron)"])
    ].copy()
    df["wave_cases_L"] = df["wave_cases"] / 1e5

    pivot = df.pivot(index="state", columns="wave", values="wave_cases_L").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax,
               color=["#4C72B0", "#DD8452", "#55A868"],
               edgecolor="white", width=0.7)

    ax.set_title("Wave-wise COVID-19 Cases by State (Top 10)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Cases (Lakhs)")
    ax.legend(title="Wave", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_wave_breakdown.png"))
    plt.close()


# ── Plot 4: Case Fatality Rate by State ──────────────────────────────────────
def plot_cfr(summary):
    print("  Plot 4: CFR comparison...")
    df = summary.sort_values("overall_cfr", ascending=False)

    colors = ["#C0392B" if c > df["overall_cfr"].median() else "#27AE60"
              for c in df["overall_cfr"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["state"], df["overall_cfr"], color=colors, edgecolor="white")
    ax.axvline(df["overall_cfr"].median(), color="black",
               linestyle="--", linewidth=1.2, label=f"Median CFR ({df['overall_cfr'].median():.2f}%)")
    ax.set_xlabel("Case Fatality Rate (%)")
    ax.set_title("Case Fatality Rate by State", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.legend()

    red_patch  = mpatches.Patch(color="#C0392B", label="Above median (worse)")
    green_patch = mpatches.Patch(color="#27AE60", label="Below median (better)")
    ax.legend(handles=[red_patch, green_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_cfr_by_state.png"))
    plt.close()


# ── Plot 5: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(summary):
    print("  Plot 5: Correlation heatmap...")
    cols = [
        "cases_per_million", "deaths_per_million", "overall_cfr",
        "vaccination_coverage_pct", "literacy_rate",
        "hospital_beds_per_1000", "population_density",
        "urban_population_pct", "median_age",
    ]
    corr = summary[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation: Demographics vs COVID Outcomes", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_correlation_heatmap.png"))
    plt.close()


# ── Plot 6: Vaccination Coverage vs Death Rate ────────────────────────────────
def plot_vax_vs_deaths(summary):
    print("  Plot 6: Vaccination vs death rate...")
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        summary["vaccination_coverage_pct"],
        summary["deaths_per_million"],
        c=summary["literacy_rate"],
        cmap="YlOrRd", s=120, edgecolors="grey", linewidths=0.5, alpha=0.9
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Literacy Rate (%)", fontsize=10)

    for _, row in summary.iterrows():
        ax.annotate(row["state"], (row["vaccination_coverage_pct"], row["deaths_per_million"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points", alpha=0.8)

    # Trend line
    z = np.polyfit(summary["vaccination_coverage_pct"], summary["deaths_per_million"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(summary["vaccination_coverage_pct"].min(),
                         summary["vaccination_coverage_pct"].max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.2, alpha=0.6, label="Trend")

    ax.set_xlabel("Vaccination Coverage (%)")
    ax.set_ylabel("Deaths per Million")
    ax.set_title("Vaccination Coverage vs Deaths per Million\n(Color = Literacy Rate)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_vax_vs_deaths.png"))
    plt.close()


# ── Plot 7: Hospital Beds vs CFR ──────────────────────────────────────────────
def plot_beds_vs_cfr(summary):
    print("  Plot 7: Hospital beds vs CFR...")
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        summary["hospital_beds_per_1000"],
        summary["overall_cfr"],
        c=summary["population_density"],
        cmap="Blues", s=120, edgecolors="grey", linewidths=0.5, alpha=0.9
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Population Density (per sq km)", fontsize=10)

    for _, row in summary.iterrows():
        ax.annotate(row["state"], (row["hospital_beds_per_1000"], row["overall_cfr"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points", alpha=0.8)

    z = np.polyfit(summary["hospital_beds_per_1000"], summary["overall_cfr"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(summary["hospital_beds_per_1000"].min(),
                         summary["hospital_beds_per_1000"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1.2, alpha=0.7, label="Trend")

    ax.set_xlabel("Hospital Beds per 1,000 People")
    ax.set_ylabel("Case Fatality Rate (%)")
    ax.set_title("Healthcare Infrastructure vs Case Fatality Rate\n(Color = Population Density)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_beds_vs_cfr.png"))
    plt.close()


# ── Plot 8: Urban Population vs Cases per Million ─────────────────────────────
def plot_urban_vs_cases(summary):
    print("  Plot 8: Urbanisation vs cases per million...")
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(summary["urban_population_pct"], summary["cases_per_million"],
               color="#8E44AD", s=120, edgecolors="grey", linewidths=0.5, alpha=0.85)

    for _, row in summary.iterrows():
        ax.annotate(row["state"], (row["urban_population_pct"], row["cases_per_million"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points", alpha=0.8)

    z = np.polyfit(summary["urban_population_pct"], summary["cases_per_million"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(summary["urban_population_pct"].min(),
                         summary["urban_population_pct"].max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.2, alpha=0.6, label="Trend")

    ax.set_xlabel("Urban Population (%)")
    ax.set_ylabel("Cases per Million")
    ax.set_title("Urbanisation vs COVID-19 Cases per Million", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_urban_vs_cases.png"))
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    cleaned, summary, wave_df = load()

    print("Generating visualizations...")
    plot_national_timeseries(cleaned)
    plot_top_states(summary)
    plot_wave_breakdown(wave_df)
    plot_cfr(summary)
    plot_correlation_heatmap(summary)
    plot_vax_vs_deaths(summary)
    plot_beds_vs_cfr(summary)
    plot_urban_vs_cases(summary)

    print(f"\n✅ All 8 plots saved to /outputs/")
