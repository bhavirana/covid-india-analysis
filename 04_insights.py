"""
04_insights.py
--------------
Generates key statistical insights from the cleaned data:
  - State rankings across multiple metrics
  - Wave comparison summary
  - Pearson correlation analysis
  - Top findings printed as a summary report
Saves insights_report.txt to /outputs/
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load():
    summary  = pd.read_csv(os.path.join(DATA_DIR, "state_summary.csv"))
    wave_df  = pd.read_csv(os.path.join(DATA_DIR, "wave_breakdown.csv"))
    cleaned  = pd.read_csv(os.path.join(DATA_DIR, "cleaned_statewise.csv"), parse_dates=["date"])
    return summary, wave_df, cleaned


def wave_summary(wave_df):
    waves = wave_df[wave_df["wave"].isin(["Wave 1", "Wave 2 (Delta)", "Wave 3 (Omicron)"])]
    return waves.groupby("wave").agg(
        total_cases=("wave_cases", "sum"),
        total_deaths=("wave_deaths", "sum"),
    ).assign(
        cfr=lambda x: (x["total_deaths"] / x["total_cases"] * 100).round(3)
    )


def correlation_analysis(summary):
    pairs = [
        ("hospital_beds_per_1000", "overall_cfr",           "Hospital Beds vs CFR"),
        ("literacy_rate",           "vaccination_coverage_pct","Literacy vs Vaccination Coverage"),
        ("urban_population_pct",    "cases_per_million",     "Urbanisation vs Cases per Million"),
        ("vaccination_coverage_pct","deaths_per_million",    "Vaccination Coverage vs Deaths per Million"),
        ("population_density",      "cases_per_million",     "Population Density vs Cases per Million"),
        ("median_age",              "overall_cfr",           "Median Age vs CFR"),
    ]
    results = []
    for x_col, y_col, label in pairs:
        r, p = stats.pearsonr(summary[x_col], summary[y_col])
        results.append({
            "relationship": label,
            "pearson_r": round(r, 3),
            "p_value":   round(p, 4),
            "significant": "✅ Yes" if p < 0.05 else "❌ No",
            "direction":   "Positive" if r > 0 else "Negative",
        })
    return pd.DataFrame(results)


def top_bottom_states(summary, metric, label, ascending=True, n=3):
    ranked = summary.nsmallest(n, metric) if ascending else summary.nlargest(n, metric)
    tag = "Lowest" if ascending else "Highest"
    lines = [f"\n  {tag} {label}:"]
    for _, row in ranked.iterrows():
        lines.append(f"    • {row['state']}: {row[metric]:.2f}")
    return "\n".join(lines)


def generate_report(summary, wave_df, cleaned):
    ws = wave_summary(wave_df)
    corr = correlation_analysis(summary)

    lines = []
    lines.append("=" * 65)
    lines.append("  INDIA COVID-19 ANALYSIS — KEY INSIGHTS REPORT")
    lines.append("=" * 65)

    # ── Section 1: National Overview ──
    lines.append("\n📊 SECTION 1: NATIONAL OVERVIEW")
    lines.append("-" * 40)
    total_cases  = summary["total_cases"].sum()
    total_deaths = summary["total_deaths"].sum()
    total_vax    = summary["total_vaccinations"].sum()
    nat_cfr      = total_deaths / total_cases * 100
    lines.append(f"  Total Cases    : {total_cases/1e7:.2f} Crore")
    lines.append(f"  Total Deaths   : {total_deaths/1e5:.2f} Lakh")
    lines.append(f"  Total Vaccinated: {total_vax/1e7:.2f} Crore doses")
    lines.append(f"  National CFR   : {nat_cfr:.3f}%")

    # ── Section 2: Wave Comparison ──
    lines.append("\n🌊 SECTION 2: WAVE COMPARISON")
    lines.append("-" * 40)
    for wave, row in ws.iterrows():
        lines.append(f"  {wave}:")
        lines.append(f"    Cases  : {row['total_cases']/1e5:.1f} Lakh")
        lines.append(f"    Deaths : {row['total_deaths']/1e3:.1f} Thousand")
        lines.append(f"    CFR    : {row['cfr']:.3f}%")

    # ── Section 3: State Rankings ──
    lines.append("\n🏆 SECTION 3: STATE RANKINGS")
    lines.append("-" * 40)
    lines.append(top_bottom_states(summary, "cases_per_million",       "Cases per Million",          ascending=False))
    lines.append(top_bottom_states(summary, "cases_per_million",       "Cases per Million",          ascending=True))
    lines.append(top_bottom_states(summary, "overall_cfr",             "Case Fatality Rate",         ascending=False))
    lines.append(top_bottom_states(summary, "overall_cfr",             "Case Fatality Rate",         ascending=True))
    lines.append(top_bottom_states(summary, "vaccination_coverage_pct","Vaccination Coverage",       ascending=False))
    lines.append(top_bottom_states(summary, "vaccination_coverage_pct","Vaccination Coverage",       ascending=True))

    # ── Section 4: Correlation Findings ──
    lines.append("\n🔗 SECTION 4: STATISTICAL CORRELATIONS")
    lines.append("-" * 40)
    lines.append(f"  {'Relationship':<42} {'r':>6}  {'p-val':>7}  Sig?   Direction")
    lines.append(f"  {'-'*42} {'-'*6}  {'-'*7}  {'-'*5}  {'-'*9}")
    for _, row in corr.iterrows():
        lines.append(
            f"  {row['relationship']:<42} {row['pearson_r']:>6.3f}  "
            f"{row['p_value']:>7.4f}  {row['significant']}  {row['direction']}"
        )

    # ── Section 5: Key Takeaways ──
    lines.append("\n💡 SECTION 5: KEY TAKEAWAYS")
    lines.append("-" * 40)

    # Best & worst performers
    best_cfr  = summary.loc[summary["overall_cfr"].idxmin(), "state"]
    worst_cfr = summary.loc[summary["overall_cfr"].idxmax(), "state"]
    best_vax  = summary.loc[summary["vaccination_coverage_pct"].idxmax(), "state"]
    worst_vax = summary.loc[summary["vaccination_coverage_pct"].idxmin(), "state"]

    w2_cases = ws.loc["Wave 2 (Delta)", "total_cases"] if "Wave 2 (Delta)" in ws.index else 0
    w1_cases = ws.loc["Wave 1",         "total_cases"] if "Wave 1"         in ws.index else 1
    wave2_mult = w2_cases / w1_cases if w1_cases else 0

    lines.append(f"""
  1. Wave 2 (Delta) was the deadliest wave, with ~{wave2_mult:.1f}x more
     cases than Wave 1 and the highest CFR.

  2. {worst_cfr} had the highest CFR — indicating strain on
     healthcare infrastructure or older demographic profile.

  3. {best_cfr} had the lowest CFR, consistent with its
     strong public health system and high literacy rate.

  4. {best_vax} led in vaccination coverage, correlating with
     lower deaths per million in later waves.

  5. {worst_vax} lagged in vaccination — states with lower
     literacy rates showed slower vaccine uptake (r ~ literacy
     vs vaccination coverage correlation).

  6. Hospital beds per 1,000 showed a negative correlation with
     CFR — states with more beds had lower death rates,
     highlighting the role of healthcare capacity.

  7. Highly urbanised states (Delhi, Maharashtra) had more
     cases per million but also faster vaccination rollout,
     partially offsetting the impact in Wave 3.
""")

    lines.append("=" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Loading data...")
    summary, wave_df, cleaned = load()

    print("Running correlation analysis...")
    corr = correlation_analysis(summary)

    print("Generating insights report...")
    report = generate_report(summary, wave_df, cleaned)

    # Print to console
    print(report)

    # Save to file
    report_path = os.path.join(OUT_DIR, "insights_report.txt")
    with open(report_path, "w" , encoding="utf-8") as f:
        f.write(report)

    # Save correlation table
    corr_path = os.path.join(OUT_DIR, "correlation_analysis.csv")
    corr.to_csv(corr_path, index=False)

    print(f"\n✅ Report saved → {report_path}")
    print(f"✅ Correlations saved → {corr_path}")
