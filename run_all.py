"""
run_all.py
----------
Master script — runs the full analysis pipeline end to end.
Usage: python run_all.py
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")

steps = [
    ("01_data_ingestion.py",   "Data ingestion"),
    ("02_data_cleaning.py",    "Data cleaning & feature engineering"),
    ("03_eda_visualizations.py","EDA & visualizations"),
    ("04_insights.py",         "Statistical insights report"),
]

def run(script, label):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    path = os.path.join(SCRIPTS_DIR, script)
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Failed at: {script}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n🚀 India COVID-19 Analysis Pipeline Starting...\n")
    for script, label in steps:
        run(script, label)
    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE!")
    print("  📁 Visualizations → /outputs/*.png")
    print("  📄 Insights report → /outputs/insights_report.txt")
    print("="*60 + "\n")
