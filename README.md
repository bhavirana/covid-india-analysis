# 🦠 India COVID-19 Impact Analysis & Demographic Insights

A data science project analyzing the three waves of COVID-19 across Indian states,
exploring how demographics, healthcare infrastructure, and vaccination speed
influenced outcomes.

---

## 📁 Project Structure

```
covid_india_analysis/
├── run_all.py                  ← Run entire pipeline with one command
├── scripts/
│   ├── 01_data_ingestion.py    ← Downloads & builds datasets
│   ├── 02_data_cleaning.py     ← Cleaning, feature engineering, wave tagging
│   ├── 03_eda_visualizations.py← 8 publication-quality charts
│   └── 04_insights.py          ← Statistical correlations & insights report
├── data/                       ← Auto-generated CSVs
└── outputs/                    ← Charts (PNG) + insights report (TXT)
```

---

## 📊 Visualizations Generated

| # | Plot | Key Question |
|---|------|-------------|
| 1 | National Daily Cases (Time Series) | How did the 3 waves differ in scale? |
| 2 | Top 10 States by Total Cases | Which states bore the highest burden? |
| 3 | Wave-wise Breakdown by State | Did the same states lead all waves? |
| 4 | CFR by State | Which states had the deadliest outbreaks? |
| 5 | Correlation Heatmap | What demographics correlate with outcomes? |
| 6 | Vaccination Coverage vs Deaths | Did faster vaccination save lives? |
| 7 | Hospital Beds vs CFR | Did infrastructure reduce fatality rates? |
| 8 | Urbanisation vs Cases per Million | Did dense cities get hit harder? |

---

## 🔍 Key Findings

- **Wave 2 (Delta)** was the most devastating — highest case counts and CFR
- **Bihar & UP** lagged in vaccination, correlating with higher CFR
- **Kerala** had low CFR despite high cases — strong health system effect
- **Hospital beds per 1,000** showed a statistically significant negative correlation with CFR
- **Literacy rate** positively correlated with vaccination uptake speed
- **Urbanisation** increased cases per million but also accelerated vaccine rollout

---

## ⚙️ Setup & Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn plotly folium scipy requests

# Run full pipeline
python run_all.py

# Or run individual steps
python scripts/01_data_ingestion.py
python scripts/02_data_cleaning.py
python scripts/03_eda_visualizations.py
python scripts/04_insights.py
```

---

## 📦 Data Sources

- **COVID-19 data**: Our World in Data (OWID) — India national data
- **State-level proportions**: Ministry of Health & Family Welfare (MoHFW) records
- **Demographics**: Census of India 2011, National Health Profile

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `SciPy` · `Requests`

---

## 💡 Future Extensions

- [ ] Add Streamlit dashboard for interactivity
- [ ] District-level analysis with Folium choropleth maps
- [ ] Time-series forecasting (Prophet) for future wave prediction
- [ ] ML model: predict state-level CFR from demographics
