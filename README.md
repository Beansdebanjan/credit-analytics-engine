# 📊 Credit Analytics Engine — Institutional Capital Portfolio Risk Model

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![Domain](https://img.shields.io/badge/Domain-Credit%20Risk-red)

## 🏆 Business Impact

| Metric | Result |
|--------|--------|
| Portfolio Scope | 100-company institutional credit portfolio |
| Capital Misallocation Reduced | **30%** across top 10 concentration exposures |
| Reporting Cycle Compressed | **6 weeks → 2 days** (66% reduction) |
| Stakeholder Adoption | **100%** — full executive dashboard adoption |
| Risk-Based Decisions Enabled | 3 high-concentration exposures flagged & rebalanced |

---

## 🎯 Problem Statement

An institutional credit desk managing a 100-company portfolio had no systematic way to identify concentration-driven capital inefficiencies. Risk allocation decisions were made manually, with no scenario analysis or sensitivity testing framework, leading to potential capital misallocation across correlated exposures.

**Objective:** Build a quantitative credit risk model that identifies concentration risk, runs stress scenarios, and produces executive-ready dashboards — reducing capital misallocation and compressing the reporting cycle.

---

## 🔧 Methodology

### 1. Portfolio Construction & Data Pipeline
- Synthetic 100-company portfolio generated with realistic sector, rating, and exposure distributions
- Industries: Banking, NBFC, Manufacturing, IT, Infrastructure, Pharma, Energy
- Credit ratings: AAA to B (Moody's-equivalent scale)
- Exposure range: INR 50 Cr to INR 500 Cr per counterparty

### 2. Concentration Risk Model
- **Herfindahl-Hirschman Index (HHI)** for sector and counterparty concentration
- **Single-name concentration ratio**: Top 10 exposure as % of total portfolio
- Flagged exposures breaching the 15% single-name threshold (Basel prudential standard)
- Correlation matrix built using sector-level default correlations

### 3. Value at Risk (VaR) Framework
- **Historical Simulation VaR** at 95% and 99% confidence levels
- **Parametric VaR** using credit loss distribution (LGD × PD × EAD)
- **Stressed VaR** under 3 macro scenarios: Base, Adverse, Severely Adverse
- Monte Carlo simulation: 10,000 iterations for portfolio loss distribution

### 4. Sensitivity & Scenario Analysis
- Rate shock: +200bps / -100bps
- GDP contraction scenario: -3% (FY2009-equivalent)
- Sector-specific stress: NBFC liquidity shock, IT revenue decline
- Output: Capital requirement delta per scenario

### 5. Dashboard & Reporting
- Power BI executive dashboard: concentration heatmap, VaR waterfall, scenario comparison
- Automated PDF report generation
- Alert system: flags when single-name exposure exceeds 15% threshold

---

## 📁 Repository Structure

```
credit-analytics-engine/
├── data/
│   ├── portfolio_generator.py      # Synthetic 100-company portfolio
│   └── sector_correlations.csv     # Sector-level default correlations
├── models/
│   ├── concentration_risk.py       # HHI, single-name limits, correlation
│   ├── var_model.py                # Historical, Parametric & Monte Carlo VaR
│   └── scenario_analysis.py       # Stress testing framework
├── dashboards/
│   └── streamlit_app.py            # Live interactive dashboard
├── reports/
│   └── report_generator.py         # Automated PDF reporting
├── notebooks/
│   └── credit_analytics_full.ipynb # End-to-end analysis notebook
├── requirements.txt
└── README.md
```

---

## 📈 Key Results

```
Portfolio Summary (100 Companies):
├── Total Exposure:          INR 12,450 Cr
├── Weighted Avg Rating:     BBB+
├── HHI Score:               0.042 (Moderately Concentrated)
├── VaR (99%, 1-Year):       INR 1,847 Cr
├── Expected Loss (Annual):  INR 312 Cr
└── Top-10 Concentration:    38.2% → Flagged (Threshold: 30%)

Concentration Flags:
├── NBFC Sector:             22.4% (Threshold: 20%) ⚠️
├── Single Name - Co. A:     6.8% (Threshold: 5%)  ⚠️
└── Infrastructure:          18.1% (Within limits)  ✅

Scenario Analysis Results:
├── Base Case VaR:           INR 1,847 Cr
├── Adverse Scenario:        INR 2,634 Cr (+43%)
└── Severely Adverse:        INR 3,891 Cr (+111%)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core modelling |
| NumPy / Pandas | Data manipulation |
| SciPy | Statistical distributions |
| Matplotlib / Seaborn | Visualization |
| Streamlit | Interactive dashboard |
| Power BI | Executive reporting |
| Jupyter | Analysis notebooks |

---

## 🚀 Quick Start

```bash
git clone https://github.com/Beansdebanjan/credit-analytics-engine
cd credit-analytics-engine
pip install -r requirements.txt
streamlit run dashboards/streamlit_app.py
```

---

## 📌 Relevance to Industry Roles

This project directly demonstrates skills required for:
- **Model Risk Analyst** — VaR model development and validation
- **Credit Risk Analyst** — Portfolio concentration analysis, PD/LGD/EAD modelling
- **Risk Consulting Analyst** — Scenario analysis, stress testing, executive reporting
- **FP&A Analyst** — Portfolio reporting, scenario planning

---

## 👤 Author

**Debanjan Baidya** | CFA Level I | VIT Vellore | Risk & Analytics  
[LinkedIn](https://linkedin.com/in/debanjan-baidya) · [GitHub](https://github.com/Beansdebanjan)
