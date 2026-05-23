# Insurance Risk Analytics — AlphaCare Insurance Solutions

**10 Academy KAIM9 | Week 3 Challenge**

End-to-end insurance risk analytics and predictive modelling pipeline for AlphaCare Insurance Solutions (ACIS), covering EDA, A/B hypothesis testing, data version control, and risk-based premium pricing models on 18 months of South African auto-insurance data (Feb 2014 – Aug 2015).

---

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── data/                         # Tracked by DVC — NOT committed to Git
│   └── insurance_data.csv
├── notebooks/
│   ├── 01_eda.ipynb              # Task 1 — Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb  # Task 3 — A/B Hypothesis Testing
│   └── 03_modeling.ipynb         # Task 4 — Predictive Modeling
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing utilities
│   ├── eda_utils.py              # Reusable EDA helper functions
│   ├── hypothesis_tests.py       # Statistical test wrappers
│   └── modeling.py               # Model training and evaluation
├── reports/
│   └── final_report.md
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
├── .dvc/                         # DVC metadata
├── dvc.yaml                      # DVC pipeline stages
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/insurance-risk-analytics.git
cd insurance-risk-analytics
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Reproduce the data pipeline (DVC)
```bash
dvc pull
```
This fetches the versioned dataset from the configured DVC remote. See the **Data Version Control** section below for details.

---

## Data

The dataset covers auto-insurance policies issued by ACIS from **February 2014 to August 2015**. It includes policy, client, vehicle, plan, and claims information (~50 columns). Place the raw file at `data/insurance_data.csv`.

**Key derived metrics:**
- `LossRatio = TotalClaims / TotalPremium`
- `Margin = TotalPremium - TotalClaims`

The `data/` directory is listed in `.gitignore` and tracked exclusively via DVC.

---

## Data Version Control

Two data versions are tracked:

| Version | File | Description |
|---------|------|-------------|
| v1 | `data/insurance_data.csv` | Raw scraped data — no modifications |
| v2 | `data/insurance_data_cleaned.csv` | After missing-value handling, type casting, and feature engineering |

```bash
# Push data to local remote
dvc push

# Switch between versions via Git tags
git checkout v1-raw-data
dvc checkout
```

---

## Running the Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## CI/CD

GitHub Actions runs on every push to any branch:
- `flake8` linting
- `pytest` test suite

See `.github/workflows/ci.yml`.

---

## Tasks

| Task | Branch | Status |
|------|--------|--------|
| Task 1 — EDA | task-1 | ✅ |
| Task 2 — DVC | task-2 | ⬜ |
| Task 3 — Hypothesis Testing | task-3 | ⬜ |
| Task 4 — Modeling | task-4 | ⬜ |

---

## Author

AlphaCare Insurance Solutions Analytics Team | 10 Academy KAIM9 | May 2026
