```markdown
# 🏠 AmesValue AI — End-to-End Property Valuation Engine

<p align="left">
  <a href="https://housepriceprediction-omyzczf5lc8wpuudkqpaum.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Model-XGBoost_Regression-008080?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/Holdout_R²-0.8497-brightgreen?style=for-the-badge" alt="Holdout R2">
</p>

An end-to-end Machine Learning web application that predicts residential property values from 80 housing features using an optimized **XGBoost Regressor** with **log-transformed target estimation** and **SHAP-driven interpretability**.

---

### ⚡ Quick Links
[🚀 Launch Interactive App](https://housepriceprediction-omyzczf5lc8wpuudkqpaum.streamlit.app/) • [📊 Model Benchmarks](#-model-benchmarks) • [🧠 Explainability](#-explainable-ai-shap) • [🛠️ Local Setup](#️-quick-start)

---

## 🎯 Executive Summary

| Category | Implementation Details |
| :--- | :--- |
| **Objective** | Predict house sale prices with high variance resilience and feature interpretability |
| **Dataset** | Ames Housing Dataset (80 features: nominal, ordinal, discrete, continuous) |
| **Core Architecture** | Scikit-learn Pipeline + Custom Median/Mode Imputation + One-Hot Encoding |
| **Best Model** | **XGBoost with Target Transformation** (`log1p` / `expm1`) |
| **Primary Metrics** | **R²: 0.8497** \| **MAE: $21,293** \| **RMSE: $28,813** |
| **Inference Interface** | Streamlit Cloud featuring dark-glass UI & dynamic SHAP waterfall plots |

---

## 📊 Model Benchmarks

Five regression algorithms were systematically benchmarked on identical cross-validation splits. Target transformation via $\log(1 + y)$ yielded a **~7.5% drop in MAE** by penalizing percentage errors rather than raw dollar variance.

| Model Pipeline | Holdout $R^2$ | RMSE | MAE | Status |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Linear Regression | 0.7443 | $37,579 | $27,552 | Evaluated |
| Random Forest (Default) | 0.8142 | $32,034 | $23,259 | Evaluated |
| Gradient Boosting Regressor | 0.8278 | $30,840 | $23,140 | Evaluated |
| XGBoost (Untransformed) | 0.8346 | $30,224 | $22,691 | Evaluated |
| **XGBoost + $\log(1+y)$ Target** | **0.8497** | **$28,813** | **$21,293** | **Production Selected** |

> **Cross-Validation Stability:**  
> The production pipeline achieved a 5-fold CV score of **$R^2 = 0.8172 \pm 0.0226$**, demonstrating strong out-of-fold generalization across differing price distributions.

---

## 🔄 End-to-End System Architecture


```

```
                           RAW INPUT (80 Features)
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │    Pipeline Preprocessing (Scikit)    │
                  ├───────────────────┬───────────────────┤
                  │ Numerical Columns │ Categorical Cols  │
                  │ ➔ Median Impute   │ ➔ Mode Impute     │
                  │                   │ ➔ One-Hot Encode  │
                  └───────────────────┴───────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │         XGBoost Regressor Core        │
                  │       Estimates: log1p(SalePrice)     │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │       Inverse Transform: expm1        │
                  └───────────────────┬───────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
   ┌──────────────────────┐                     ┌────────────────────────┐
   │   Point Prediction   │                     │  SHAP TreeExplainer    │
   │  e.g., "$243,500"    │                     │ Local Attribution Plot │
   └──────────────────────┘                     └────────────────────────┘

```

```

---

## 🧠 Explainable AI (SHAP)

Rather than treating gradient boosting as an uninterpretable system, AmesValue AI incorporates **TreeSHAP** to quantify exact dollar-value feature attributions per transaction:

* **Global Drivers:** Overall quality (`OverallQual`), Ground living area (`GrLivArea`), Neighborhood affinity, and Total Basement Area dictate primary price brackets.
* **Local Attribution:** Every inference displays individual push/pull factors, exposing how specific amenities (e.g., modern kitchen, remodeled garage) offset baseline market values.

---

## 🛠️ Technical Stack


```

Runtime:         Python 3.10+
Core ML:         Scikit-learn • XGBoost
Interpretability:SHAP (SHapley Additive exPlanations)
Data Processing: Pandas • NumPy
Visual Analytics:Matplotlib • Seaborn
Serving & UI:    Streamlit Community Cloud
Artifact Storage:Joblib (Frozen Serialization)

```

<details>
<summary>📂 <strong>Click to view Repository Layout</strong></summary>

```bash
house-price-prediction/
├── app.py                          # Streamlit application entry point & inference logic
├── requirements.txt                # Production dependency declarations
├── .gitignore                      # Git exclusion rules
├── README.md                       # Repository documentation
├── model/
│   └── house_price_xgb_model.pkl   # Serialized pipeline (preprocessor + model weights)
└── notebooks/
    └── house_price_prediction.ipynb # Complete research, EDA, tuning & validation notebook

```

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/](https://github.com/)<your-username>/house-price-prediction.git
cd house-price-prediction

# Create isolated environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

```

### 2. Install Dependencies & Launch

```bash
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

```

---

## 👨‍💻 Engineering & Ownership

**Mukul Chahar**

*B.Tech in Computer Science Engineering (Specialization: AI & Machine Learning)*

Focus Areas: Applied Machine Learning • Predictive Systems • Production MLOps

```

***

### Key Visual Upgrades Applied
* **Top Ribbon Badges:** Replaced walls of bullet points with badges for key tech tags and model validation scores.
* **Executive Summary Matrix:** Formatted the problem, dataset, target trick, and results into a scannable table right at the top.
* **Streamlined Pipeline Diagram:** Cleaned up the ASCII flow so an engineering manager or recruiter can see the handling of data leakage, transformation, and SHAP output in 5 seconds.
* **Foldable Directory Structure:** Wrapped the file tree in an HTML `<details>` toggle so it doesn't take up vertical real estate.

```
