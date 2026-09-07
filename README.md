````markdown
# 🏠 AmesValue AI

> AI-powered house price prediction using XGBoost, advanced preprocessing, and SHAP explainability.

[🚀 Live Demo](https://housepriceprediction-omyzczf5lc8wpuudkqpaum.streamlit.app/)

---

## 🎯 What is AmesValue AI?

AmesValue AI is an end-to-end machine learning application that estimates
the selling price of a residential property from its characteristics.

The project goes beyond simply training a regression model. It covers the
complete ML workflow:

Data Cleaning → EDA → Feature Engineering → Preprocessing →
Model Comparison → Hyperparameter Tuning → Explainability → Deployment

The final model uses **XGBoost Regression with a log-transformed target**.

---

## ✨ What Can the App Do?

- 🏠 Estimate a property's market value from 80 housing features
- ⚡ Generate predictions instantly through an interactive UI
- 🧠 Explain predictions using SHAP-based feature contributions
- 📊 Display model performance and validation metrics
- 🔍 Show the important characteristics influencing the prediction
- 🛡️ Handle categorical and numerical features automatically
- 📱 Provide a responsive, production-style interface
- ☁️ Run completely through Streamlit Cloud

---

## 🤖 Machine Learning

### Final Model

**XGBoost Regressor**

The final model was selected after comparing multiple regression approaches.

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.7443 | $37,579 | $27,552 |
| Random Forest | 0.8142 | $32,034 | $23,259 |
| Gradient Boosting | 0.8278 | $30,840 | $23,140 |
| XGBoost | 0.8346 | $30,224 | $22,691 |
| **XGBoost + log1p Target** | **0.8497** | **$28,813** | **$21,293** |

### Why XGBoost?

XGBoost was chosen because it can effectively model:

- Non-linear relationships
- Feature interactions
- Mixed feature distributions
- Complex relationships between house characteristics and price

---

## 📈 Target Transformation

House prices are strongly right-skewed.

Instead of directly predicting:

```text
SalePrice
````

the final model learns:

```text
log1p(SalePrice)
```

and converts the prediction back using:

```text
expm1(prediction)
```

This reduced target skewness and improved generalization.

**Target skewness**

```text
Before transformation : ~2.05
After log1p            : ~0.23
```

The log-target XGBoost model achieved:

```text
Holdout R² : 0.8497
CV R²      : 0.8172 ± 0.0226
MAE        : ~$21.3K
RMSE       : ~$28.8K
```

---

## 🧠 Explainable AI

AmesValue AI doesn't just output a number.

The application uses **SHAP (SHapley Additive exPlanations)** to show which
features contribute to an individual prediction.

For example, the model can identify that characteristics related to:

* Basement quality
* Kitchen quality
* Exterior quality
* Neighborhood
* Fireplace
* Garage
* Central air

have strong influence on predictions.

This makes the model more interpretable instead of treating it as a black box.

> Feature importance indicates predictive contribution, not causation.

---

## ⚙️ ML Concepts Used

### Data Processing

* Missing-value handling
* Numerical feature processing
* Categorical feature encoding
* One-hot encoding
* Feature validation
* Outlier investigation

### Exploratory Data Analysis

* Correlation analysis
* Distribution analysis
* Skewness analysis
* Outlier detection
* Group-based price analysis
* Multicollinearity investigation

### Machine Learning

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression
* Ensemble learning
* Hyperparameter optimization
* Cross-validation
* Holdout validation

### Model Evaluation

* MAE
* MSE
* RMSE
* R²
* Residual analysis
* Actual vs Predicted analysis

### Explainable AI

* SHAP
* TreeExplainer
* Local feature contributions
* Global feature importance

### Deployment

* Streamlit
* Joblib model serialization
* Production inference pipeline
* Streamlit Cloud

---

## 🔄 Prediction Pipeline

```text
                    PROPERTY INPUT
                          │
                          ▼
              ┌─────────────────────┐
              │ Input Validation     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Preprocessing       │
              │                     │
              │ Numerical → Impute  │
              │ Categorical → OHE   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ XGBoost Regression  │
              │                     │
              │ log1p(SalePrice)    │
              └──────────┬──────────┘
                         │
                         ▼
                  expm1(prediction)
                         │
                         ▼
              ┌─────────────────────┐
              │ Estimated Property  │
              │ Value               │
              └──────────┬──────────┘
                         │
                         ▼
                 SHAP Explanation
```

---

## 🧩 Preprocessing Architecture

The application uses a **Scikit-learn Pipeline + ColumnTransformer** architecture.

### Numerical Features

```text
Numerical Data
      ↓
Median Imputation
      ↓
Model
```

### Categorical Features

```text
Categorical Data
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
      ↓
Model
```

This preprocessing is stored together with the model, ensuring that the same
transformation logic is used during inference.

### Why Pipeline?

It helps:

* Prevent data leakage
* Keep preprocessing and prediction synchronized
* Reproduce training transformations
* Simplify deployment
* Handle unseen categorical values safely

---

## 🔬 Model Validation

The final model was evaluated using a separate holdout set as well as
5-fold cross-validation.

### Final Performance

```text
Holdout R²       : 0.8497
5-Fold CV R²     : 0.8172
CV Std           : 0.0226
MAE              : $21,293
RMSE             : $28,813
```

The cross-validation score is lower than the holdout score, which is expected
because different train/validation splits can produce different performance.

The relatively small CV standard deviation indicates more consistent
performance across folds.

---

## 🎨 Application Interface

AmesValue AI was designed as an actual ML product rather than a basic
prediction notebook.

### Interface Highlights

* Dark glassmorphism UI
* Responsive layout
* Structured property input sections
* Interactive controls
* Model performance panel
* Prediction result card
* Property snapshot
* AI-generated model insights
* SHAP-powered explanations
* Input validation
* Reset / New Valuation workflow

---

## 🏗️ Project Structure

```text
house-price-prediction/
│
├── model/
│   └── house_price_xgb_model.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Responsibilities

| File                           | Purpose                                               |
| ------------------------------ | ----------------------------------------------------- |
| `app.py`                       | Streamlit application and inference logic             |
| `house_price_xgb_model.pkl`    | Trained XGBoost pipeline                              |
| `house_price_prediction.ipynb` | Complete ML experimentation workflow                  |
| `requirements.txt`             | Python dependencies                                   |
| `.gitignore`                   | Prevents unnecessary/local files from being committed |
| `README.md`                    | Project documentation                                 |

---

## 🛠️ Tech Stack

**Language**

`Python`

**Data Science**

`NumPy` · `Pandas` · `Matplotlib` · `Seaborn`

**Machine Learning**

`Scikit-learn` · `XGBoost`

**Explainable AI**

`SHAP`

**Model Persistence**

`Joblib`

**Application**

`Streamlit`

**Deployment**

`Streamlit Community Cloud`

**Development**

`Jupyter Notebook` · `VS Code` · `Git` · `GitHub`

---

## 🚀 Run Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd house-price-prediction
```

Create a virtual environment:

```bash
python -m venv .myenv
```

Activate it on Windows:

```bash
.myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Live Application

### 🚀 Try AmesValue AI

**[Open the Live App →](https://housepriceprediction-omyzczf5lc8wpuudkqpaum.streamlit.app/)**

Enter the property characteristics and generate an estimated sale price.

---

## ⚠️ Important Note

This application is an **educational and portfolio machine-learning
project** built using the Ames Housing dataset.

Predictions are model estimates and should not be considered professional
real-estate appraisals or guaranteed market values.

---

## 📌 Key Takeaways

This project demonstrates an end-to-end understanding of:

```text
Data
 ↓
EDA
 ↓
Cleaning
 ↓
Preprocessing
 ↓
Feature Analysis
 ↓
Model Comparison
 ↓
Ensemble Learning
 ↓
XGBoost
 ↓
Hyperparameter Tuning
 ↓
Cross Validation
 ↓
Log Target Transformation
 ↓
SHAP Explainability
 ↓
Model Serialization
 ↓
Streamlit Application
 ↓
Cloud Deployment
```

---

## 👨‍💻 Author

### Mukul Chahar

B.Tech CSE — AI/ML

Interested in:

`Artificial Intelligence` · `Machine Learning` · `Data Science` · `AI Engineering`

---

⭐ **If you found this project useful, consider giving the repository a star.**

```

### Why I prefer this version

This one is structured so a recruiter can **scan it quickly**:

**1. What is it?** → immediately  
**2. What can it do?** → app capabilities  
**3. How good is the model?** → metrics  
**4. What ML concepts did you actually use?** → technical depth  
**5. How does it work?** → pipeline  
**6. What technologies?** → stack  
**7. Can I run it?** → setup  
**8. Can I try it?** → live app  

And importantly, it describes **what we actually implemented** rather than filling the README with generic ML terminology.
```
