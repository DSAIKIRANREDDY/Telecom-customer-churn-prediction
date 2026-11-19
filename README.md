# Telecom Customer Churn Prediction

An end-to-end Machine Learning application that predicts customer churn using customer demographics, subscribed services, contract details, and billing information. The project includes model development, hyperparameter tuning, threshold optimization, model interpretability, SHAP explainability, a FastAPI inference backend, and a Streamlit frontend.

## Project Overview

Customer churn occurs when customers discontinue a company's services. Identifying customers who are likely to leave helps businesses take proactive retention measures.

This project builds a binary classification system to predict whether a telecom customer will churn (`Yes`) or remain (`No`).

### Project Objectives

- Analyze customer data and identify churn patterns.
- Build and compare multiple classification models.
- Optimize model hyperparameters using Optuna.
- Optimize the classification threshold based on validation performance.
- Interpret model predictions using feature importance and permutation importance.
- Explain individual and batch predictions using SHAP.
- Develop a REST API using FastAPI.
- Build an interactive application using Streamlit.

## Dataset

**Dataset:** IBM Telco Customer Churn

- Total records: 7,043
- Original columns: 21
- Target variable: `Churn`
- Prediction type: Binary classification

| Target | Customer count |
|---|---:|
| No | 5,174 |
| Yes | 1,869 |

The dataset contains customer information such as:

- Demographics: gender, SeniorCitizen, Partner, Dependents.
- Account information: tenure, Contract, PaymentMethod, PaperlessBilling.
- Services: InternetService, OnlineSecurity, OnlineBackup, TechSupport, StreamingTV, StreamingMovies.
- Billing: MonthlyCharges, TotalCharges.

`customerID` is used as an identifier and is not used as a model feature.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Churn distribution analysis
- Numerical and categorical feature analysis
- Correlation analysis

### 2. Data Validation

- Schema validation
- Data type validation
- Missing value checks
- Duplicate checks
- Target validation

### 3. Data Preprocessing

- Convert `TotalCharges` into a numerical feature.
- Handle missing numerical values using median imputation.
- Handle missing categorical values using most-frequent imputation.
- Apply StandardScaler to numerical features.
- Apply OneHotEncoder to categorical features.

The preprocessing steps are integrated into a Scikit-learn pipeline.

### 4. Feature Engineering

The project uses a modular feature engineering component. The current implementation preserves the original feature columns without introducing additional engineered features.

### 5. Model Training and Comparison

The following classification algorithms were explored:

- Logistic Regression
- Random Forest
- XGBoost

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

### 6. Hyperparameter Optimization

Optuna was used to tune Logistic Regression and XGBoost.

Configuration:
- Optimization framework: Optuna
- Sampler: Tree-structured Parzen Estimator (TPE)
- Trials: 50 per model
- Cross-validation: 3-fold StratifiedKFold
- Optimization objective: Average Precision

### 7. Classification Threshold Optimization

The default classification threshold of 0.5 was evaluated and optimized using validation data.

The selected threshold is **0.65**, based on validation F1-score optimization.

The test dataset was kept separate for final evaluation.

### 8. Model Interpretability

Two feature-importance approaches were implemented:

- Built-in XGBoost feature importance
- Permutation feature importance

These approaches help understand which input features influence model performance and predictions.

### 9. SHAP Explainability

SHAP was implemented at two levels:

**Global and batch explainability**
- SHAP beeswarm plot
- Aggregated feature importance across customers

**Individual customer explainability**
- SHAP waterfall plot
- Feature-level contributions
- Positive and negative contributions to the model output

The explanations help users understand why a particular customer received a churn prediction.

## Final Model Performance

The tuned XGBoost model was evaluated on the held-out test dataset using the selected classification threshold of 0.65.

| Metric | Test performance |
|---|---:|
| Accuracy | 79.13% |
| Precision | 59.01% |
| Recall | 70.05% |
| F1-score | 64.06% |
| ROC-AUC | 84.76% |
| Average Precision | 66.73% |

**Confusion Matrix**

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 853 | 182 |
| Actual Yes | 112 | 262 |

The model identifies approximately 70% of actual churn customers on the held-out test set at the selected threshold.

## Application Architecture

The project separates model development, inference, API services, and frontend presentation.

```text
             Customer
                |
                v
        Streamlit Frontend
                |
          HTTP / JSON
                |
                v
          FastAPI Backend
                |
        -----------------
        |               |
        v               v
  Churn Predictor   SHAP Explainer
        |               |
        v               v
  XGBoost Pipeline  Explanations
        |               |
        -----------------
                |
                v
       Prediction Results
```

## Application Features

### 1. Dashboard

- Dataset overview
- Total customers
- Churned and retained customer counts
- Churn rate
- Model information
- Backend health status

### 2. Single Customer Prediction

- Enter customer information.
- Generate churn probability.
- View predicted churn class.
- View customer risk level: Low, Medium, or High.
- Generate an individual SHAP waterfall explanation.

### 3. Batch Customer Prediction

- Upload a customer CSV file.
- Generate predictions for multiple customers.
- View prediction results in a table.
- Download prediction results.
- Generate batch SHAP beeswarm and feature-importance visualizations.
- Select individual customers for SHAP waterfall explanations.

### 4. About

Project description, objectives, workflow, and technology stack.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| Testing | Pytest, HTTPX |
| Dependency Management | uv |
| Version Control | Git, GitHub |
| Data Versioning | DVC |

## Project Structure

```text
telecom-churn-prediction/
│
├── config/
│   ├── config.yaml
│   └── schema.yaml
│
├── DataInput/
│
├── data/
│   ├── interim/
│   ├── validated/
│   ├── processed/
│   └── featured/
│
├── artifacts/
│   ├── models/
│   ├── reports/
│   └── logs/
│
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── dashboard.py
│   │   ├── single_prediction.py
│   │   ├── batch_prediction.py
│   │   └── about.py
│   ├── services/
│   │   └── api_client.py
│   └── utils/
│
├── src/
│   └── telecom_churn/
│       ├── api/
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── entity/
│       ├── inference/
│       └── utils/
│
├── tests/
├── test/
├── Notebooks/
├── scripts/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## Local Setup and Execution

### Prerequisites

- Python 3.11
- uv
- Git

### 1. Clone the repository

```bash
git clone https://github.com/DSAIKIRANREDDY/Telecom-customer-churn-prediction.git

cd Telecom-customer-churn-prediction
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Prepare project data and model

Ensure the required dataset and trained model artifact are available locally.

The inference service expects the trained model at:

```text
artifacts/models/tuned/xgboost_tuned_model.pkl
```

Data and model artifacts are excluded from ordinary Git tracking and are intended to be managed separately. The DVC remote and complete artifact-retrieval workflow still need to be configured.

### 4. Start the FastAPI backend

```bash
uv run uvicorn telecom_churn.api.main:app --reload --app-dir src
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit frontend

Open another terminal from the project root:

```bash
uv run streamlit run frontend/app.py
```

The application will be available at:

```text
http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend health |
| GET | `/health/live` | Liveness check |
| GET | `/health/ready` | Readiness check |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch customer prediction |
| POST | `/explain` | Individual SHAP explanation |
| GET | `/explain/plot/{customer_id}` | Retrieve individual SHAP plot |
| POST | `/explain/batch` | Batch SHAP analysis |
| GET | `/explain/batch/plot/{plot_type}` | Retrieve batch SHAP plots |

Interactive API documentation is available at `/docs`.

## Testing

The project includes tests for:

- Data ingestion and validation
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Model tuning
- Threshold optimization
- Model interpretability
- SHAP explainability
- FastAPI endpoints
- Batch prediction and explainability

Run the test suite:

```bash
uv run pytest
```

## Future Improvements

- Complete DVC remote configuration and data versioning.
- Integrate MLflow for experiment tracking and model management.
- Dockerize the FastAPI backend and Streamlit frontend.
- Deploy the application.
- Implement CI/CD workflows.
- Add production monitoring and model-drift detection.

## Author

**Kiran Reddy**

GitHub: [DSAIKIRANREDDY](https://github.com/DSAIKIRANREDDY)

Project Repository: [Telecom Customer Churn Prediction](https://github.com/DSAIKIRANREDDY/Telecom-customer-churn-prediction)
