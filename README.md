Employee Promotion Prediction (Machine Learning Project)
Overview

This project predicts whether an employee is likely to be promoted in the next review cycle based on their profile, training history, performance ratings, and other HR factors.
It was built end-to-end as a practical machine learning workflow — starting from data exploration and feature engineering through model selection, hyperparameter tuning, and deployment.

The goal was not just to achieve good model accuracy, but to build a production-ready pipeline that could realistically be deployed as an API or interactive demo.

Key Features

Exploratory Data Analysis (EDA):
Identified department-wise promotion trends, education impact, regional variations, and correlations among training and performance metrics.

Feature Engineering:
Cleaned missing data, handled outliers, encoded categorical features, capped skewed variables, and created frequency-based features for regional imbalance.

Model Training & Tuning:
Experimented with multiple algorithms — Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
LightGBM gave the best balance of precision and recall after fine-tuning and threshold optimisation.

Threshold Calibration:
Instead of relying on a 0.5 cutoff, the model dynamically adjusts its probability threshold (≈0.24) to balance recall and precision for a realistic promotion rate.

Deployment Ready:

REST API built with FastAPI

Gradio UI for quick, no-code testing

Modular and version-controlled pipeline for easy retraining and inference

🧩 Tech Stack

Python 3.12

Libraries: pandas, numpy, scikit-learn, LightGBM, joblib

Serving: FastAPI, Uvicorn

Interface: Gradio

Version Control: Git + GitHub

📂 Project Structure
employee_promotion_project/
│
├── api/                     # FastAPI service for predictions
│   └── app.py
│
├── ui/                      # Gradio UI for demo and testing
│   └── gradio_app.py
│
├── models/                  # Saved model artefacts
│   ├── final_lgbm_model.joblib
│   ├── preprocessor_state.json
│   └── decision_threshold.json
│
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md

🧪 How to Run Locally
1. Clone the repository
git clone https://github.com/SarbjitK88/employee_promotion_ML.git
cd employee_promotion_ML

2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Gradio UI (easiest way to test)
python ui/gradio_app.py


Then open the link displayed (usually http://127.0.0.1:7860).

Can also  be tested by running the FastAPI service

  python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload


Open http://127.0.0.1:8000/docs
 to access the Swagger interface.

📈 Model Performance
Metric	Value
PR-AUC	0.568
ROC-AUC	0.827
F1-Score (balanced threshold 0.246)	0.536

LightGBM was selected as the final model due to its balance between interpretability, scalability, and accuracy.

🔍 Future Improvements
Integrate CI/CD workflow with automated retraining.
Deploy on Azure or AWS with a managed endpoint.

🧑‍💻 Author
Sarbjit Kaur
Data Scientist | AL/ ML Engineer
📍 United Kingdom
