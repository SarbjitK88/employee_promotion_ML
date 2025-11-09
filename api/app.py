from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib, json, os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Employee Promotion Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

ART_DIR = "./models"
MODEL_PATH = os.path.join(ART_DIR, "final_lgbm_model.joblib")
THRESH_PATH = os.path.join(ART_DIR, "decision_threshold.json")
STATE_PATH = os.path.join(ART_DIR, "preprocessor_state.json")

model = joblib.load(MODEL_PATH)
with open(THRESH_PATH) as f:
    DECISION_THRESHOLD = float(json.load(f)["threshold"])
with open(STATE_PATH) as f:
    STATE = json.load(f)

CAT_LOW = STATE["cat_low"]
NUM_BASE = STATE["num_base"]

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["education"] = df["education"].fillna(STATE["edu_mode"])
    df["prev_rating_missing"] = df["previous_year_rating"].isna().astype(int)
    df["previous_year_rating"] = df["previous_year_rating"].fillna(STATE["prev_med"])
    df["length_of_service"] = np.clip(df["length_of_service"], None, STATE["cap_service"])
    df["no_of_trainings"] = np.clip(df["no_of_trainings"], None, STATE["cap_trainings"])
    df["region_freq"] = df["region"].map(STATE["region_freq"]).fillna(0).astype(float)
    ohe = pd.get_dummies(df[CAT_LOW], drop_first=False)
    ohe = ohe.reindex(columns=STATE["ohe_cols"], fill_value=0)
    num_block = df[STATE["num_cols"]]
    out = pd.concat([num_block.reset_index(drop=True), ohe.reset_index(drop=True)], axis=1)
    return out[STATE["feature_order"]]

class EmployeeRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    employee_id: Optional[int] = None
    department: str
    region: str
    education: Optional[str] = None
    gender: str
    recruitment_channel: str
    awards_won_: int = Field(..., alias="awards_won?")
    age: int
    length_of_service: int
    avg_training_score: float
    previous_year_rating: Optional[float] = None
    no_of_trainings: int

class PredictRequest(BaseModel):
    records: List[EmployeeRecord]

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in req.records])
    for col in (CAT_LOW + ["region"] + NUM_BASE):
        if col not in df.columns:
            df[col] = np.nan
    X = preprocess_df(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    results = []
    for i in range(len(df)):
        results.append({
            "employee_id": None if pd.isna(df.loc[i, "employee_id"]) else int(df.loc[i, "employee_id"]),
            "probability": float(proba[i]),
            "prediction": int(pred[i]),
        })
    return {"threshold": DECISION_THRESHOLD, "results": results}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}
