import json, joblib, os
import gradio as gr
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

# --- load artefacts ---
model = joblib.load(os.path.join(MODEL_DIR, "final_lgbm_model.joblib"))
with open(os.path.join(MODEL_DIR, "decision_threshold.json")) as f:
    THRESH = float(json.load(f)["threshold"])
with open(os.path.join(MODEL_DIR, "preprocessor_state.json")) as f:
    STATE = json.load(f)

CAT_LOW = STATE["cat_low"]
NUM_BASE = STATE["num_cols"][:5]  # ["age","length_of_service","avg_training_score","previous_year_rating","no_of_trainings"]

def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # required columns
    needed = CAT_LOW + ["region"] + ["age","length_of_service","avg_training_score","previous_year_rating","no_of_trainings"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # education
    df["education"] = df["education"].fillna(STATE["edu_mode"])

    # prev rating flag + impute
    df["prev_rating_missing"] = df["previous_year_rating"].isna().astype(int)
    df["previous_year_rating"] = df["previous_year_rating"].fillna(STATE["prev_med"])

    # caps
    df["length_of_service"] = np.clip(df["length_of_service"], None, STATE["cap_service"])
    df["no_of_trainings"]   = np.clip(df["no_of_trainings"],   None, STATE["cap_trainings"])

    # region frequency
    df["region_freq"] = df["region"].map(STATE["region_freq"]).fillna(0).astype(float)

    # one-hot for small categoricals
    ohe = pd.get_dummies(df[CAT_LOW], drop_first=False)
    ohe = ohe.reindex(columns=STATE["ohe_cols"], fill_value=0)

    num_block = df[STATE["num_cols"]]
    out = pd.concat([num_block.reset_index(drop=True),
                     ohe.reset_index(drop=True)], axis=1)
    return out[STATE["feature_order"]]

def predict_one(department, region, education, gender, recruitment_channel,
                awards_won, age, length_of_service, avg_training_score,
                previous_year_rating, no_of_trainings):
    # build single-row frame (note the exact column name for awards)
    row = pd.DataFrame([{
        "department": department.strip(),
        "region": region.strip(),
        "education": education if education else None,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "awards_won?": int(awards_won),
        "age": int(age),
        "length_of_service": int(length_of_service),
        "avg_training_score": float(avg_training_score),
        "previous_year_rating": None if previous_year_rating in ("", None) else float(previous_year_rating),
        "no_of_trainings": int(no_of_trainings),
    }])

    Xp = preprocess_df(row)
    proba = float(model.predict_proba(Xp)[:,1][0])
    pred = int(proba >= THRESH)
    return round(proba, 4), pred

def predict_csv(file):
    df = pd.read_csv(file.name)
    Xp = preprocess_df(df)
    proba = model.predict_proba(Xp)[:,1]
    pred = (proba >= THRESH).astype(int)
    out = df.copy()
    out["promotion_probability"] = proba
    out["is_promoted_pred"] = pred
    # save
    save_path = os.path.join(MODEL_DIR, "predictions.csv")
    out.to_csv(save_path, index=False)
    return save_path

with gr.Blocks(title="Employee Promotion Predictor") as demo:
    gr.Markdown("# Employee Promotion Predictor")

    with gr.Tab("Single Prediction"):
        with gr.Row():
            department = gr.Textbox(label="department (e.g., Sales & Marketing)")
            region = gr.Textbox(label="region (e.g., region_2)")
            education = gr.Textbox(label="education (Bachelor's/Master's/Unknown)")
        with gr.Row():
            gender = gr.Dropdown(choices=["m","f"], value="m", label="gender")
            recruitment_channel = gr.Textbox(label="recruitment_channel (other/sourcing/referred)", value="other")
            awards_won = gr.Radio(choices=[0,1], value=0, label="awards_won?")
        with gr.Row():
            age = gr.Number(value=33, label="age", precision=0)
            length_of_service = gr.Number(value=5, label="length_of_service", precision=0)
            no_of_trainings = gr.Number(value=1, label="no_of_trainings", precision=0)
        with gr.Row():
            avg_training_score = gr.Number(value=68, label="avg_training_score")
            previous_year_rating = gr.Textbox(value="4", label="previous_year_rating (blank if unknown)")

        btn = gr.Button("Predict")
        proba = gr.Number(label="promotion_probability")
        pred = gr.Number(label="is_promoted_pred (0/1)")
        btn.click(
            predict_one,
            inputs=[department, region, education, gender, recruitment_channel, awards_won,
                    age, length_of_service, avg_training_score, previous_year_rating, no_of_trainings],
            outputs=[proba, pred]
        )

    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV containing these columns: "
            "`department, region, education, gender, recruitment_channel, awards_won?, age, length_of_service, avg_training_score, previous_year_rating, no_of_trainings`"
        )
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        file_out = gr.File(label="Download Predictions CSV")
        run_batch = gr.Button("Run Batch Inference")
        run_batch.click(predict_csv, inputs=[file_in], outputs=[file_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
