from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================================
# LOAD FINAL MODEL
# =========================================
bundle = joblib.load(
    r"C:\Users\ADMIN\Desktop\zamarano_prdctn_tool\models\ESBL_TWO_PHASE_FINAL.joblib"
)

phase1_model = bundle["phase1_model"]
phase2_model = bundle["phase2_model"]
phase1_threshold = bundle["phase1_threshold"]
phase2_threshold = bundle["phase2_threshold"]
predictors = bundle["predictors"]

# =========================================
# HOME PAGE
# =========================================
@app.route("/")
def home():
    return render_template("index.html", features=predictors)

# =========================================
# PREDICTION ENGINE
# =========================================
@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------
    # GET INPUTS
    # -----------------------------
    input_data = []

    for feature in predictors:
        value = request.form.get(feature)
        input_data.append(float(value))

    X = np.array(input_data).reshape(1, -1)

    # -----------------------------
    # PHASE 1 PREDICTION
    # -----------------------------
    phase1_prob = phase1_model.predict_proba(X)[0][1]

    if phase1_prob < phase1_threshold:
        return render_template(
            "result.html",
            result="LOW RISK - No ESBL suspicion",
            phase1_prob=round(phase1_prob, 3)
        )

    # -----------------------------
    # PHASE 2 PREDICTION
    # -----------------------------
    phase2_prob = phase2_model.predict_proba(X)[0][1]

    if phase2_prob >= phase2_threshold:
        final_result = "ESBL CONFIRMED (HIGH CONFIDENCE)"
    else:
        final_result = "ESBL POSSIBLE (LOWER CONFIDENCE)"

    return render_template(
        "result.html",
        result=final_result,
        phase1_prob=round(phase1_prob, 3),
        phase2_prob=round(phase2_prob, 3)
    )

# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)