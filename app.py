# app.py  -  run with:  streamlit run app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BMI_BINS = [0, 18.5, 25, 30, 35, 40, 200]


def add_features(d):
    """Domain features; every one is tested before it is kept."""
    d = d.copy()
    d["BMI_cat"] = pd.cut(d["BMI"], BMI_BINS, labels=False).astype(int)
    d["Age_grp"] = np.select([d["Age"] <= 5, d["Age"] <= 9], [0, 1], 2)
    d["Cardio_score"] = (d["HighBP"] + d["HighChol"] +
                         d["HeartDiseaseorAttack"] + d["Stroke"])
    d["BMI_x_Age"] = d["BMI"] * d["Age"]
    return d


lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def apply_calibration(kind, cal, p):
    p = np.asarray(p, dtype=float)
    if kind == "platt":
        return cal.predict_proba(lg(p).reshape(-1, 1))[:, 1]
    return cal.predict(p)


bundle = joblib.load("diabetes_screening_model.joblib")
st.title("Diabetes Risk Screening Aid")
st.warning("Screening aid, NOT a diagnosis. Confirm with HbA1c or "
           "fasting glucose.")
c1, c2 = st.columns(2)
inp = {}
for col in ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]:
    inp[col] = int(c1.checkbox(col))
inp["Sex"] = int(c2.selectbox("Sex", ["Female", "Male"]) == "Male")
inp["BMI"] = c2.slider("BMI", 12, 60, 27)
inp["GenHlth"] = c2.slider("General health (1=excellent, 5=poor)", 1, 5, 3)
inp["MentHlth"] = c2.slider("Poor mental-health days (30)", 0, 30, 0)
inp["PhysHlth"] = c2.slider("Poor physical-health days (30)", 0, 30, 0)
inp["Age"] = c2.slider("Age band (1=18-24 ... 13=80+)", 1, 13, 7)
inp["Education"] = c2.slider("Education (1-6)", 1, 6, 4)
inp["Income"] = c2.slider("Income band (1-8)", 1, 8, 5)
if st.button("Estimate risk"):
    x = add_features(pd.DataFrame([inp]))[bundle["features"]]
    raw_p = bundle["model"].predict_proba(x)[:, 1]
    p = apply_calibration(bundle["kind"], bundle["calibrator"], raw_p)[0]
    st.metric("Estimated diabetes risk", f"{p:.1%}")
    st.write("Refer for lab test" if p >= bundle["threshold"]
             else "Below screening threshold")
