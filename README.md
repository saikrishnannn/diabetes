Diabetes Risk Screening from Lifestyle and Health Indicators

A Fair, Calibrated, and Cost-Sensitive Prediction Approach

📌 Project Overview

Mass laboratory screening (HbA1c or fasting plasma glucose) across entire populations is resource-intensive. This project develops a low-cost, accountable machine learning screening aid that estimates diabetes risk using easily collected self-reported survey indicators (e.g., blood pressure, BMI, age, general health status).

Rather than chasing vanity metrics like accuracy—which is misleading on imbalanced clinical data—this project addresses critical production challenges:

Threshold Optimization: Moving beyond default 0.5 decision thresholds to cost-sensitive clinical triage.

Probability Calibration: Correcting distorted model confidence scores for reliable risk communication.

Subgroup Fairness: Auditing and quantifying performance disparities across age, sex, and socioeconomic groups.

Explainability: Providing SHAP-based global and local decision audits for clinicians.

Medical Disclaimer: This tool is designed strictly as a non-invasive pre-screening triage aid, not a diagnostic device. A negative screening result should never replace clinical judgment or confirmatory laboratory diagnostics.

📊 Dataset & Preprocessing

Source: US CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 (diabetes_binary_health_indicators_BRFSS2015.csv).

Initial Size: 253,680 respondents across 21 feature indicators.

Leakage Prevention & De-duplication:

24,206 exact duplicate rows (9.5%) were identified.

Experimentation proved that retaining identical rows led to artificial train/test twin contamination.

De-duplicating prior to splitting yielded an honest evaluation dataset of 229,474 observations (15.29% prevalence).

Partitioning: Stratified 60/20/20 split (137,684 train / 45,895 validation / 45,895 test).

🔬
