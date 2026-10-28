# DIABETES RISK SCREENING FROM LIFESTYLE AND HEALTH INDICATORS
# Source : CDC BRFSS 2015 - Diabetes Health Indicators (Kaggle)
# ============================================================
import os
import json
import time
import inspect
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, brier_score_loss,
                             accuracy_score)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
DATA_PATH = "diabetes_binary_health_indicators_BRFSS2015.csv"
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"      # debugging on a subsample
R = {}                                      # every reported number lives here
T0 = time.time()

# ── GLOBAL STYLE ─────────────────────────────────────────────
PALETTE = {
    "primary": "#1B4F72", "secondary": "#2E86C1", "accent": "#E74C3C",
    "warm": "#E67E22", "teal": "#1ABC9C", "purple": "#8E44AD",
    "light": "#ECF0F1", "dark": "#1C2833", "muted": "#7F8C8D",
    "gridline": "#D5D8DC",
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#FDFEFE",
    "axes.edgecolor": PALETTE["gridline"], "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": PALETTE["gridline"],
    "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "DejaVu Sans", "text.color": PALETTE["dark"],
})


def style_axis(ax, title, xlabel="", ylabel="", subtitle=""):
    """Apply consistent professional axis styling."""
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=PALETTE["primary"], pad=14, loc="left")
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=PALETTE["muted"], va="bottom")
    ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["muted"], labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["muted"], labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ── HELPER FUNCTIONS ─────────────────────────────────────────
def metrics_at(y, p, thr):
    """Confusion-matrix based metrics at a probability threshold."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(thr=float(thr), tn=int(tn), fp=int(fp), fn=int(fn),
                tp=int(tp), accuracy=(tp + tn) / len(y),
                precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                fpr=fp / (fp + tn) if (fp + tn) else 0.0,
                fnr=fn / (fn + tp) if (fn + tp) else 0.0,
                flagged=int(tp + fp))


def recall_at_precision(y, p, prec_target=0.30):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr[:-1] >= prec_target
    return float(rc[:-1][ok].max()) if ok.any() else 0.0


def boot_ci(y, p, fn, B=300, seed=SEED):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn(y[idx], p[idx]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    chi2, p, _, _ = sps.chi2_contingency(tab, correction=False)
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (tab.values.sum() * k))), float(p)


def rank_biserial(x1, x0):
    u, p = sps.mannwhitneyu(x1, x0, alternative="two-sided")
    return float(2 * u / (len(x1) * len(x0)) - 1), float(p)


# ============================================================
# PHASE 1 — LOAD & CLEAN
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 1 : DATA LOADING & CLEANING")
print("=" * 60)
raw = pd.read_csv(DATA_PATH)
raw.columns = raw.columns.str.strip()
TARGET = "Diabetes_binary"
FEATS = [c for c in raw.columns if c != TARGET]
BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
          "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
          "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
          "Sex"]
ORDINAL = ["GenHlth", "Age", "Education", "Income"]
NUMERIC = ["BMI", "MentHlth", "PhysHlth"]

R["raw_rows"], R["raw_cols"] = raw.shape
R["missing_values"] = int(raw.isna().sum().sum())
R["raw_pos_rate"] = float(raw[TARGET].mean())
R["raw_pos_n"] = int(raw[TARGET].sum())
R["n_duplicates"] = int(raw.duplicated().sum())
print(f" Raw shape : {raw.shape}")
print(f" Missing values : {R['missing_values']}")
print(f" Positive rate (raw) : {R['raw_pos_rate']:.4f}")
print(f" Duplicate rows : {R['n_duplicates']:,}")

if QUICK:
    raw, _ = train_test_split(raw, train_size=50000, stratify=raw[TARGET],
                              random_state=SEED)

# ── Duplicate-removal experiment ─────────────────────────────
def quick_eval(tr, te):
    spw = (1 - tr[TARGET].mean()) / tr[TARGET].mean()
    m = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                      tree_method="hist", n_jobs=1, scale_pos_weight=spw,
                      random_state=SEED)
    m.fit(tr[FEATS], tr[TARGET])
    p = m.predict_proba(te[FEATS])[:, 1]
    return dict(auc=roc_auc_score(te[TARGET], p),
                prauc=average_precision_score(te[TARGET], p),
                n_test=int(len(te)),
                pos_rate=float(te[TARGET].mean()))


rowhash = lambda d: pd.util.hash_pandas_object(d[FEATS], index=False)
trA, teA = train_test_split(raw, test_size=0.2, stratify=raw[TARGET],
                            random_state=SEED)
leak_mask = rowhash(teA).isin(set(rowhash(trA)))
dedup_all = raw.drop_duplicates().reset_index(drop=True)
trB, teB = train_test_split(dedup_all, test_size=0.2,
                            stratify=dedup_all[TARGET], random_state=SEED)
R["dup_experiment"] = {
    "A_keep_all_random_split": quick_eval(trA, teA),
    "B_dedup_then_split": quick_eval(trB, teB),
    "C_keep_but_drop_test_twins": quick_eval(trA, teA[~leak_mask]),
    "test_rows_with_twin_in_train": float(leak_mask.mean()),
    "pos_rate_in_duplicated_rows": float(
        raw[raw.duplicated(keep=False)][TARGET].mean()),
    "pos_rate_after_dedup": float(dedup_all[TARGET].mean()),
}
print(json.dumps(R["dup_experiment"], indent=1))

# Decision: de-duplicate BEFORE splitting (conservative, leakage-free)
df = dedup_all.copy()
R["clean_rows"] = int(len(df))
R["clean_pos_rate"] = float(df[TARGET].mean())
R["clean_pos_n"] = int(df[TARGET].sum())
log(f"Clean rows: {len(df):,}  positive rate: {df[TARGET].mean():.4f}")

# ── Outlier inspection ───────────────────────────────────────
R["outliers"] = {}
for c in NUMERIC:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    R["outliers"][c] = dict(q1=float(q1), q3=float(q3), lo=float(lo),
                            hi=float(hi), n_out=n_out,
                            pct=float(n_out / len(df) * 100),
                            min=float(df[c].min()), max=float(df[c].max()),
                            pos_rate_out=float(df.loc[(df[c] < lo) |
                                                      (df[c] > hi), TARGET].mean()))
    print(f" {c:<9} IQR fence [{lo:.1f},{hi:.1f}]  outliers {n_out:,} "
          f"({n_out / len(df):.1%})  range {df[c].min():.0f}-{df[c].max():.0f}")
# Decision: keep all values (they lie in the valid BRFSS range); flag only.

# ── Stratified train / validation / test split (60 / 20 / 20) ─
df_trval, df_te = train_test_split(df, test_size=0.20, stratify=df[TARGET],
                                   random_state=SEED)
df_tr, df_va = train_test_split(df_trval, test_size=0.25,
                                stratify=df_trval[TARGET], random_state=SEED)
R["split"] = {k: int(len(v)) for k, v in
              dict(train=df_tr, validation=df_va, test=df_te).items()}
R["split_pos_rate"] = {k: float(v[TARGET].mean()) for k, v in
                       dict(train=df_tr, validation=df_va, test=df_te).items()}
print(" Split sizes :", R["split"])

# ============================================================
# PHASE 2 — VISUAL EDA
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 2 : VISUAL EDA")
print("=" * 60)
BMI_BINS = [0, 18.5, 25, 30, 35, 40, 200]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese I",
              "Obese II", "Obese III"]
AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
INCOME_LABELS = ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k",
                 "50-75k", ">=75k"]
EDU_LABELS = ["None", "Elem.", "Some HS", "HS grad", "Some coll.", "Coll. grad"]

# ── Plot 1 : class distribution + duplicate experiment ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
cnt = df[TARGET].value_counts().sort_index()
bars = ax.bar(["No diabetes (0)", "Pre-diabetes / diabetes (1)"], cnt.values,
              color=[PALETTE["secondary"], PALETTE["accent"]], width=0.5)
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + cnt.max() * 0.01,
            f"{v:,}\n({v / cnt.sum():.1%})", ha="center", fontsize=9,
            color=PALETTE["muted"])
ax.set_ylim(0, cnt.max() * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
style_axis(ax, "Class Distribution", ylabel="Respondents",
           subtitle="After removing exact duplicate rows")
ax = axes[1]
labs = ["A: keep all\nrandom split", "B: de-dup\nthen split",
        "C: keep all,\ndrop test twins"]
keys = ["A_keep_all_random_split", "B_dedup_then_split",
        "C_keep_but_drop_test_twins"]
x = np.arange(3)
ax.bar(x - 0.18, [R["dup_experiment"][k]["auc"] for k in keys], 0.34,
       color=PALETTE["primary"], label="ROC-AUC")
ax.bar(x + 0.18, [R["dup_experiment"][k]["prauc"] for k in keys], 0.34,
       color=PALETTE["warm"], label="PR-AUC")
for i, k in enumerate(keys):
    ax.text(i - 0.18, R["dup_experiment"][k]["auc"] + 0.01,
            f'{R["dup_experiment"][k]["auc"]:.3f}', ha="center", fontsize=8)
    ax.text(i + 0.18, R["dup_experiment"][k]["prauc"] + 0.01,
            f'{R["dup_experiment"][k]["prauc"]:.3f}', ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=8)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper right")
style_axis(ax, "Duplicate-Removal Experiment",
           subtitle="Same XGBoost model, three ways of handling duplicates")
save_fig(fig, "fig01_class_and_duplicates")

# ── Plot 2 : numeric distributions + outliers ───────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for j, c in enumerate(NUMERIC):
    ax = axes[0, j]
    hi = df[c].quantile(0.995)
    for lab, col, name in [(0, PALETTE["secondary"], "No diabetes"),
                           (1, PALETTE["accent"], "Diabetes")]:
        ax.hist(df.loc[df[TARGET] == lab, c].clip(upper=hi), bins=30,
                alpha=0.55, color=col, density=True, label=name)
    ax.legend(fontsize=8)
    style_axis(ax, f"{c} distribution", ylabel="Density" if j == 0 else "",
               subtitle="Clipped at 99.5th percentile for display")
    ax = axes[1, j]
    ax.boxplot([df.loc[df[TARGET] == 0, c], df.loc[df[TARGET] == 1, c]],
               patch_artist=True,
               boxprops=dict(facecolor="#D6EAF8"),
               medianprops=dict(color=PALETTE["accent"], linewidth=2),
               flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No diabetes", "Diabetes"])
    style_axis(ax, f"{c} by outcome")
save_fig(fig, "fig02_numeric_distributions")

# ── Plot 3 : prevalence by group ─────────────────────────────
tmp = df.copy()
tmp["BMI_cat"] = pd.cut(tmp["BMI"], BMI_BINS, labels=BMI_LABELS)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [("Age", AGE_LABELS, "Age band", 1), ("BMI_cat", BMI_LABELS,
          "BMI category", None), ("Income", INCOME_LABELS, "Income (USD)", 1),
          ("Education", EDU_LABELS, "Education", 1),
          ("Sex", ["Female", "Male"], "Sex", 0),
          ("PhysActivity", ["Inactive", "Active"], "Physical activity", 0)]
prev_tab = {}
for ax, (col, labs, ttl, off) in zip(axes.flatten(), panels):
    g = tmp.groupby(col, observed=True)[TARGET].mean() * 100
    prev_tab[col] = [float(v) for v in g.values]
    ax.bar(range(len(g)), g.values, color=PALETTE["secondary"], width=0.65)
    ax.axhline(df[TARGET].mean() * 100, color=PALETTE["accent"],
               linestyle="--", linewidth=1)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7.5,
                color=PALETTE["muted"])
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labs[:len(g)], rotation=40, ha="right", fontsize=8)
    style_axis(ax, f"Prevalence by {ttl}", ylabel="Diabetes prevalence (%)")
R["prevalence_tables"] = prev_tab
save_fig(fig, "fig03_prevalence_groups")

# ── Plot 4 : correlation heatmap (Spearman; = phi for binary pairs) ─
corr = df.corr(method="spearman")
R["corr_with_target"] = {k: float(v) for k, v in
                         corr[TARGET].drop(TARGET).sort_values(
                             ascending=False).items()}
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 6.5},
            linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.7, "label": "Spearman correlation"}, ax=ax)
ax.set_title("Correlation Heatmap | Spearman (equals phi for binary pairs)",
             fontsize=12, fontweight="bold", color=PALETTE["primary"],
             pad=14, loc="left")
save_fig(fig, "fig04_correlation_heatmap")

# ── Plot 5 : cardiometabolic count and general health ───────
tmp["Cardio_n"] = tmp[["HighBP", "HighChol", "HeartDiseaseorAttack",
                       "Stroke"]].sum(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
g = tmp.groupby("Cardio_n")[TARGET].agg(["mean", "size"])
R["cardio_prevalence"] = {int(k): [float(v["mean"] * 100), int(v["size"])]
                          for k, v in g.iterrows()}
ax = axes[0]
ax.bar(g.index, g["mean"] * 100, color=PALETTE["accent"], width=0.6)
for i, (m, n) in zip(g.index, g.values):
    ax.text(i, m * 100 + 0.8, f"{m * 100:.1f}%\n(n={int(n):,})",
            ha="center", fontsize=8, color=PALETTE["muted"])
ax.set_ylim(0, g["mean"].max() * 100 * 1.25)
style_axis(ax, "Prevalence by Number of Cardiometabolic Conditions",
           xlabel="HighBP + HighChol + Heart disease + Stroke",
           ylabel="Diabetes prevalence (%)")
ax = axes[1]
g2 = tmp.groupby("GenHlth")[TARGET].mean() * 100
R["genhlth_prevalence"] = {int(k): float(v) for k, v in g2.items()}
ax.bar(g2.index, g2.values, color=PALETTE["secondary"], width=0.6)
for i, v in zip(g2.index, g2.values):
    ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8,
            color=PALETTE["muted"])
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1 Excellent", "2 Very good", "3 Good", "4 Fair",
                    "5 Poor"], fontsize=8)
style_axis(ax, "Prevalence by Self-Rated General Health",
           ylabel="Diabetes prevalence (%)")
save_fig(fig, "fig05_cardio_genhlth")
log("EDA figures done")

# ============================================================
# PHASE 3 — STATISTICAL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 3 : STATISTICAL ANALYSIS")
print("=" * 60)
# ── Normality ───────────────────────────────────────────────
rng = np.random.default_rng(SEED)
R["normality"] = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, c in zip(axes, NUMERIC):
    samp = rng.choice(df[c].values, 5000, replace=False)
    w, p = sps.shapiro(samp)
    R["normality"][c] = dict(shapiro_W=float(w), shapiro_p=float(p),
                             skew=float(sps.skew(df[c])),
                             kurtosis=float(sps.kurtosis(df[c])))
    sps.probplot(samp, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_color(PALETTE["secondary"])
    ax.get_lines()[1].set_color(PALETTE["accent"])
    style_axis(ax, f"Q-Q plot | {c}",
               subtitle=f"skew = {R['normality'][c]['skew']:.2f}")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
save_fig(fig, "fig06_qq_plots")

# ── Mann-Whitney + chi-square with effect sizes ─────────────
eff = []
for c in NUMERIC + ORDINAL:
    rb, p = rank_biserial(df.loc[df[TARGET] == 1, c].values,
                          df.loc[df[TARGET] == 0, c].values)
    x1, x0 = df.loc[df[TARGET] == 1, c], df.loc[df[TARGET] == 0, c]
    sp = np.sqrt((x1.var() + x0.var()) / 2)
    d = float((x1.mean() - x0.mean()) / sp)
    eff.append(dict(feature=c, test="Mann-Whitney U", effect=rb,
                    effect_name="rank-biserial", cohen_d=d, p=p,
                    mean_pos=float(x1.mean()), mean_neg=float(x0.mean())))
for c in BINARY:
    v, p = cramers_v(df[c], df[TARGET])
    eff.append(dict(feature=c, test="Chi-square", effect=v,
                    effect_name="Cramer's V", cohen_d=np.nan, p=p,
                    mean_pos=float(df.loc[df[TARGET] == 1, c].mean()),
                    mean_neg=float(df.loc[df[TARGET] == 0, c].mean())))
eff_df = pd.DataFrame(eff)
eff_df["abs_effect"] = eff_df["effect"].abs()
eff_df = eff_df.sort_values("abs_effect", ascending=False).reset_index(drop=True)
R["effect_sizes"] = eff_df.drop(columns="abs_effect").to_dict("records")
R["n_significant_p05"] = int((eff_df["p"] < 0.05).sum())
print(eff_df[["feature", "test", "effect", "p"]].round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 7.5))
e = eff_df.iloc[::-1]
cols = [PALETTE["accent"] if t == "Mann-Whitney U" else PALETTE["secondary"]
        for t in e["test"]]
ax.barh(e["feature"], e["effect"], color=cols, height=0.62)
ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
for i, v in enumerate(e["effect"]):
    ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7.5,
            color=PALETTE["muted"])
style_axis(ax, "Effect Sizes vs Diabetes Status",
           subtitle="Red = rank-biserial (numeric / ordinal) | "
                    "Blue = Cramer's V (binary)",
           xlabel="Effect size")
save_fig(fig, "fig07_effect_sizes")

# ── Logistic-regression odds ratios (training split) ────────
Xtr0 = df_tr[FEATS].astype(float)
ytr0 = df_tr[TARGET].values
logit = sm.Logit(ytr0, sm.add_constant(Xtr0)).fit(disp=0, maxiter=200)
ci = logit.conf_int()
or_df = pd.DataFrame({"feature": logit.params.index,
                      "OR": np.exp(logit.params.values),
                      "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values),
                      "p": logit.pvalues.values}).query("feature != 'const'")
sc0 = StandardScaler().fit(Xtr0)
logit_s = sm.Logit(ytr0, sm.add_constant(pd.DataFrame(
    sc0.transform(Xtr0), columns=FEATS))).fit(disp=0, maxiter=200)
ci_s = logit_s.conf_int()
or_s = pd.DataFrame({"feature": logit_s.params.index,
                     "OR_per_SD": np.exp(logit_s.params.values),
                     "lo_sd": np.exp(ci_s[0].values),
                     "hi_sd": np.exp(ci_s[1].values),
                     "coef": logit_s.params.values}).query("feature != 'const'")
or_df = or_df.merge(or_s, on="feature").sort_values("OR_per_SD",
                                                   ascending=False)
R["odds_ratios"] = or_df.to_dict("records")
top5_or = or_df.reindex(or_df["coef"].abs().sort_values(
    ascending=False).index).head(5)["feature"].tolist()
R["top5_by_odds_ratio"] = top5_or
print(or_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
o = or_df.sort_values("OR").reset_index(drop=True)
ax.errorbar(o["OR"], range(len(o)),
            xerr=[o["OR"] - o["lo"], o["hi"] - o["OR"]], fmt="o",
            color=PALETTE["primary"], ecolor=PALETTE["secondary"],
            capsize=2.5, markersize=5)
ax.axvline(1, color=PALETTE["accent"], linestyle="--", linewidth=1)
ax.set_yticks(range(len(o)))
ax.set_yticklabels(o["feature"], fontsize=8.5)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
style_axis(ax, "Odds Ratios with 95% Confidence Intervals",
           subtitle="Multivariable logistic regression on the training "
                    "split (per one-unit change; log scale)",
           xlabel="Odds ratio")
save_fig(fig, "fig08_odds_ratios")

# ── Variance Inflation Factor ───────────────────────────────
vs = sm.add_constant(df_tr[FEATS].astype(float).sample(
    min(30000, len(df_tr)), random_state=SEED))
vif = pd.DataFrame({"feature": FEATS,
                    "VIF": [variance_inflation_factor(vs.values, i + 1)
                            for i in range(len(FEATS))]
                    }).sort_values("VIF", ascending=False)
R["vif"] = vif.to_dict("records")
print(vif.round(2).to_string())
log("statistics done")

# ============================================================
# PHASE 4 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 4 : FEATURE ENGINEERING")
print("=" * 60)


def add_features(d):
    """Domain features; every one is tested before it is kept."""
    d = d.copy()
    d["BMI_cat"] = pd.cut(d["BMI"], BMI_BINS, labels=False).astype(int)
    d["Age_grp"] = np.select([d["Age"] <= 5, d["Age"] <= 9], [0, 1], 2)
    d["Cardio_score"] = (d["HighBP"] + d["HighChol"] +
                         d["HeartDiseaseorAttack"] + d["Stroke"])
    d["BMI_x_Age"] = d["BMI"] * d["Age"]
    return d


NEW_FEATS = ["BMI_cat", "Age_grp", "Cardio_score", "BMI_x_Age"]
trF = add_features(df_tr)
sub = trF.sample(min(60000, len(trF)), random_state=SEED)
skf3 = StratifiedKFold(3, shuffle=True, random_state=SEED)


def xgb_base(spw=1.0, **kw):
    params = dict(n_estimators=150, max_depth=4, learning_rate=0.1,
                  tree_method="hist", n_jobs=1, scale_pos_weight=spw,
                  random_state=SEED)
    params.update(kw)
    return XGBClassifier(**params)


def fe_eval(cols):
    out = {}
    for name, mk in [("XGB", lambda: xgb_base(6.0)),
                     ("LR", lambda: Pipeline([("s", StandardScaler()),
                                              ("m", LogisticRegression(
                                                  max_iter=500))]))]:
        aps = []
        for a, b in skf3.split(sub, sub[TARGET]):
            m = mk().fit(sub.iloc[a][cols], sub.iloc[a][TARGET])
            aps.append(average_precision_score(
                sub.iloc[b][TARGET], m.predict_proba(sub.iloc[b][cols])[:, 1]))
        out[name] = float(np.mean(aps))
    return out


fe_res = {"baseline": fe_eval(FEATS)}
for f in NEW_FEATS:
    fe_res["+" + f] = fe_eval(FEATS + [f])
fe_res["+all four"] = fe_eval(FEATS + NEW_FEATS)
for k, v in fe_res.items():
    v["delta_XGB"] = v["XGB"] - fe_res["baseline"]["XGB"]
    v["delta_LR"] = v["LR"] - fe_res["baseline"]["LR"]
    print(f" {k:<14} XGB {v['XGB']:.4f}  LR {v['LR']:.4f}")
R["feature_engineering"] = fe_res
KEEP = [f for f in NEW_FEATS
        if (fe_res["+" + f]["delta_XGB"] + fe_res["+" + f]["delta_LR"]) / 2
        > 0.0003]
R["kept_features"] = KEEP
FEATS_FINAL = FEATS + KEEP
print(" Kept engineered features :", KEEP)

fig, ax = plt.subplots(figsize=(11, 5))
names = [k for k in fe_res if k != "baseline"]
x = np.arange(len(names))
ax.bar(x - 0.18, [fe_res[k]["delta_XGB"] * 1000 for k in names], 0.34,
       color=PALETTE["secondary"], label="XGBoost")
ax.bar(x + 0.18, [fe_res[k]["delta_LR"] * 1000 for k in names], 0.34,
       color=PALETTE["warm"], label="Logistic regression")
ax.axhline(0, color=PALETTE["dark"], linewidth=0.8)
ax.axhline(0.3, color=PALETTE["accent"], linestyle="--", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.legend(fontsize=8.5)
style_axis(ax, "Change in Cross-Validated PR-AUC from Each Engineered Feature",
           subtitle="Units: PR-AUC x 1000 vs baseline | dashed line = "
                    "keep threshold",
           ylabel="Delta PR-AUC (x 1000)")
save_fig(fig, "fig09_feature_engineering")
log("feature engineering done")

# ── Final design matrices ───────────────────────────────────
Xtr = add_features(df_tr)[FEATS_FINAL]
Xva = add_features(df_va)[FEATS_FINAL]
Xte = add_features(df_te)[FEATS_FINAL]
ytr, yva, yte = (df_tr[TARGET].values, df_va[TARGET].values,
                 df_te[TARGET].values)
SPW = float((ytr == 0).sum() / (ytr == 1).sum())
R["scale_pos_weight"] = SPW

# ============================================================
# PHASE 5 — CLASS-IMBALANCE HANDLING
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 5 : CLASS-IMBALANCE HANDLING (inside CV folds)")
print("=" * 60)
subI = Xtr.assign(y=ytr).sample(min(60000, len(Xtr)), random_state=SEED)
Xs, ys = subI[FEATS_FINAL], subI["y"].values
methods = {
    "No handling": ImbPipeline([("m", xgb_base(1.0))]),
    "Class weights": ImbPipeline([("m", xgb_base(SPW))]),
    "SMOTE": ImbPipeline([("s", SMOTE(random_state=SEED)),
                          ("m", xgb_base(1.0))]),
    "Random undersampling": ImbPipeline([("s", RandomUnderSampler(
        random_state=SEED)), ("m", xgb_base(1.0))]),
}
imb = {}
for name, pipe in methods.items():
    rows = []
    for a, b in skf3.split(Xs, ys):
        pipe.fit(Xs.iloc[a], ys[a])
        p = pipe.predict_proba(Xs.iloc[b])[:, 1]
        m5 = metrics_at(ys[b], p, 0.5)
        rows.append([average_precision_score(ys[b], p),
                     roc_auc_score(ys[b], p),
                     recall_at_precision(ys[b], p, 0.30), m5["recall"],
                     m5["precision"]])
    mu = np.mean(rows, axis=0)
    imb[name] = dict(prauc=float(mu[0]), auc=float(mu[1]),
                     recall_at_p30=float(mu[2]), recall_05=float(mu[3]),
                     precision_05=float(mu[4]))
    print(f" {name:<22} PR-AUC {mu[0]:.4f}  AUC {mu[1]:.4f}  "
          f"R@P30 {mu[2]:.3f}  recall@0.5 {mu[3]:.3f}")
R["imbalance"] = imb

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
nm = list(imb.keys())
for ax, (key, ttl, col) in zip(axes, [
        ("prauc", "PR-AUC (ranking quality)", PALETTE["primary"]),
        ("recall_at_p30", "Recall at precision >= 0.30", PALETTE["teal"]),
        ("recall_05", "Recall at default 0.5 threshold", PALETTE["accent"])]):
    vals = [imb[k][key] for k in nm]
    ax.bar(range(4), vals, color=col, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5,
                color=PALETTE["muted"])
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace(" ", "\n") for n in nm], fontsize=8)
    ax.set_ylim(0, max(vals) * 1.18)
    style_axis(ax, ttl)
fig.suptitle("Imbalance Handling Compared | resampling applied only "
             "inside training folds", fontsize=12, fontweight="bold",
             color=PALETTE["primary"], y=1.03)
save_fig(fig, "fig10_imbalance")
log("imbalance done")

# ============================================================
# PHASE 6 — MODEL BUILDING, TUNING, CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 6 : MODELS, TUNING AND 5-FOLD CROSS-VALIDATION")
print("=" * 60)
skf5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
tune_n = 20000 if QUICK else 60000
tsub = Xtr.assign(y=ytr).sample(min(tune_n, len(Xtr)), random_state=SEED)
Xt, yt = tsub[FEATS_FINAL], tsub["y"].values

# -- Logistic regression: grid over C
gs_lr = GridSearchCV(
    Pipeline([("s", StandardScaler()),
              ("m", LogisticRegression(class_weight="balanced",
                                       max_iter=1000))]),
    {"m__C": [0.01, 0.1, 1, 10]}, scoring="average_precision", cv=skf3,
    n_jobs=1).fit(Xt, yt)
# -- Decision tree: grid over depth / leaf size
gs_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    {"max_depth": [4, 5, 6, 8], "min_samples_leaf": [50, 200, 500]},
    scoring="average_precision", cv=skf3, n_jobs=1).fit(Xt, yt)
# -- Random forest: randomised search
rs_rf = RandomizedSearchCV(
    RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                           n_jobs=1, random_state=SEED),
    {"max_depth": [10, 14, 18], "min_samples_leaf": [10, 30, 60],
     "max_features": ["sqrt", 0.5]},
    n_iter=3 if QUICK else 5, scoring="average_precision", cv=skf3,
    random_state=SEED, n_jobs=1).fit(Xt, yt)
# -- XGBoost: randomised search
rs_xgb = RandomizedSearchCV(
    xgb_base(SPW),
    {"n_estimators": [150, 250, 400], "max_depth": [3, 4, 5, 6],
     "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
     "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10],
     "reg_lambda": [1, 5, 10]},
    n_iter=4 if QUICK else 10, scoring="average_precision", cv=skf3,
    random_state=SEED, n_jobs=1).fit(Xt, yt)
R["best_params"] = {
    "Logistic Regression": {k: v for k, v in gs_lr.best_params_.items()},
    "Decision Tree": gs_dt.best_params_,
    "Random Forest": rs_rf.best_params_,
    "XGBoost": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                for k, v in rs_xgb.best_params_.items()},
}
print(json.dumps(R["best_params"], indent=1, default=str))
log("tuning done")

MODELS = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": gs_lr.best_estimator_,
    "Decision Tree": gs_dt.best_estimator_,
    "Random Forest": rs_rf.best_estimator_,
    "XGBoost": rs_xgb.best_estimator_,
    "MLP (neural net)": ImbPipeline([
        ("ros", RandomOverSampler(sampling_strategy=0.5, random_state=SEED)),
        ("s", StandardScaler()),
        ("m", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                            batch_size=512, early_stopping=True,
                            n_iter_no_change=4, max_iter=40,
                            random_state=SEED))]),
}
SCORING = {"auc": "roc_auc", "ap": "average_precision", "recall": "recall",
           "f1": "f1"}
cv_res, fold_ap = {}, {}
for name, mdl in MODELS.items():
    cvr = cross_validate(mdl, Xtr, ytr, cv=skf5, scoring=SCORING, n_jobs=1)
    cv_res[name] = {k: [float(np.mean(cvr["test_" + k])),
                        float(np.std(cvr["test_" + k]))] for k in SCORING}
    fold_ap[name] = [float(v) for v in cvr["test_ap"]]
    print(f" {name:<22} AUC {cv_res[name]['auc'][0]:.4f}+-"
          f"{cv_res[name]['auc'][1]:.4f}  PR-AUC {cv_res[name]['ap'][0]:.4f}"
          f"  recall {cv_res[name]['recall'][0]:.3f}")
    log(f"CV done: {name}")
R["cv"] = cv_res
R["cv_fold_ap"] = fold_ap

# ── Fit on full training split, score validation and test ───
val_p, te_p = {}, {}
for name, mdl in MODELS.items():
    mdl.fit(Xtr, ytr)
    val_p[name] = mdl.predict_proba(Xva)[:, 1]
    te_p[name] = mdl.predict_proba(Xte)[:, 1]

test_tab = {}
for name in MODELS:
    m5 = metrics_at(yte, te_p[name], 0.5)
    test_tab[name] = dict(
        accuracy=m5["accuracy"], precision=m5["precision"],
        recall=m5["recall"], f1=m5["f1"],
        auc=float(roc_auc_score(yte, te_p[name])),
        prauc=float(average_precision_score(yte, te_p[name])),
        recall_at_p30=recall_at_precision(yte, te_p[name], 0.30))
R["test_models"] = test_tab
for k, v in test_tab.items():
    print(f" TEST {k:<22} acc {v['accuracy']:.3f}  rec {v['recall']:.3f}  "
          f"AUC {v['auc']:.4f}  PR-AUC {v['prauc']:.4f}")

CANDIDATES = [k for k in MODELS if k != "Majority baseline"]
BEST = max(CANDIDATES, key=lambda k: cv_res[k]["ap"][0])
SECOND = sorted(CANDIDATES, key=lambda k: -cv_res[k]["ap"][0])[1]
R["best_model"], R["second_model"] = BEST, SECOND
print(f" Best by CV PR-AUC: {BEST}  | runner-up: {SECOND}")

# ── Plot 11 : CV model comparison ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cols_m = [PALETTE["muted"], PALETTE["secondary"], PALETTE["teal"],
          PALETTE["warm"], PALETTE["accent"], PALETTE["purple"]]
for ax, (key, ttl) in zip(axes, [("auc", "ROC-AUC"), ("ap", "PR-AUC"),
                                 ("recall", "Recall at 0.5 threshold")]):
    mu = [cv_res[k][key][0] for k in MODELS]
    sd = [cv_res[k][key][1] for k in MODELS]
    ax.bar(range(len(MODELS)), mu, yerr=sd, color=cols_m, width=0.62,
           capsize=3, edgecolor="white")
    for i, v in enumerate(mu):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8,
                color=PALETTE["muted"])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([k.replace(" (", "\n(").replace(" ", "\n", 1)
                        for k in MODELS], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    style_axis(ax, ttl, subtitle="Mean +- SD over 5 stratified folds")
save_fig(fig, "fig11_cv_comparison")

# ── Plot 12 : ROC and PR curves (test) ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for (name, col) in zip(CANDIDATES, cols_m[1:]):
    fpr_, tpr_, _ = roc_curve(yte, te_p[name])
    pr_, rc_, _ = precision_recall_curve(yte, te_p[name])
    axes[0].plot(fpr_, tpr_, color=col, linewidth=2,
                 label=f"{name} (AUC {test_tab[name]['auc']:.3f})")
    axes[1].plot(rc_, pr_, color=col, linewidth=2,
                 label=f"{name} (AP {test_tab[name]['prauc']:.3f})")
axes[0].plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":")
axes[1].axhline(yte.mean(), color=PALETTE["muted"], linestyle=":",
                label=f"Baseline = prevalence ({yte.mean():.3f})")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].legend(fontsize=8, loc="upper right")
style_axis(axes[0], "ROC Curves | Test Set",
           xlabel="False positive rate", ylabel="True positive rate")
style_axis(axes[1], "Precision-Recall Curves | Test Set",
           xlabel="Recall", ylabel="Precision")
save_fig(fig, "fig12_roc_pr_curves")
log("model comparison done")

# ============================================================
# PHASE 7 — STATISTICAL COMPARISON OF THE TOP TWO MODELS
# ============================================================
print("\n" + "=" * 60)
print(f" PHASE 7 : {BEST} vs {SECOND}")
print("=" * 60)
t_stat, t_p = sps.ttest_rel(fold_ap[BEST], fold_ap[SECOND])
rng = np.random.default_rng(SEED)
diffs_auc, diffs_ap = [], []
pa, pb = te_p[BEST], te_p[SECOND]
for _ in range(300):
    idx = rng.integers(0, len(yte), len(yte))
    diffs_auc.append(roc_auc_score(yte[idx], pa[idx]) -
                     roc_auc_score(yte[idx], pb[idx]))
    diffs_ap.append(average_precision_score(yte[idx], pa[idx]) -
                    average_precision_score(yte[idx], pb[idx]))
R["top2_comparison"] = dict(
    paired_t_stat=float(t_stat), paired_t_p=float(t_p),
    auc_diff=float(np.mean(diffs_auc)),
    auc_diff_ci=[float(v) for v in np.percentile(diffs_auc, [2.5, 97.5])],
    ap_diff=float(np.mean(diffs_ap)),
    ap_diff_ci=[float(v) for v in np.percentile(diffs_ap, [2.5, 97.5])])
R["best_auc_ci"] = boot_ci(yte, pa, roc_auc_score)
R["best_prauc_ci"] = boot_ci(yte, pa, average_precision_score)
print(json.dumps(R["top2_comparison"], indent=1))

# ============================================================
# PHASE 8 — CALIBRATION
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 8 : PROBABILITY CALIBRATION (fitted on validation set)")
print("=" * 60)
best_mdl = MODELS[BEST]
p_va_raw, p_te_raw = val_p[BEST], te_p[BEST]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                      (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def fit_calibrator(kind, p, y):
    if kind == "platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(
            lg(p).reshape(-1, 1), y)
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def apply_calibration(kind, cal, p):
    p = np.asarray(p, dtype=float)
    if kind == "platt":
        return cal.predict_proba(lg(p).reshape(-1, 1))[:, 1]
    return cal.predict(p)


# choose calibrator by honest split of the validation set
perm = np.random.default_rng(SEED).permutation(len(yva))
ha, hb = perm[:len(perm) // 2], perm[len(perm) // 2:]
sel = {}
for kind in ["platt", "isotonic"]:
    cal = fit_calibrator(kind, p_va_raw[ha], yva[ha])
    sel[kind] = float(brier_score_loss(yva[hb], apply_calibration(
        kind, cal, p_va_raw[hb])))
CAL_KIND = min(sel, key=sel.get)
cal_final = fit_calibrator(CAL_KIND, p_va_raw, yva)
p_va = apply_calibration(CAL_KIND, cal_final, p_va_raw)
p_te = apply_calibration(CAL_KIND, cal_final, p_te_raw)
p_te_platt = apply_calibration("platt", fit_calibrator(
    "platt", p_va_raw, yva), p_te_raw)
p_te_iso = apply_calibration("isotonic", fit_calibrator(
    "isotonic", p_va_raw, yva), p_te_raw)
R["calibration"] = dict(
    selection_brier_on_val_half=sel, chosen=CAL_KIND,
    brier_raw=float(brier_score_loss(yte, p_te_raw)),
    brier_platt=float(brier_score_loss(yte, p_te_platt)),
    brier_isotonic=float(brier_score_loss(yte, p_te_iso)),
    brier_prevalence_only=float(brier_score_loss(
        yte, np.full(len(yte), ytr.mean()))),
    ece_raw=ece_score(yte, p_te_raw), ece_platt=ece_score(yte, p_te_platt),
    ece_isotonic=ece_score(yte, p_te_iso),
    mean_pred_raw=float(p_te_raw.mean()), mean_pred_cal=float(p_te.mean()),
    observed_rate=float(yte.mean()),
    auc_after=float(roc_auc_score(yte, p_te)))
print(json.dumps(R["calibration"], indent=1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":",
        label="Perfect calibration")
for lab, p, col in [("Raw (class-weighted)", p_te_raw, PALETTE["accent"]),
                    ("Platt scaling", p_te_platt, PALETTE["warm"]),
                    ("Isotonic regression", p_te_iso, PALETTE["teal"])]:
    fy, fx = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    ax.plot(fx, fy, marker="o", linewidth=2, color=col, label=lab)
ax.legend(fontsize=8.5)
style_axis(ax, "Reliability Curve | Test Set",
           subtitle=f"Best model: {BEST}",
           xlabel="Mean predicted probability",
           ylabel="Observed fraction of diabetics")
ax = axes[1]
labs = ["Raw", "Platt", "Isotonic"]
bs = [R["calibration"]["brier_raw"], R["calibration"]["brier_platt"],
      R["calibration"]["brier_isotonic"]]
ax.bar(labs, bs, color=[PALETTE["accent"], PALETTE["warm"],
                        PALETTE["teal"]], width=0.5)
for i, v in enumerate(bs):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9,
            color=PALETTE["muted"])
ax.axhline(R["calibration"]["brier_prevalence_only"], color=PALETTE["muted"],
           linestyle="--", linewidth=1)
ax.text(2.45, R["calibration"]["brier_prevalence_only"] + 0.002,
        "prevalence-only", ha="right", fontsize=8, color=PALETTE["muted"])
style_axis(ax, "Brier Score (lower = better)", ylabel="Brier score")
save_fig(fig, "fig13_calibration")
log("calibration done")

# ============================================================
# PHASE 9 — COST-SENSITIVE THRESHOLD
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 9 : COST-SENSITIVE THRESHOLD")
print("=" * 60)
grid = np.linspace(0.01, 0.80, 160)
RATIOS = [5, 10, 20]
costs = {r: [] for r in RATIOS}
rec_curve, prec_curve = [], []
for t in grid:
    m = metrics_at(yva, p_va, t)
    rec_curve.append(m["recall"])
    prec_curve.append(m["precision"])
    for r in RATIOS:
        costs[r].append((r * m["fn"] + m["fp"]) / len(yva) * 1000)
t_cost = {r: float(grid[int(np.argmin(costs[r]))]) for r in RATIOS}
ok = np.where(np.array(rec_curve) >= 0.80)[0]
t_rec80 = float(grid[ok.max()]) if len(ok) else float(grid[0])
rec_at_tcost10 = metrics_at(yva, p_va, t_cost[10])["recall"]
T_FINAL = t_cost[10] if rec_at_tcost10 >= 0.80 else t_rec80
R["threshold"] = dict(
    t_cost={str(r): t_cost[r] for r in RATIOS}, t_recall80=t_rec80,
    final=float(T_FINAL), theoretical_t10=1 / 11,
    val_metrics={str(r): metrics_at(yva, p_va, t_cost[r]) for r in RATIOS},
    val_at_recall80=metrics_at(yva, p_va, t_rec80),
    val_final=metrics_at(yva, p_va, T_FINAL),
    cost_min={str(r): float(min(costs[r])) for r in RATIOS},
    cost_at_default={str(r): float((r * metrics_at(yva, p_va, 0.5)["fn"] +
                                    metrics_at(yva, p_va, 0.5)["fp"]) /
                                   len(yva) * 1000) for r in RATIOS})
R["test_final"] = metrics_at(yte, p_te, T_FINAL)
R["test_default05_cal"] = metrics_at(yte, p_te, 0.5)
R["test_final_ci"] = dict(
    recall=boot_ci(yte, p_te, lambda y, p: metrics_at(y, p, T_FINAL)["recall"]),
    precision=boot_ci(yte, p_te, lambda y, p: metrics_at(y, p, T_FINAL)["precision"]))
print(json.dumps({k: R["threshold"][k] for k in
                  ["t_cost", "t_recall80", "final"]}, indent=1))
print(" Test @ final threshold:", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in R["test_final"].items()})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
ax = axes[0]
for r, col in zip(RATIOS, [PALETTE["secondary"], PALETTE["accent"],
                           PALETTE["purple"]]):
    ax.plot(grid, costs[r], color=col, linewidth=2,
            label=f"FN = {r} x FP")
    ax.scatter([t_cost[r]], [min(costs[r])], color=col, s=70, zorder=5,
               edgecolor="white")
ax.axvline(T_FINAL, color=PALETTE["dark"], linestyle="--", linewidth=1,
           label=f"Chosen threshold ({T_FINAL:.2f})")
ax.legend(fontsize=8.5)
style_axis(ax, "Total Cost vs Decision Threshold",
           subtitle="Validation set | cost per 1,000 people screened "
                    "(1 unit = one false alarm)",
           xlabel="Threshold on calibrated probability",
           ylabel="Cost per 1,000 people")
ax = axes[1]
ax.plot(grid, rec_curve, color=PALETTE["accent"], linewidth=2, label="Recall")
ax.plot(grid, prec_curve, color=PALETTE["primary"], linewidth=2,
        label="Precision")
ax.axhline(0.80, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.axvline(T_FINAL, color=PALETTE["dark"], linestyle="--", linewidth=1)
ax.axvline(t_rec80, color=PALETTE["teal"], linestyle="-.", linewidth=1,
           label=f"Recall = 0.80 at {t_rec80:.2f}")
ax.legend(fontsize=8.5)
style_axis(ax, "Recall and Precision vs Threshold",
           subtitle="Validation set", xlabel="Threshold",
           ylabel="Score")
save_fig(fig, "fig14_cost_threshold")

# ── Confusion matrices on the test set ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (ttl, thr) in zip(axes, [("Default threshold 0.50", 0.5),
                                 (f"Chosen threshold {T_FINAL:.2f}",
                                  T_FINAL)]):
    m = metrics_at(yte, p_te, thr)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                linewidths=1, linecolor="white", ax=ax,
                xticklabels=["Predicted\nno diabetes", "Predicted\ndiabetes"],
                yticklabels=["Actual\nno diabetes", "Actual\ndiabetes"],
                annot_kws={"size": 12})
    ax.set_title(f"{ttl}\nrecall {m['recall']:.3f} | precision "
                 f"{m['precision']:.3f}", fontsize=11, fontweight="bold",
                 color=PALETTE["primary"], loc="left")
save_fig(fig, "fig15_confusion_matrices")
log("threshold done")

# ============================================================
# PHASE 10 — FAIRNESS AUDIT AND MITIGATION
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 10 : FAIRNESS AUDIT (sex, age group, income group)")
print("=" * 60)


def group_labels(d):
    return {
        "Sex": np.where(d["Sex"] == 1, "Male", "Female"),
        "Age group": np.select([d["Age"] <= 5, d["Age"] <= 9],
                               ["18-44", "45-64"], "65+"),
        "Income group": np.select([d["Income"] <= 4, d["Income"] <= 6],
                                  ["Low (<25k)", "Middle (25-50k)"],
                                  "High (>=50k)"),
    }


G_va, G_te = group_labels(df_va), group_labels(df_te)
target_recall = metrics_at(yva, p_va, T_FINAL)["recall"]
fair = {}
for attr in G_te:
    rows_b, rows_a, thr_g = [], [], {}
    for g in sorted(set(G_te[attr])):
        mv, mt = G_va[attr] == g, G_te[attr] == g
        pos = p_va[mv & (yva == 1)]
        thr_g[g] = float(np.quantile(pos, 1 - target_recall))
        for rows, thr in [(rows_b, T_FINAL), (rows_a, thr_g[g])]:
            m = metrics_at(yte[mt], p_te[mt], thr)
            rows.append(dict(group=g, n=int(mt.sum()),
                             prevalence=float(yte[mt].mean()), thr=float(thr),
                             recall=m["recall"], precision=m["precision"],
                             fnr=m["fnr"], fpr=m["fpr"],
                             flag_rate=m["flagged"] / mt.sum()))
    # overall effect of group-specific thresholds
    flag = np.zeros(len(yte), dtype=bool)
    for g in thr_g:
        mt = G_te[attr] == g
        flag[mt] = p_te[mt] >= thr_g[g]
    tp = int((flag & (yte == 1)).sum())
    ov = dict(recall=tp / yte.sum(), precision=tp / flag.sum(),
              flagged=int(flag.sum()), fpr=float((flag & (yte == 0)).sum() /
                                                 (yte == 0).sum()))
    b, a = pd.DataFrame(rows_b), pd.DataFrame(rows_a)
    fair[attr] = dict(
        before=rows_b, after=rows_a, thresholds=thr_g, overall_after=ov,
        recall_gap_before=float(b["recall"].max() - b["recall"].min()),
        recall_gap_after=float(a["recall"].max() - a["recall"].min()),
        fpr_gap_before=float(b["fpr"].max() - b["fpr"].min()),
        fpr_gap_after=float(a["fpr"].max() - a["fpr"].min()))
    print(f"\n {attr}: recall gap {fair[attr]['recall_gap_before']:.3f} -> "
          f"{fair[attr]['recall_gap_after']:.3f} | FPR gap "
          f"{fair[attr]['fpr_gap_before']:.3f} -> "
          f"{fair[attr]['fpr_gap_after']:.3f}")
    print(b[["group", "n", "recall", "precision", "fpr"]].round(3).to_string())
R["fairness"] = fair
R["fairness_target_recall"] = float(target_recall)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, attr in zip(axes, fair):
    b, a = pd.DataFrame(fair[attr]["before"]), pd.DataFrame(fair[attr]["after"])
    x = np.arange(len(b))
    w = 0.2
    ax.bar(x - 1.5 * w, b["recall"], w, color=PALETTE["accent"],
           label="Recall (single thr.)")
    ax.bar(x - 0.5 * w, a["recall"], w, color="#F1948A",
           label="Recall (group thr.)")
    ax.bar(x + 0.5 * w, b["fpr"], w, color=PALETTE["primary"],
           label="FPR (single thr.)")
    ax.bar(x + 1.5 * w, a["fpr"], w, color=PALETTE["secondary"],
           label="FPR (group thr.)")
    ax.set_xticks(x)
    ax.set_xticklabels(b["group"], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    style_axis(ax, f"By {attr}", subtitle="Test set")
fig.suptitle("Fairness Audit | recall and false-positive rate by subgroup",
             fontsize=12, fontweight="bold", color=PALETTE["primary"], y=1.03)
save_fig(fig, "fig16_fairness")
log("fairness done")

# ============================================================
# PHASE 11 — SHAP EXPLAINABILITY
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 11 : SHAP EXPLAINABILITY")
print("=" * 60)
xgb_model = MODELS["XGBoost"]
shap_n = 800 if QUICK else 2000
Xs_shap = Xte.sample(shap_n, random_state=SEED)
explainer = shap.TreeExplainer(xgb_model)
sv = explainer(Xs_shap)
mean_abs = pd.Series(np.abs(sv.values).mean(axis=0),
                     index=FEATS_FINAL).sort_values(ascending=False)
R["shap_mean_abs"] = {k: float(v) for k, v in mean_abs.items()}
R["top5_by_shap"] = mean_abs.head(5).index.tolist()
print(mean_abs.head(10).round(4).to_string())

plt.figure(figsize=(10, 7.5))
shap.summary_plot(sv.values, Xs_shap, show=False, max_display=15)
plt.title("SHAP Summary | Effect of each feature on predicted risk",
          fontsize=12, fontweight="bold", color=PALETTE["primary"], loc="left")
save_fig(plt.gcf(), "fig17_shap_summary")

fig, ax = plt.subplots(figsize=(10, 7))
top = mean_abs.head(15).iloc[::-1]
ax.barh(top.index, top.values, color=PALETTE["secondary"], height=0.62)
for i, v in enumerate(top.values):
    ax.text(v + top.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8,
            color=PALETTE["muted"])
style_axis(ax, "SHAP Feature Importance | Mean |SHAP value|",
           subtitle=f"{shap_n:,} test respondents, XGBoost log-odds scale",
           xlabel="Mean absolute SHAP value")
save_fig(fig, "fig18_shap_bar")

# three individual explanations: highest risk, near threshold, lowest risk
pr_s = xgb_model.predict_proba(Xs_shap)[:, 1]
cal_s = apply_calibration(CAL_KIND, cal_final, pr_s)
picks = {"high": int(np.argmax(cal_s)),
         "borderline": int(np.argmin(np.abs(cal_s - T_FINAL))),
         "low": int(np.argmin(cal_s))}
R["shap_cases"] = {}
for tag, i in picks.items():
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    R["shap_cases"][tag] = dict(
        calibrated_risk=float(cal_s[i]),
        actual=int(df_te.loc[Xs_shap.index[i], TARGET]),
        top_drivers=[FEATS_FINAL[j] for j in
                     np.argsort(-np.abs(sv.values[i]))[:3]],
        profile={c: float(Xs_shap.iloc[i][c]) for c in FEATS_FINAL})
    save_fig(plt.gcf(), f"fig19_shap_waterfall_{tag}")
log("SHAP done")

# ============================================================
# PHASE 12 — SCREENING SIMULATION (10,000 people)
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 12 : SCREENING SIMULATION")
print("=" * 60)
N_SIM = 10000
prev = R["raw_pos_rate"]          # survey prevalence (before de-duplication)
tf = R["test_final"]
diab = N_SIM * prev
healthy = N_SIM - diab
caught = diab * tf["recall"]
missed = diab - caught
false_alarms = healthy * tf["fpr"]
flagged = caught + false_alarms
R["simulation"] = dict(
    n=N_SIM, prevalence=prev, diabetics=diab, caught=caught, missed=missed,
    false_alarms=false_alarms, flagged=flagged,
    tests_saved_pct=float((1 - flagged / N_SIM) * 100),
    nns_model=float(flagged / caught), nns_random=float(1 / prev),
    ppv=float(caught / flagged),
    random_same_budget_caught=float(flagged * prev),
    lift=float((caught / flagged) / prev))
print(json.dumps(R["simulation"], indent=1))

S = R["simulation"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
parts = [("Caught\n(true +)", caught, PALETTE["teal"]),
         ("Missed\n(false -)", missed, PALETTE["accent"]),
         ("False alarms\n(false +)", false_alarms, PALETTE["warm"]),
         ("Correctly\nreassured", healthy - false_alarms,
          PALETTE["secondary"])]
ax.bar(range(4), [p[1] for p in parts], color=[p[2] for p in parts],
       width=0.6)
for i, p in enumerate(parts):
    ax.text(i, p[1] + 120, f"{p[1]:,.0f}", ha="center", fontsize=9.5,
            color=PALETTE["muted"])
ax.set_xticks(range(4))
ax.set_xticklabels([p[0] for p in parts], fontsize=8.5)
ax.set_ylim(0, healthy * 1.12)
style_axis(ax, "Outcome for 10,000 Screened People",
           subtitle=f"Threshold {T_FINAL:.2f} | prevalence {prev:.1%} | "
                    "test-set recall and FPR", ylabel="People")
ax = axes[1]
ax.bar(["Test everyone\n(no model)", "Test only people\nflagged by model"],
       [1 / prev, S["nns_model"]], color=[PALETTE["muted"],
                                          PALETTE["primary"]], width=0.5)
for i, v in enumerate([1 / prev, S["nns_model"]]):
    ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10,
            color=PALETTE["muted"])
style_axis(ax, "Lab Tests Needed per Case Found",
           subtitle="Lower is better", ylabel="Tests per diabetic found")
save_fig(fig, "fig20_screening_simulation")

# ============================================================
# PHASE 13 — SAVE MODEL AND DEMO APP
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 13 : SAVE MODEL + DEMO APP")
print("=" * 60)
bundle = dict(model=xgb_model, calibrator=cal_final, kind=CAL_KIND,
              threshold=float(T_FINAL), features=FEATS_FINAL)
joblib.dump(bundle, "diabetes_screening_model.joblib")

APP_HEAD = '''# app.py  -  run with:  streamlit run app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BMI_BINS = [0, 18.5, 25, 30, 35, 40, 200]
'''
APP_BODY = '''

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
'''
app_code = (APP_HEAD + "\n\n" + inspect.getsource(add_features) + "\n\n" +
            "lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / "
            "(1 - np.clip(p, 1e-6, 1 - 1e-6)))\n\n" +
            inspect.getsource(apply_calibration) + APP_BODY)
open("app.py", "w").write(app_code)

# smoke-test the same prediction path the app uses
demo = {c: 0 for c in FEATS}
demo.update(HighBP=1, HighChol=1, CholCheck=1, BMI=34, GenHlth=4, Age=9,
            Sex=1, Education=4, Income=3, PhysHlth=10, DiffWalk=1)
xd = add_features(pd.DataFrame([demo]))[bundle["features"]]
pd_raw = bundle["model"].predict_proba(xd)[:, 1]
pd_cal = float(apply_calibration(CAL_KIND, cal_final, pd_raw)[0])
R["demo_patient"] = dict(profile=demo, risk=pd_cal,
                         flagged=bool(pd_cal >= T_FINAL))
print(f" Demo patient risk: {pd_cal:.3f}  flagged: {pd_cal >= T_FINAL}")

# ============================================================
# OBJECTIVE SCORECARD
# ============================================================
sc = dict(
    obj1_recall=R["test_final"]["recall"], obj1_auc=R["test_models"][BEST]["auc"],
    obj1_prauc=R["test_models"][BEST]["prauc"],
    obj1_recall_met=bool(R["test_final"]["recall"] >= 0.80),
    obj1_auc_met=bool(R["test_models"][BEST]["auc"] >= 0.80))
R["scorecard"] = sc
R["runtime_s"] = float(time.time() - T0)
json.dump(R, open("results.json", "w"), indent=1, default=float)

print("\n" + "=" * 60)
print(" PROJECT COMPLETE")
print("=" * 60)
print(json.dumps(sc, indent=1))
