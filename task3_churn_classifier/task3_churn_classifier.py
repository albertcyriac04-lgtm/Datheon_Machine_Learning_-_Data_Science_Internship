"""
Task 3 — Predictive Customer Churn Classifier
=============================================
Dataset  : Telco Customer Churn (IBM / ICP4D)
           https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Models   : Random Forest | XGBoost
Imbalance: class_weight='balanced' + scale_pos_weight
Metrics  : ROC-AUC, Accuracy, Precision, Recall, F1
Plots    : ROC curves, Confusion matrices, Feature importance (top 10),
           Class distribution, Precision-Recall curve
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_curve, average_precision_score
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGB_AVAILABLE = False
    print("  [WARN] xgboost not installed — using GradientBoostingClassifier as substitute")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SEED     = 42
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
COLORS   = {"Random Forest": "#4C72B0", "XGBoost": "#DD8452"}
OUT_DIR  = "."

print("=" * 65)
print("  TASK 3 — CUSTOMER CHURN CLASSIFIER (RF & XGBoost)")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1 ▸ LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading dataset …")
df = pd.read_csv(DATA_URL)
print(f"    ✔ Loaded from given URL → {df.shape}")
print(df.head(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2 ▸ PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Preprocessing …")

# Fix TotalCharges (sometimes stored as string)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# Binary encode Churn
df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

# Class distribution
churn_counts = df["Churn"].value_counts()
neg, pos = churn_counts[0], churn_counts[1]
imbalance_ratio = neg / pos
print(f"    Class distribution: No={neg} ({neg/(neg+pos)*100:.1f}%)  Yes={pos} ({pos/(neg+pos)*100:.1f}%)")
print(f"    Imbalance ratio: {imbalance_ratio:.2f}:1")

# Encode categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
le       = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f"    ✔ Label-encoded: {cat_cols}")

# ── 2a  Class distribution plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["No Churn", "Churn"], [neg, pos], color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[0].set_title("Class Distribution (Before Balancing)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
for rect, val in zip(axes[0].patches, [neg, pos]):
    axes[0].text(rect.get_x() + rect.get_width()/2, rect.get_height() + 30,
                 f"{val}", ha="center", fontsize=12, fontweight="bold")

# Tenure vs Monthly charges colored by churn
scatter = axes[1].scatter(
    df["tenure"], df["MonthlyCharges"],
    c=df["Churn"], cmap="coolwarm", alpha=0.4, s=15
)
plt.colorbar(scatter, ax=axes[1], label="Churn (1=Yes)")
axes[1].set_title("Tenure vs Monthly Charges (by Churn)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Monthly Charges")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "churn_1_eda.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ churn_1_eda.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3 ▸ SPLIT
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [c for c in df.columns if c != "Churn"]
X = df[FEATURES].values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n    Train: {X_train_s.shape[0]}  |  Test: {X_test_s.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# 4 ▸ MODEL TRAINING (class_weight / scale_pos_weight for imbalance)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Training models …")

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced",
    min_samples_leaf=5, random_state=SEED, n_jobs=-1
)

if XGB_AVAILABLE:
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=SEED, n_jobs=-1
    )
else:
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=SEED
    )

models = {"Random Forest": rf_model, "XGBoost": xgb_model}
results = {}
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred     = model.predict(X_test_s)
    y_proba    = model.predict_proba(X_test_s)[:, 1]
    roc_auc    = roc_auc_score(y_test, y_proba)
    cv_roc     = cross_val_score(model, X_train_s, y_train, cv=skf, scoring="roc_auc")

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision":precision_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc,
        "cv_roc":   cv_roc,
    }
    print(f"    ✔ {name}")
    print(f"       ROC-AUC={roc_auc:.4f}  Acc={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}")
    print(f"       CV-5 ROC-AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5 ▸ BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  BENCHMARK REPORT")
print("=" * 65)
for name, r in results.items():
    print(f"\n  {name}")
    print(f"  {'─'*40}")
    print(classification_report(y_test, r["y_pred"], target_names=["No Churn", "Churn"], digits=4))
    bench_row = {k: round(r[k], 4) for k in ("accuracy","precision","recall","f1","roc_auc")}
    print(f"  ROC-AUC: {bench_row['roc_auc']}  |  CV ROC-AUC: {r['cv_roc'].mean():.4f} ± {r['cv_roc'].std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6 ▸ VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Plotting ROC & PR curves …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROC & Precision-Recall Curves", fontsize=14, fontweight="bold")

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={r['roc_auc']:.4f})", color=COLORS[name])

    prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
    ap = average_precision_score(y_test, r["y_proba"])
    axes[1].plot(rec, prec, lw=2, label=f"{name} (AP={ap:.4f})", color=COLORS[name])

axes[0].plot([0,1],[0,1],"k--",linewidth=1, label="Random (AUC=0.5)")
axes[0].set_title("ROC Curve", fontsize=12); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend(); axes[0].fill_between([0,1],[0,1],alpha=0.05,color="grey")

axes[1].set_title("Precision-Recall Curve", fontsize=12); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "churn_2_roc_pr.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ churn_2_roc_pr.png")

# ── Confusion matrices ───────────────────────────────────────────────────────
print("[5/6] Plotting confusion matrices …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices — Churn Prediction", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}  |  ROC-AUC={r['roc_auc']:.4f}", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "churn_3_confusion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ churn_3_confusion.png")

# ── Feature Importance (top 10) ──────────────────────────────────────────────
print("[6/6] Plotting feature importances …")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Top 10 Feature Importances", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    fi = pd.Series(r["model"].feature_importances_, index=FEATURES)
    fi_top = fi.nlargest(10).sort_values()
    ax.barh(fi_top.index, fi_top.values, color=COLORS[name], edgecolor="white")
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(f"{name} — Feature Importance", fontsize=12)
    ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "churn_4_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ churn_4_feature_importance.png")

print("\n" + "=" * 65)
print("  TASK 3 COMPLETE — 4 figures saved")
print("=" * 65)
print("""
  Outputs
  ───────
  churn_1_eda.png               Class distribution + tenure/charges scatter
  churn_2_roc_pr.png            ROC curve + Precision-Recall curve
  churn_3_confusion.png         Confusion matrices (RF & XGBoost)
  churn_4_feature_importance.png Top-10 feature importances (RF & XGBoost)
""")
