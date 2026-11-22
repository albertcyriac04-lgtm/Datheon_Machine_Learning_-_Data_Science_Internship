"""
Multi-Class Benchmark Classification Pipeline
==============================================
Dataset  : Iris CSV — https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
Models   : Logistic Regression | Decision Tree
Metrics  : Accuracy, Precision, Recall, F1-Score
Plots    : Pairplot, Confusion Matrices, Decision Boundaries, Benchmark Bar
"""

# ── Standard Library ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ══════════════════════════════════════════════════════════════════════════════
# 0 ▸ GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE  = ["#4C72B0", "#DD8452", "#55A868"]   # blue / orange / green
SEED     = 42

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# ══════════════════════════════════════════════════════════════════════════════
# 1 ▸ DATA LOADING & EDA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  MULTI-CLASS CLASSIFICATION PIPELINE — IRIS DATASET")
print("=" * 65)

# ── Load directly from the given CSV URL ────────────────────────────────────
print(f"\n[1/6] Loading dataset from:\n  {DATA_URL}\n")
df = pd.read_csv(DATA_URL)
print(f"  Shape : {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(df.head(5).to_string())

# Standardise column names (seaborn CSV uses: sepal_length, sepal_width, etc.)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET   = "species"   # string labels in this CSV

# Encode species → integer
le = LabelEncoder()
df["target"] = le.fit_transform(df[TARGET])
CLASS_MAP    = {i: cls for i, cls in enumerate(le.classes_)}
print(f"\n  Classes: {CLASS_MAP}")
print(f"\n  Describe:\n{df[FEATURES].describe().round(3).to_string()}")
print(f"\n  Class distribution:\n{df[TARGET].value_counts().to_string()}")


# ── 1a  Pairplot ────────────────────────────────────────────────────────────
print("\n[2/6] Generating pairplot …")
pair_grid = sns.pairplot(
    df,
    vars=FEATURES,
    hue=TARGET,          # "species" string column
    palette=PALETTE,
    diag_kind="kde",
    plot_kws=dict(alpha=0.7, edgecolors="white", linewidth=0.3),
    corner=True,
)
pair_grid.figure.suptitle("Iris — Feature Pairplot (KDE diagonal)", y=1.01, fontsize=14, fontweight="bold")
pair_grid.savefig("iris_1_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_1_pairplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2 ▸ PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Preprocessing: train/test split (80/20) + StandardScaler …")

X = df[FEATURES].values
y = df["target"].values     # integer-encoded labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train: {X_train_s.shape[0]} samples | Test: {X_test_s.shape[0]} samples")
print(f"    Feature means after scaling (train): {X_train_s.mean(axis=0).round(4)}")
print(f"    Feature stds  after scaling (train): {X_train_s.std(axis=0).round(4)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3 ▸ MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/6] Training models …")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto", random_state=SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, criterion="gini", random_state=SEED
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall":    recall_score(y_test, y_pred, average="weighted"),
        "f1":        f1_score(y_test, y_pred, average="weighted"),
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
    }
    print(f"    ✔ {name} trained  |  CV-5 Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4 ▸ BENCHMARK REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  BENCHMARK REPORT")
print("=" * 65)

metrics_rows = []
for name, r in results.items():
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1-Score  : {r['f1']:.4f}")
    print(f"  CV-5 Acc  : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, r["y_pred"],
        target_names=list(CLASS_MAP.values()), digits=4
    ))
    metrics_rows.append({
        "Model":     name,
        "Accuracy":  round(r["accuracy"],  4),
        "Precision": round(r["precision"], 4),
        "Recall":    round(r["recall"],    4),
        "F1-Score":  round(r["f1"],        4),
        "CV-5 Acc":  f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
    })

metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
print("\n  Summary Table:")
print(metrics_df.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5 ▸ VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Plotting confusion matrices …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices — Iris Test Set (n=30)", fontsize=15, fontweight="bold", y=1.02)

for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=list(CLASS_MAP.values()))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.4f}", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    for text in ax.texts:
        text.set_fontsize(13)

plt.tight_layout()
fig.savefig("iris_2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_2_confusion_matrices.png")

# ── 5b  Benchmark Bar Chart ─────────────────────────────────────────────────
print("\n[6/6] Plotting benchmark bar chart …")

metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
bench_df    = pd.DataFrame(metrics_rows).set_index("Model")[metric_cols]

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = [PALETTE[0], PALETTE[1]]
bench_df.T.plot(
    kind="bar", ax=ax, color=bar_colors,
    edgecolor="white", linewidth=0.6, width=0.65,
)
ax.set_ylim(0.85, 1.02)
ax.set_title("Model Benchmark — Iris Dataset", fontsize=14, fontweight="bold")
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticklabels(metric_cols, rotation=0, fontsize=11)
ax.legend(title="Model", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)

for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", fontsize=8, padding=2)

plt.tight_layout()
fig.savefig("iris_3_benchmark_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_3_benchmark_bar.png")

# ── 5c  Decision Boundary (petal_length vs petal_width — 2 best features) ───
print("      Plotting decision boundaries …")

FEAT_IDX = [2, 3]   # petal_length, petal_width  (most discriminative)
FEAT_NAMES = ["petal_length (cm)", "petal_width (cm)"]

X2_train = X_train_s[:, FEAT_IDX]
X2_test  = X_test_s[:, FEAT_IDX]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Decision Boundaries (petal_length vs petal_width — scaled)",
    fontsize=14, fontweight="bold", y=1.01
)

cmap_light = ListedColormap(["#AEC6E8", "#F5C8A0", "#A8D5B5"])
cmap_dark  = ListedColormap(PALETTE)

for ax, (name, r) in zip(axes, results.items()):
    model_2d = r["model"].__class__(**{
        k: v for k, v in r["model"].get_params().items()
    })
    model_2d.fit(X2_train, y_train)

    x_min, x_max = X2_train[:, 0].min() - 0.6, X2_train[:, 0].max() + 0.6
    y_min, y_max = X2_train[:, 1].min() - 0.6, X2_train[:, 1].max() + 0.6
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_light)

    for cls_idx, cls_name in CLASS_MAP.items():
        mask_tr = y_train == cls_idx
        mask_te = y_test  == cls_idx
        ax.scatter(X2_train[mask_tr, 0], X2_train[mask_tr, 1],
                   c=PALETTE[cls_idx], label=f"{cls_name} (train)",
                   edgecolors="white", linewidth=0.5, s=55, alpha=0.9)
        ax.scatter(X2_test[mask_te, 0], X2_test[mask_te, 1],
                   c=PALETTE[cls_idx], marker="*",
                   edgecolors="black", linewidth=0.6, s=130, alpha=1.0)

    ax.set_title(f"{name}\n(2D projection, ★ = test samples)", fontsize=11)
    ax.set_xlabel(FEAT_NAMES[0], fontsize=10)
    ax.set_ylabel(FEAT_NAMES[1], fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig("iris_4_decision_boundaries.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_4_decision_boundaries.png")

# ── 5d  Decision Tree Visualisation ─────────────────────────────────────────
print("      Plotting decision tree structure …")
dt_model = results["Decision Tree"]["model"]
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    dt_model,
    feature_names=FEATURES,
    class_names=list(CLASS_MAP.values()),
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title("Decision Tree Structure (max_depth=4, Gini)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("iris_5_decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_5_decision_tree.png")

# ── 5e  Feature Importance (Decision Tree) ──────────────────────────────────
fi = pd.Series(dt_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(fi.index, fi.values, color=PALETTE[0], edgecolor="white")
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
ax.set_title("Feature Importances — Decision Tree", fontsize=13, fontweight="bold")
ax.set_xlabel("Gini Importance", fontsize=11)
ax.set_xlim(0, fi.max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
fig.savefig("iris_6_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ iris_6_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — 6 figures saved to working directory")
print("=" * 65)
print("""
  Outputs
  ───────
  iris_1_pairplot.png            EDA pairplot with KDE diagonals
  iris_2_confusion_matrices.png  Side-by-side confusion matrices
  iris_3_benchmark_bar.png       Accuracy / Precision / Recall / F1 bars
  iris_4_decision_boundaries.png 2D decision region plots (petal space)
  iris_5_decision_tree.png       Full Decision Tree diagram
  iris_6_feature_importance.png  Gini feature importance bar chart
""")
