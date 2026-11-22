"""
Task 2 — Real Estate Housing Price Prediction
==============================================
Dataset  : California Housing CSV
           https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
Models   : OLS (LinearRegression) | Ridge (L2) | Lasso (L1)
Tuning   : GridSearchCV across alpha values
Metrics  : RMSE, MAE, R²
Plots    : Correlation heatmap, Predicted vs Actual, Residual distribution,
           Alpha tuning curve, Coefficient comparison
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SEED      = 42
DATA_URL  = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
OUT_DIR   = "."

print("=" * 65)
print("  TASK 2 — HOUSING PRICE PREDICTION (Ridge & Lasso)")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1 ▸ LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading dataset …")
df = pd.read_csv(DATA_URL)
print(f"    ✔ Loaded from given URL → {df.shape}")
print(df.head(3).to_string())
print(f"\n  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2 ▸ EDA — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] EDA — correlation heatmap …")

# Drop non-numeric for correlation
df_num = df.select_dtypes(include=[np.number])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
corr = df_num.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=axes[0],
    cmap="coolwarm", center=0, annot=True, fmt=".2f",
    linewidths=0.4, square=True, cbar_kws={"shrink": 0.8}
)
axes[0].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")

# Target distribution
target_col = "median_house_value" if "median_house_value" in df.columns else "MedHouseVal"
axes[1].hist(df[target_col], bins=50, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[1].set_title("Target Distribution (Median House Value)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Median House Value (\\$)", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].axvline(df[target_col].mean(), color="red", linestyle="--", label=f"Mean: {df[target_col].mean():,.0f}")
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_1_eda.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_1_eda.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3 ▸ FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Feature engineering & preprocessing …")

# Interaction features
if "total_rooms" in df.columns and "households" in df.columns:
    df["rooms_per_household"]    = df["total_rooms"]    / df["households"]
    df["bedrooms_ratio"]         = df.get("total_bedrooms", df["total_rooms"]*0.2) / df["total_rooms"]
    df["population_per_household"] = df["population"]  / df["households"]
    print("    ✔ Engineered: rooms_per_household, bedrooms_ratio, population_per_household")

# One-hot encode ocean_proximity if present
if "ocean_proximity" in df.columns:
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=False)
    print("    ✔ One-hot encoded: ocean_proximity")

FEATURES = [c for c in df.columns if c != target_col]
X = df[FEATURES].values
y = df[target_col].values

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# 4 ▸ MODEL TRAINING + GRIDSEARCH
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Training OLS / Ridge / Lasso with GridSearchCV …")

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

models_config = {
    "OLS": (LinearRegression(), {}),
    "Ridge": (
        Ridge(random_state=SEED),
        {"alpha": ALPHAS},
    ),
    "Lasso": (
        Lasso(max_iter=10000, random_state=SEED),
        {"alpha": ALPHAS},
    ),
}

results    = {}
best_alpha = {}

for name, (estimator, param_grid) in models_config.items():
    if param_grid:
        gs = GridSearchCV(estimator, param_grid, cv=5,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        model = gs.best_estimator_
        best_alpha[name] = gs.best_params_["alpha"]
        print(f"    {name:6s} → best alpha={gs.best_params_['alpha']}")
    else:
        model = estimator
        model.fit(X_train_s, y_train)
        print(f"    OLS   → no regularization")

    y_pred = model.predict(X_test_s)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    results[name] = {
        "model": model, "y_pred": y_pred,
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "residuals": residuals,
    }
    print(f"           RMSE={rmse:,.1f}  MAE={mae:,.1f}  R²={r2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5 ▸ BENCHMARK REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  BENCHMARK REPORT")
print("=" * 65)
bench_rows = []
for name, r in results.items():
    bench_rows.append({
        "Model": name,
        "RMSE":  round(r["RMSE"], 2),
        "MAE":   round(r["MAE"],  2),
        "R²":    round(r["R2"],   4),
    })
bench_df = pd.DataFrame(bench_rows).set_index("Model")
print(bench_df.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 6 ▸ VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"OLS": "#4C72B0", "Ridge": "#DD8452", "Lasso": "#55A868"}

# ── 6a  Predicted vs Actual ──────────────────────────────────────────────────
print("\n[5/6] Plotting predicted vs actual …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Predicted vs Actual — Housing Prices", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(r["y_pred"], y_test, alpha=0.3, s=10, color=COLORS[name])
    lims = [min(r["y_pred"].min(), y_test.min()), max(r["y_pred"].max(), y_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"{name}  |  R²={r['R2']:.4f}", fontsize=11)
    ax.set_xlabel("Predicted Value", fontsize=10)
    ax.set_ylabel("Actual Value", fontsize=10)
    ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_2_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_2_pred_vs_actual.png")

# ── 6b  Residual Distributions ──────────────────────────────────────────────
print("[6/6] Plotting residual distributions …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Residual Distribution — Housing Regression", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    residuals = r["residuals"]
    ax.hist(residuals, bins=60, color=COLORS[name], edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{name}  |  μ={residuals.mean():,.0f}  σ={residuals.std():,.0f}", fontsize=11)
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_3_residuals.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_3_residuals.png")

# ── 6c  Alpha tuning curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regularization Alpha Tuning (5-Fold CV)", fontsize=14, fontweight="bold")
for ax, (reg_name, model_cls) in zip(axes, [("Ridge", Ridge), ("Lasso", Lasso)]):
    rmse_vals = []
    for a in ALPHAS:
        m = model_cls(alpha=a, max_iter=10000, random_state=SEED) if reg_name == "Lasso" else model_cls(alpha=a)
        cv = cross_val_score(m, X_train_s, y_train, cv=5, scoring="neg_root_mean_squared_error")
        rmse_vals.append(-cv.mean())
    ax.semilogx(ALPHAS, rmse_vals, marker="o", color=COLORS[reg_name], linewidth=2)
    best_a = best_alpha.get(reg_name, ALPHAS[np.argmin(rmse_vals)])
    ax.axvline(best_a, color="red", linestyle="--", label=f"Best α={best_a}")
    ax.set_title(f"{reg_name} — RMSE vs Alpha", fontsize=12)
    ax.set_xlabel("Alpha (log scale)", fontsize=11)
    ax.set_ylabel("CV RMSE", fontsize=11)
    ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_4_alpha_tuning.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_4_alpha_tuning.png")

# ── 6d  Coefficient comparison (top 15) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Top 15 Feature Coefficients", fontsize=14, fontweight="bold")
for ax, reg_name in zip(axes, ["Ridge", "Lasso"]):
    coefs = pd.Series(results[reg_name]["model"].coef_, index=FEATURES)
    top   = coefs.abs().nlargest(15).index
    coefs_top = coefs[top].sort_values()
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in coefs_top]
    ax.barh(coefs_top.index, coefs_top.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{reg_name} Coefficients", fontsize=12)
    ax.set_xlabel("Coefficient Value", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_5_coefficients.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_5_coefficients.png")

# ── 6e  Benchmark bar ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Benchmark — Housing Regression", fontsize=14, fontweight="bold")
for ax, metric in zip(axes, ["RMSE", "MAE", "R²"]):
    vals  = [results[n][{"RMSE":"RMSE","MAE":"MAE","R²":"R2"}[metric]] for n in results]
    names = list(results.keys())
    bars  = ax.bar(names, vals, color=[COLORS[n] for n in names], edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f" if metric in ("RMSE","MAE") else "%.4f", padding=3, fontsize=10)
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel(metric, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    if metric == "R²": ax.set_ylim(max(0, min(vals)-0.05), 1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "housing_6_benchmark.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ housing_6_benchmark.png")

print("\n" + "=" * 65)
print("  TASK 2 COMPLETE — 6 figures saved")
print("=" * 65)
print("""
  Outputs
  ───────
  housing_1_eda.png              Correlation heatmap + target distribution
  housing_2_pred_vs_actual.png   Predicted vs Actual scatter (all 3 models)
  housing_3_residuals.png        Residual histograms
  housing_4_alpha_tuning.png     CV RMSE vs Alpha (Ridge & Lasso)
  housing_5_coefficients.png     Top-15 feature coefficients
  housing_6_benchmark.png        RMSE / MAE / R² comparison bar
""")
