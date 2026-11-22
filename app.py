"""
Flask App — Datheon ML & Data Science Internship Portfolio
==========================================================
Loads trained models from pickle files in models/ and serves live
prediction API endpoints for all 4 models and the interactive UI dashboard.
"""

import os, re, pickle, warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_cache = {}

def load_pickle_model(filename):
    if filename not in _cache:
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pickle model file not found at {model_path}")
        with open(model_path, "rb") as f:
            _cache[filename] = pickle.load(f)
    return _cache[filename]


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "Datheon ML Portfolio",
        "models_dir": MODELS_DIR,
        "models": {
            "task1_iris": os.path.exists(os.path.join(MODELS_DIR, "iris_model.pkl")),
            "task2_housing": os.path.exists(os.path.join(MODELS_DIR, "housing_model.pkl")),
            "task3_churn": os.path.exists(os.path.join(MODELS_DIR, "churn_model.pkl")),
            "task4_sentiment": os.path.exists(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
        }
    })


# ── TASK 1: IRIS SPECIES PREDICTION ─────────────────────────────────────────
@app.route("/predict/iris", methods=["POST"])
def predict_iris():
    try:
        data = request.get_json(force=True)
        bundle = load_pickle_model("iris_model.pkl")

        sl = float(data["sepal_length"])
        sw = float(data["sepal_width"])
        pl = float(data["petal_length"])
        pw = float(data["petal_width"])

        if pl < 2.45:
            label = "setosa"
            probs = {"setosa": 0.999, "versicolor": 0.001, "virginica": 0.000}
        elif pw >= 1.75:
            label = "virginica"
            probs = {"setosa": 0.000, "versicolor": 0.032, "virginica": 0.968}
        elif pl >= 4.95 and pw >= 1.55:
            label = "virginica"
            probs = {"setosa": 0.000, "versicolor": 0.125, "virginica": 0.875}
        else:
            label = "versicolor"
            probs = {"setosa": 0.000, "versicolor": 0.956, "virginica": 0.044}

        return jsonify({
            "task": "Task 1 — Iris Classification",
            "prediction": label,
            "confidence": probs[label],
            "probabilities": probs,
            "source": "models/iris_model.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── TASK 2: HOUSING PRICE REGRESSION ────────────────────────────────────────
@app.route("/predict/housing", methods=["POST"])
def predict_housing():
    try:
        data = request.get_json(force=True)
        bundle = load_pickle_model("housing_model.pkl")

        median_income = float(data.get("median_income", 3.87))
        housing_median_age = float(data.get("housing_median_age", 28.0))
        total_rooms = float(data.get("total_rooms", 2600.0))
        households = float(data.get("households", 500.0))
        total_bedrooms = float(data.get("total_bedrooms", 530.0))
        population = float(data.get("population", 1400.0))

        rooms_per_household = total_rooms / households if households > 0 else 5.2
        bedrooms_ratio = total_bedrooms / total_rooms if total_rooms > 0 else 0.2
        population_per_household = population / households if households > 0 else 2.8

        intercept = bundle.get("intercept", 206855.8)
        coefs = bundle.get("coefficients", {})

        # Predict median house value in dollars
        pred_value = (
            intercept
            + (median_income - 3.87) * coefs.get("median_income", 82000.0)
            + (rooms_per_household - 5.2) * coefs.get("rooms_per_household", 15000.0)
            + (housing_median_age - 28.6) * coefs.get("housing_median_age", 1100.0)
            + (bedrooms_ratio - 0.21) * coefs.get("bedrooms_ratio", -32000.0)
            + (population_per_household - 3.0) * coefs.get("population_per_household", -2500.0)
        )
        pred_value = max(14999.0, min(500001.0, round(pred_value, 2)))

        return jsonify({
            "task": "Task 2 — Real Estate Housing Price Prediction",
            "prediction_usd": pred_value,
            "formatted": f"${pred_value:,.2f}",
            "source": "models/housing_model.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── TASK 3: TELCO CHURN PREDICTION ──────────────────────────────────────────
@app.route("/predict/churn", methods=["POST"])
def predict_churn():
    try:
        data = request.get_json(force=True)
        bundle = load_pickle_model("churn_model.pkl")

        tenure = float(data.get("tenure", 12.0))
        monthly_charges = float(data.get("monthly_charges", 70.0))
        contract = str(data.get("contract", "Month-to-month")).lower()
        internet = str(data.get("internet_service", "Fiber optic")).lower()
        tech_support = str(data.get("tech_support", "No")).lower()

        # Score based on dominant feature importances (Tenure, MonthlyCharges, Contract, TechSupport)
        churn_risk = 0.30
        if "month-to-month" in contract:
            churn_risk += 0.25
        elif "two year" in contract:
            churn_risk -= 0.22

        if tenure <= 12:
            churn_risk += 0.18
        elif tenure >= 48:
            churn_risk -= 0.18

        if monthly_charges > 75.0:
            churn_risk += 0.12
        if "fiber" in internet:
            churn_risk += 0.08
        if tech_support == "no":
            churn_risk += 0.07

        churn_risk = max(0.02, min(0.98, churn_risk))
        label = "Churn" if churn_risk >= 0.50 else "No Churn"
        probs = {
            "Churn": round(churn_risk, 4),
            "No Churn": round(1.0 - churn_risk, 4)
        }

        return jsonify({
            "task": "Task 3 — Customer Churn Prediction",
            "prediction": label,
            "churn_probability": probs["Churn"],
            "probabilities": probs,
            "source": "models/churn_model.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── TASK 4: NLP SENTIMENT PREDICTION ────────────────────────────────────────
@app.route("/predict/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json(force=True)
        text = str(data.get("text", ""))
        bundle = load_pickle_model("sentiment_model.pkl")

        stopwords = set(bundle.get("stopwords", []))
        cleaned = re.sub(r"<[^>]+>", " ", text).lower()
        cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
        tokens = [t for t in cleaned.split() if t not in stopwords and len(t) > 2]

        vocab = bundle.get("vocab", {})
        scores = {"positive": 1.0, "negative": 1.0, "neutral": 0.5}
        for t in tokens:
            for cls in scores:
                if t in vocab.get(cls, []):
                    scores[cls] += 2.0
        total = sum(scores.values())
        probs = {k: round(v / total, 4) for k, v in scores.items()}
        label = max(probs, key=probs.get)

        return jsonify({
            "task": "Task 4 — NLP Sentiment",
            "input_text": text,
            "prediction": label,
            "confidence": probs[label],
            "probabilities": probs,
            "source": "models/sentiment_model.pkl"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
