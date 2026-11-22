"""
Task 4 — NLP Sentiment Classification (TF-IDF)
===============================================
Dataset  : sentiment_reviews.csv  (provided CSV — 10 reviews)
           Columns: review_id, review_text, sentiment
Pipeline : Clean → Tokenize → Lemmatize → TF-IDF → Classify
Models   : Multinomial Naive Bayes | Logistic Regression
Metrics  : Accuracy, Precision, Recall, F1 (weighted)
CV       : Leave-One-Out (LOO) — appropriate for very small dataset
Plots    : Class distribution, top TF-IDF words, confusion matrices,
           benchmark bar, word cloud (optional)
Extra    : predict_sentiment() interactive inference function
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import warnings, os, re, string
warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ── Optional NLP libs ───────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    for pkg in ["stopwords", "wordnet", "punkt", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    NLTK_AVAILABLE = True
    STOPWORDS      = set(stopwords.words("english"))
    LEMMATIZER     = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = {
        "i","me","my","we","our","you","your","he","him","his","she","her",
        "it","its","they","them","their","this","that","these","those","am",
        "is","are","was","were","be","been","have","has","had","do","does",
        "did","a","an","the","and","but","if","or","as","of","at","by","for",
        "with","about","into","through","to","from","in","out","on","off",
        "not","no","nor","so","than","very","just","can","will","don","should",
    }
    LEMMATIZER = None
    print("  [INFO] NLTK not installed — using basic text cleaning (install nltk for better results)")

try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except ImportError:
    WC_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SEED    = 42
OUT_DIR = "."

# Colour map per class
COLORS_CLASS = {"positive": "#55A868", "negative": "#DD8452", "neutral": "#4C72B0"}
MODEL_COLORS  = {"Multinomial NB": "#4C72B0", "Logistic Regression": "#DD8452"}

print("=" * 65)
print("  TASK 4 — NLP SENTIMENT CLASSIFICATION (TF-IDF)")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1 ▸ LOAD THE PROVIDED CSV DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading provided sentiment_reviews.csv …")

# Try the uploaded path first; fallback to local copy in working directory
UPLOADED_PATH = os.path.join(
    r"C:\Users\albert\.gemini\antigravity\brain"
    r"\d2b5812c-f7b5-4143-a723-c7f636a38834\.user_uploaded",
    "media_1795360003844_374a95b1.csv"
)
LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "sentiment_reviews.csv")

for path in (UPLOADED_PATH, LOCAL_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"    ✔ Loaded from: {path}")
        break
else:
    raise FileNotFoundError(
        "sentiment_reviews.csv not found.\n"
        "Place it in the same folder as this script or use the uploaded path."
    )

# ── Normalise column names ───────────────────────────────────────────────────
df.columns = [c.strip().lower() for c in df.columns]

# Accept either 'review_text' or 'review' as the text column
if "review_text" in df.columns:
    df = df.rename(columns={"review_text": "text"})
elif "text" not in df.columns:
    str_cols = df.select_dtypes(include="object").columns.tolist()
    df = df.rename(columns={str_cols[0]: "text"})

df["sentiment"] = df["sentiment"].str.strip().str.lower()
df = df.dropna(subset=["text", "sentiment"])
df = df[df["sentiment"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

print(f"\n  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  All reviews:")
for _, row in df.iterrows():
    print(f"    [{row['sentiment']:8s}] {row['text']}")
print(f"\n  Class distribution:\n{df['sentiment'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2 ▸ CLASS DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Dataset Overview — Sentiment Reviews", fontsize=14, fontweight="bold")

counts = df["sentiment"].value_counts()
bar_colors = [COLORS_CLASS.get(s, "#888") for s in counts.index]
bars = axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", width=0.5)
axes[0].bar_label(bars, padding=3, fontsize=12, fontweight="bold")
axes[0].set_title("Class Distribution", fontsize=12)
axes[0].set_xlabel("Sentiment"); axes[0].set_ylabel("Count")
axes[0].set_ylim(0, counts.max() * 1.3)

# Review text length distribution
df["text_length"] = df["text"].str.split().str.len()
for cls, grp in df.groupby("sentiment"):
    axes[1].hist(grp["text_length"], bins=5, alpha=0.7,
                 label=cls, color=COLORS_CLASS.get(cls, "#888"), edgecolor="white")
axes[1].set_title("Review Word-Count Distribution", fontsize=12)
axes[1].set_xlabel("Number of Words"); axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "nlp_1_dataset_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n    ✔ nlp_1_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3 ▸ TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Text cleaning (HTML strip → lowercase → punctuation → stopwords → lemmatize) …")

def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline."""
    text = re.sub(r"<[^>]+>", " ", str(text))          # strip HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = text.lower()                                  # lowercase
    text = re.sub(r"[^a-z\s']", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

print("\n  Before / After cleaning:")
for _, row in df[["text", "clean_text", "sentiment"]].head(5).iterrows():
    print(f"    [{row['sentiment']:8s}]")
    print(f"      Raw  : {row['text']}")
    print(f"      Clean: {row['clean_text']}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4 ▸ ENCODE LABELS
# ══════════════════════════════════════════════════════════════════════════════
le      = LabelEncoder()
y       = le.fit_transform(df["sentiment"])
classes = le.classes_
X_text  = df["clean_text"].values

print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"  Total samples: {len(X_text)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5 ▸ TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] TF-IDF vectorization (max_features=500, ngram_range=(1,2), sublinear_tf) …")

# Use smaller max_features appropriate for the small corpus
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(X_text)
print(f"    Feature matrix: {X_tfidf.shape}  ({X_tfidf.shape[0]} docs × {X_tfidf.shape[1]} TF-IDF features)")

# ── Top TF-IDF terms per class ───────────────────────────────────────────────
print("\n[4/6] Plotting top TF-IDF terms per class …")

present_classes = [c for c in classes if (df["sentiment"] == c).sum() > 0]
ncols = len(present_classes)
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
if ncols == 1:
    axes = [axes]
fig.suptitle("Top TF-IDF Terms by Sentiment Class", fontsize=14, fontweight="bold")

for ax, cls_name in zip(axes, present_classes):
    cls_idx  = le.transform([cls_name])[0]
    mask     = y == cls_idx
    if mask.sum() == 0:
        continue
    scores   = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
    top_n    = min(10, (scores > 0).sum())
    top_idx  = scores.argsort()[-top_n:][::-1]
    top_terms = [tfidf.get_feature_names_out()[i] for i in top_idx]
    top_vals  = scores[top_idx]
    ax.barh(top_terms[::-1], top_vals[::-1],
            color=COLORS_CLASS.get(cls_name, "#888"), edgecolor="white")
    ax.set_title(f"'{cls_name}'  (n={mask.sum()})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean TF-IDF Score")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "nlp_2_tfidf_terms.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ nlp_2_tfidf_terms.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6 ▸ MODEL TRAINING  — Scikit-Learn Pipelines with Leave-One-Out CV
#    (LOO is the standard choice when n < 30; maximises training data usage)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Training MultinomialNB + LogisticRegression (Leave-One-Out CV) …")
print(f"      Dataset size: n={len(X_text)}  →  LOO gives {len(X_text)} folds")

PIPE_CONFIGS = {
    "Multinomial NB": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   MultinomialNB(alpha=0.5)),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                      multi_class="auto", random_state=SEED)),
    ]),
}

loo     = LeaveOneOut()
results = {}

for name, pipe in PIPE_CONFIGS.items():
    # LOO cross-validated predictions (each sample predicted once held-out)
    y_pred_loo = cross_val_predict(pipe, X_text, y, cv=loo)

    acc  = accuracy_score(y, y_pred_loo)
    prec = precision_score(y, y_pred_loo, average="weighted", zero_division=0)
    rec  = recall_score(y, y_pred_loo, average="weighted", zero_division=0)
    f1   = f1_score(y, y_pred_loo, average="weighted", zero_division=0)

    # Also fit on full data for inference
    pipe.fit(X_text, y)

    results[name] = {
        "pipe":      pipe,
        "y_pred":    y_pred_loo,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }

    print(f"\n  {name}")
    print(f"  {'─'*50}")
    print(f"  LOO Accuracy : {acc:.4f}")
    print(f"  LOO Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"\n  Classification Report (LOO predictions):")
    print(classification_report(y, y_pred_loo, target_names=classes,
                                digits=4, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
# 7 ▸ VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Generating plots …")

# ── 7a  Confusion matrices (LOO predictions) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices — LOO Predictions (Sentiment)", fontsize=14, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y, r["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}  |  LOO Acc={r['accuracy']:.4f}", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "nlp_3_confusion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ nlp_3_confusion.png")

# ── 7b  Benchmark bar ────────────────────────────────────────────────────────
metrics   = ["accuracy", "precision", "recall", "f1"]
met_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
bench_data = {n: [results[n][m] for m in metrics] for n in results}
bench_df   = pd.DataFrame(bench_data, index=met_labels)

fig, ax = plt.subplots(figsize=(10, 5))
bench_df.T.plot(kind="bar", ax=ax,
                color=["#4C72B0", "#DD8452", "#55A868", "#c0785c"],
                edgecolor="white", width=0.65)
ax.set_title("NLP Model Benchmark — Sentiment Classification (LOO-CV)", fontsize=13, fontweight="bold")
ax.set_ylabel("Score"); ax.set_xticklabels(list(results.keys()), rotation=0)
ax.set_ylim(0, 1.15); ax.legend(title="Metric")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "nlp_4_benchmark.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    ✔ nlp_4_benchmark.png")

# ── 7c  Word cloud (if wordcloud installed) ──────────────────────────────────
if WC_AVAILABLE:
    fig, axes = plt.subplots(1, len(present_classes), figsize=(6 * len(present_classes), 4))
    if len(present_classes) == 1:
        axes = [axes]
    fig.suptitle("Word Clouds — Provided Dataset", fontsize=14, fontweight="bold")
    for ax, cls_name in zip(axes, present_classes):
        texts = " ".join(df.loc[df["sentiment"] == cls_name, "clean_text"])
        if not texts.strip():
            ax.axis("off"); continue
        cmap_name = "Greens" if cls_name == "positive" else ("Reds" if cls_name == "negative" else "Blues")
        wc = WordCloud(width=400, height=300, background_color="white",
                       colormap=cmap_name, max_words=50).generate(texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off"); ax.set_title(f"{cls_name.capitalize()}  (n={df['sentiment'].eq(cls_name).sum()})", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nlp_5_wordclouds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✔ nlp_5_wordclouds.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8 ▸ predict_sentiment() — INFERENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# Pick best model by LOO accuracy
best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_pipe  = results[best_name]["pipe"]
print(f"\n    Best model: {best_name}  (LOO Accuracy: {results[best_name]['accuracy']:.4f})")

def predict_sentiment(text: str, pipe=best_pipe, label_enc=le) -> dict:
    """
    Predict the sentiment label for any raw text string.

    Parameters
    ----------
    text : str  — raw review text (HTML, punctuation, etc. handled internally)

    Returns
    -------
    dict  — {sentiment, confidence, probabilities}
    """
    cleaned   = clean_text(text)
    pred_idx  = pipe.predict([cleaned])[0]
    proba     = pipe.predict_proba([cleaned])[0]
    label     = label_enc.inverse_transform([pred_idx])[0]
    conf      = proba[pred_idx]
    prob_dict = {label_enc.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}
    return {"sentiment": label, "confidence": round(float(conf), 4), "probabilities": prob_dict}

# ── Live test on the provided reviews ──────────────────────────────────────
print("\n" + "=" * 65)
print("  SENTIMENT PREDICTIONS — PROVIDED DATASET REVIEWS")
print("=" * 65)
for _, row in df.iterrows():
    result = predict_sentiment(row["text"])
    match  = "✓" if result["sentiment"] == row["sentiment"] else "✗"
    bar    = "▓" * int(result["confidence"] * 20)
    print(f"\n  {match} True: [{row['sentiment']:8s}]  Pred: [{result['sentiment']:8s}]  "
          f"Conf: {result['confidence']:.2%}  {bar}")
    print(f"    Text: {row['text'][:75]}")

# ── Interactive live test examples ─────────────────────────────────────────
print("\n" + "=" * 65)
print("  INTERACTIVE EXAMPLES")
print("=" * 65)
test_phrases = [
    "The product is absolutely amazing and works perfectly!",
    "Terrible quality, broken on arrival, complete waste of money.",
    "It is okay. Nothing special but does the basic job fine.",
]
for phrase in test_phrases:
    r   = predict_sentiment(phrase)
    bar = "▓" * int(r["confidence"] * 20)
    print(f"\n  Input  : {phrase}")
    print(f"  Result : [{r['sentiment'].upper():10s}]  Conf: {r['confidence']:.2%}  {bar}")
    print(f"  Probs  : {r['probabilities']}")

print("\n" + "=" * 65)
print("  TASK 4 COMPLETE")
print("=" * 65)
print("""
  Outputs
  ───────
  nlp_1_dataset_overview.png   Class distribution + word count histogram
  nlp_2_tfidf_terms.png        Top TF-IDF terms per sentiment class
  nlp_3_confusion.png          Confusion matrices (LOO-CV, NB & LR)
  nlp_4_benchmark.png          Accuracy/Precision/Recall/F1 benchmark bar
  nlp_5_wordclouds.png         Word clouds (requires: pip install wordcloud)

  Function: predict_sentiment("your custom text here")

  Note: Leave-One-Out CV is used (n=10 dataset) — maximises training
        data utilisation for very small corpora.
""")

# ── Allow interactive CLI mode ──────────────────────────────────────────────
if __name__ == "__main__":
    print("  Enter phrases to test (type 'quit' to exit):")
    while True:
        try:
            user_input = input("\n  > ")
            if user_input.strip().lower() in ("quit", "exit", "q", ""):
                break
            r = predict_sentiment(user_input)
            print(f"  → Sentiment : {r['sentiment'].upper()}")
            print(f"     Confidence: {r['confidence']:.2%}")
            print(f"     Scores    : {r['probabilities']}")
        except (KeyboardInterrupt, EOFError):
            break
    print("\n  Done.")
