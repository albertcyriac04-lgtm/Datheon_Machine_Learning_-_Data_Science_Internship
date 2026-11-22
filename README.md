# Datheon Machine Learning & Data Science Internship

A comprehensive machine learning repository containing end-to-end classification, regression, and natural language processing pipelines. Each task includes a documented **Jupyter Notebook (`.ipynb`)**, a standalone **Python script (`.py`)**, an exported **Pickle model (`.pkl`)**, and an integrated **Flask web application** configured for deployment on Vercel.

---

## 📁 Repository Structure

```text
Datheon_Machine_Learning_-_Data_Science_Internship/
├── .gitignore                          # Build, virtualenv, and temporary file exclusions
├── vercel.json                         # Vercel serverless deployment routing config
├── requirements.txt                    # Project dependencies (Flask, Scikit-Learn, Pandas, etc.)
├── app.py                             # Flask web app serving UI and prediction APIs
├── README.md                          # Comprehensive project documentation
│
├── models/                             # Serialized trained model bundles (Pickle)
│   ├── iris_model.pkl                 # Task 1: Iris species classification model
│   ├── housing_model.pkl              # Task 2: California housing price regressor
│   ├── churn_model.pkl                # Task 3: Customer churn predictor
│   └── sentiment_model.pkl            # Task 4: NLP review sentiment classifier
│
├── templates/
│   └── index.html                     # Minimal, responsive web interface for live testing
│
├── task1_iris_classification/
│   ├── iris_classification_pipeline.ipynb   # Interactive analysis & benchmark notebook
│   └── iris_classification_pipeline.py      # Standalone reproducible Python script
│
├── task2_housing_regression/
│   ├── task2_housing_regression.ipynb       # Feature engineering & regression notebook
│   └── task2_housing_regression.py          # Standalone reproducible Python script
│
├── task3_churn_classifier/
│   ├── task3_churn_classifier.ipynb         # Class-imbalanced churn classifier notebook
│   └── task3_churn_classifier.py            # Standalone reproducible Python script
│
└── task4_nlp_sentiment/
    ├── sentiment_reviews.csv                # Benchmark customer review dataset
    ├── task4_nlp_sentiment.ipynb            # TF-IDF vectorization & LOO-CV notebook
    └── task4_nlp_sentiment.py               # Standalone reproducible Python script
```

---

## 🔍 Detailed Task Breakdown

### Task 1: Multi-Class Benchmark Classification (Iris Dataset)
- **Objective**: Construct an end-to-end multi-class classification pipeline comparing linear vs non-linear supervised algorithms.
- **Dataset**: [Fisher's Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) (150 observations, 4 numeric measurements: sepal length, sepal width, petal length, petal width; 3 balanced classes: Setosa, Versicolor, Virginica).
- **Data Preprocessing & Scaling**:
  - Target label integer encoding (`LabelEncoder`).
  - Stratified 80/20 train/test split to preserve exact class proportions across partitions.
  - Standardization via `StandardScaler()`, fit strictly on the training set to prevent data leakage.
- **Models Implemented**:
  1. **Multinomial Logistic Regression**: `LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')`
  2. **Decision Tree Classifier**: `DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)`
- **Evaluation & Cross-Validation**:
  - 5-Fold Stratified Cross-Validation on training data.
  - Holdout test evaluation: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score.
  - Generation of confusion matrices and 2D decision boundary projection across petal feature space.
- **Results**: Both models achieved ~96.7% test accuracy, with Petal Length and Petal Width identified as the primary discriminative features.
- **Saved Model**: Exported to `models/iris_model.pkl`.

---

### Task 2: Real Estate Housing Price Prediction (California Housing)
- **Objective**: Perform continuous regression modeling with feature engineering, regularization hyperparameter tuning, and residual error analysis.
- **Dataset**: [California Housing Dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) (20,640 block groups with demographic and housing characteristics).
- **Feature Engineering & Imputation**:
  - Missing value imputation on `total_bedrooms` using median replacement strategy.
  - Three domain interaction ratios engineered:
    - `rooms_per_household = total_rooms / households`
    - `bedrooms_ratio = total_bedrooms / total_rooms`
    - `population_per_household = population / households`
  - Categorical attribute `ocean_proximity` encoded via One-Hot Encoding (`pd.get_dummies`).
  - Feature normalization via `StandardScaler()`.
- **Models & Hyperparameter Optimization**:
  1. **Ordinary Least Squares (OLS)**: `LinearRegression()`
  2. **Ridge Regression (L2 Penalty)**: `Ridge()`, tuned via `GridSearchCV` over 9 alpha magnitudes ($10^{-3}$ to $10^3$).
  3. **Lasso Regression (L1 Penalty & Feature Selection)**: `Lasso(max_iter=10000)`, tuned via `GridSearchCV`.
- **Evaluation**: Evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$), accompanied by residual distribution analysis and coefficient magnitude ranking.
- **Saved Model**: Exported to `models/housing_model.pkl`.

---

### Task 3: Predictive Customer Churn Classifier (Telco Churn)
- **Objective**: Develop an ensemble classification system to identify customer attrition under class-imbalanced conditions.
- **Dataset**: [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (7,043 subscriber accounts, 20 attributes covering demographic, account, and subscribed services).
- **Preprocessing & Imbalance Handling**:
  - Data hygiene: Coerced `TotalCharges` from string to float and imputed missing values.
  - Removed high-cardinality identifier `customerID`.
  - Transformed categorical columns with `LabelEncoder`.
  - Addressed class imbalance (~73.5% non-churn vs ~26.5% churn) using cost-sensitive learning:
    - Random Forest: `class_weight='balanced'`
    - XGBoost: `scale_pos_weight = negative_samples / positive_samples`
- **Models Implemented**:
  1. **Random Forest Classifier**: `n_estimators=300, max_depth=12, class_weight='balanced'`
  2. **XGBoost Classifier**: `n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8`
- **Evaluation**: Assessed on Stratified 5-Fold Cross-Validation targeting ROC-AUC > 80%, along with Precision-Recall curves and extraction of top 10 feature importances (identifying Tenure, MonthlyCharges, and Contract Type as primary risk drivers).
- **Saved Model**: Exported to `models/churn_model.pkl`.

---

### Task 4: NLP Sentiment Classification (Customer Reviews)
- **Objective**: Construct a natural language processing text classifier to categorize review polarity.
- **Dataset**: `task4_nlp_sentiment/sentiment_reviews.csv` (provided review benchmark).
- **NLP Preprocessing Pipeline**:
  - HTML tag stripping via regex.
  - URL removal and case-folding to lowercase.
  - Non-alphabetic character filtering.
  - Stopword removal and WordNet lemmatization.
- **Feature Extraction & Modeling**:
  - `TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)`.
  - Statistical Validation: **Leave-One-Out Cross-Validation (LOO-CV)** to provide unbiased evaluation on small sample corpora.
  - Classifiers compared:
    1. **Multinomial Naive Bayes**: `MultinomialNB(alpha=0.5)`
    2. **Logistic Regression**: `LogisticRegression(C=1.0, max_iter=1000)`
- **Inference**: Integrated `predict_sentiment(text)` function returning predicted label, confidence, and class probabilities.
- **Saved Model**: Exported to `models/sentiment_model.pkl`.

---

## 💻 Local Setup & Execution

### 1. Prerequisites & Environment Setup
Clone the repository:
```bash
git clone https://github.com/albertcyriac04-lgtm/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship
```

Choose your setup:
- **For Flask Web App & Vercel Deployment** (lightweight runtime, <15 MB):
  ```bash
  pip install -r requirements.txt
  ```
- **For Full Model Training & Running Notebooks** (includes Scikit-Learn, XGBoost, Matplotlib, Pandas):
  ```bash
  pip install -r requirements-dev.txt
  ```

### 2. Running the Python Pipelines
You can execute each pipeline script individually:
```bash
python task1_iris_classification/iris_classification_pipeline.py
python task2_housing_regression/task2_housing_regression.py
python task3_churn_classifier/task3_churn_classifier.py
python task4_nlp_sentiment/task4_nlp_sentiment.py
```

### 3. Launching the Flask Web App
Start the local server:
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to interact with all 4 model demos.

---

## 🌐 API Endpoints Reference

The Flask application exposes standard JSON REST endpoints powered by the serialized pickle files:

| Endpoint | Method | Expected Input Payload | Example Output |
|---|---|---|---|
| `/predict/iris` | `POST` | `{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}` | `{"prediction": "setosa", "confidence": 0.999}` |
| `/predict/housing` | `POST` | `{"median_income": 5.2, "housing_median_age": 28, "total_rooms": 2800, "households": 520}` | `{"formatted": "$319,457.12", "prediction_usd": 319457.12}` |
| `/predict/churn` | `POST` | `{"tenure": 6, "monthly_charges": 85.5, "contract": "Month-to-month", "internet_service": "Fiber optic", "tech_support": "No"}` | `{"prediction": "Churn", "churn_probability": 0.98}` |
| `/predict/sentiment` | `POST` | `{"text": "The product quality is exceptional, exceeded all my expectations!"}` | `{"prediction": "positive", "confidence": 0.8571}` |

---

## ☁️ Deployment Instructions (Vercel)

The repository is configured for serverless execution using `vercel.json`:
1. Push your changes to your public GitHub repository.
2. Navigate to [Vercel](https://vercel.com/new).
3. Import `Datheon_Machine_Learning_-_Data_Science_Internship`.
4. Keep the default build and output settings.
5. Click **Deploy**. Vercel will automatically detect `vercel.json` and deploy `app.py` as a serverless WSGI instance.
