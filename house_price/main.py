# ============================================================
# House Prices - Advanced Regression Techniques (Kaggle)
# ============================================================

# --- Step 1: Setup & Imports ---
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

# --- Helper function ---
def rmse_cv(model, X, y, folds=5, random_state=42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
    return scores

# --- Step 2: Load Data ---
DATA_DIR = Path('.')
train_path = DATA_DIR / 'train.csv'
test_path = DATA_DIR / 'test.csv'

if not train_path.exists():
    raise FileNotFoundError("❌ train.csv not found. Please download it from Kaggle and place it here.")

train = pd.read_csv(train_path)
print("✅ Train shape:", train.shape)

if test_path.exists():
    test = pd.read_csv(test_path)
    print("✅ Test shape:", test.shape)
else:
    test = None
    print("⚠️ test.csv not found; skipping submission step.")

# --- Step 3: Target & Features ---
y = np.log1p(train["SalePrice"])  # log transform target (stabilizes variance)
X = train.drop(columns=["SalePrice"])

# --- Step 4: Feature Types ---
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric: {len(num_feats)} features, Categorical: {len(cat_feats)} features")

# --- Step 5: Preprocessing Pipelines ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_feats),
        ("cat", categorical_transformer, cat_feats)
    ]
)

# --- Step 6: Base Models ---
ridge = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=10.0))
])

rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

hgb = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(random_state=42))
])

# --- Step 7: Cross-validation baseline ---
print("\n🔹 Ridge CV RMSE:", rmse_cv(ridge, X, y).mean())
print("🔹 RandomForest CV RMSE:", rmse_cv(rf, X, y).mean())
print("🔹 HistGradientBoosting CV RMSE:", rmse_cv(hgb, X, y).mean())

# --- Step 8: Hyperparameter Tuning ---
rf_param_dist = {
    "model__n_estimators": randint(100, 400),
    "model__max_depth": randint(6, 30),
    "model__max_features": ["auto", "sqrt", 0.2, 0.5]
}

hgb_param_dist = {
    "model__learning_rate": uniform(0.01, 0.2),
    "model__max_iter": randint(100, 500),
    "model__max_leaf_nodes": randint(10, 50)
}

print("\n⏳ Tuning RandomForest...")
rf_search = RandomizedSearchCV(
    rf, rf_param_dist, n_iter=10,
    scoring="neg_root_mean_squared_error", cv=3,
    random_state=42, n_jobs=-1
)
rf_search.fit(X, y)
print("Best RF params:", rf_search.best_params_)

print("\n⏳ Tuning HistGradientBoosting...")
hgb_search = RandomizedSearchCV(
    hgb, hgb_param_dist, n_iter=10,
    scoring="neg_root_mean_squared_error", cv=3,
    random_state=42, n_jobs=-1
)
hgb_search.fit(X, y)
print("Best HGB params:", hgb_search.best_params_)

# --- Step 9: Stacking Ensemble ---
estimators = [
    ("ridge", ridge),
    ("rf", rf_search.best_estimator_),
    ("hgb", hgb_search.best_estimator_)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    n_jobs=-1
)

print("\n🔹 Stacking CV RMSE:", rmse_cv(stack, X, y).mean())

# --- Step 10: Train on full data ---
stack.fit(X, y)

# --- Step 11: Predict & Save Submission ---
if test is not None:
    preds = np.expm1(stack.predict(test))  # revert log1p
    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": preds
    })
    submission.to_csv("submission_stack.csv", index=False)
    print("\n✅ submission_stack.csv saved successfully!")
else:
    print("⚠️ No test.csv found; skipping submission creation.")
