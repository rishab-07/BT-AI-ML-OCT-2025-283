from gettext import install


"""
Titanic classification project
File: titanic_classification_pipeline.py
Description: end-to-end pipeline for the Kaggle Titanic dataset: EDA, preprocessing,
feature engineering, training (Logistic Regression & Decision Tree), evaluation,
and producing submission.csv for Kaggle.

How to use:
1. Place `train.csv` and `test.csv` (from Kaggle Titanic competition) in the same folder as this file.
2. Install requirements:
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
3. Run in terminal or Jupyter:
   python titanic_classification_pipeline.py
   (or open in Jupyter and run cells manually)

Outputs:
- models/logistic_regression.joblib
- models/decision_tree.joblib
- outputs/metrics.txt (evaluation on training hold-out / CV)
- outputs/submission_lr.csv (Kaggle submission using logistic regression)
- outputs/submission_dt.csv (Kaggle submission using decision tree)

Notes:
- This file is written to be readable and easy to modify. It uses standard ML choices for
  a beginner-friendly pipeline.

"""

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- Utility functions ---------------------------

def load_data(train_path: str = "train.csv", test_path: str = "test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def title_from_name(name: str) -> str:
    """Extract title (Mr, Mrs, Miss, etc.) from a passenger name."""
    match = re.search(r",\s*([^.]*)\.", name)
    if match:
        return match.group(1).strip()
    return "Unknown"


def preprocess(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """Feature engineering and basic cleaning for Titanic dataset.

    Steps:
    - Title extraction from Name
    - Fill missing Fare by median
    - Fill missing Embarked with mode
    - Create 'FamilySize' from SibSp + Parch
    - Create 'IsAlone' based on FamilySize
    - Map Cabin to deck (first letter) and fill missing as 'U'
    - Keep a concise set of features for modelling
    """
    df = df.copy()

    # Title
    df['Title'] = df['Name'].apply(title_from_name)
    # Group rare titles
    common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
    df['Title'] = df['Title'].apply(lambda t: t if t in common_titles else 'Rare')

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin deck
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0])

    # Fill Embarked
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Age will be imputed later in pipeline

    # Drop columns not useful directly
    drop_cols = ['PassengerId', 'Ticket', 'Name', 'Cabin']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    return df


# --------------------------- Build pipelines ---------------------------

def build_preprocessor(numeric_features, categorical_features):
    """Return a ColumnTransformer for numeric and categorical processing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def build_models(preprocessor):
    """Create model pipelines for logistic regression and decision tree."""
    pipe_lr = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipe_dt = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    return pipe_lr, pipe_dt


# --------------------------- Training & Evaluation ---------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test, name='model'):
    """Train model and report common metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    roc = roc_auc_score(y_test, proba) if proba is not None else float('nan')

    print(f"--- Evaluation for {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    return {
        'accuracy': acc,
        'roc_auc': roc,
        'report': report,
        'confusion_matrix': cm
    }


# --------------------------- Main flow ---------------------------

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    train, test = load_data()
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)

    train_p = preprocess(train, is_train=True)
    test_p = preprocess(test, is_train=False)

    # Separate target
    y = train_p['Survived']
    X = train_p.drop(columns=['Survived'])

    # Decide features
    numeric_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print('Numeric features:', numeric_features)
    print('Categorical features:', categorical_features)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipe_lr, pipe_dt = build_models(preprocessor)

    # Train-test split for quick evaluation (hold-out)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Evaluate logistic regression
    metrics_lr = evaluate_model(pipe_lr, X_train, X_val, y_train, y_val, name='Logistic Regression')
    joblib.dump(pipe_lr, 'models/logistic_regression.joblib')

    # Evaluate decision tree
    metrics_dt = evaluate_model(pipe_dt, X_train, X_val, y_train, y_val, name='Decision Tree')
    joblib.dump(pipe_dt, 'models/decision_tree.joblib')

    # Cross-validation scores (5-fold)
    print('\nCross-validation (5-fold) Logistic Regression accuracy:')
    cv_scores_lr = cross_val_score(pipe_lr, X, y, cv=5, scoring='accuracy')
    print(cv_scores_lr)
    print('Mean:', cv_scores_lr.mean())

    print('\nCross-validation (5-fold) Decision Tree accuracy:')
    cv_scores_dt = cross_val_score(pipe_dt, X, y, cv=5, scoring='accuracy')
    print(cv_scores_dt)
    print('Mean:', cv_scores_dt.mean())

    # Train on full train data, create submission for test set
    best_lr = pipe_lr.fit(X, y)
    preds_test_lr = best_lr.predict(test_p)
    submission_lr = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds_test_lr.astype(int)})
    submission_lr.to_csv('outputs/submission_lr.csv', index=False)

    best_dt = pipe_dt.fit(X, y)
    preds_test_dt = best_dt.predict(test_p)
    submission_dt = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds_test_dt.astype(int)})
    submission_dt.to_csv('outputs/submission_dt.csv', index=False)

    # Save metrics
    with open('outputs/metrics.txt', 'w') as f:
        f.write('Logistic Regression metrics (hold-out):\n')
        f.write(str(metrics_lr) + '\n\n')
        f.write('Decision Tree metrics (hold-out):\n')
        f.write(str(metrics_dt) + '\n\n')
        f.write('CV LR: ' + str(list(cv_scores_lr)) + '\n')
        f.write('CV DT: ' + str(list(cv_scores_dt)) + '\n')

    print('\nDone. Models and outputs saved in models/ and outputs/ directories.')


if __name__ == '__main__':
    main()
