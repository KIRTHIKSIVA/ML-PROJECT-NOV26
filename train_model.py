import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =========================================================
# 1. LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv("data/mini_cicids2017.csv")

print("Dataset shape:", df.shape)


# =========================================================
# 2. CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)


# =========================================================
# 3. REMOVE LABEL COLUMNS WE DON'T NEED
# =========================================================

target_column = "ProjectLabel"

X = df.drop(columns=[target_column])
y = df[target_column]


# =========================================================
# 4. REMOVE NON-NUMERIC COLUMNS
# =========================================================

print("\nOriginal feature count:", X.shape[1])

numeric_columns = X.select_dtypes(
    include=["number"]
).columns

X = X[numeric_columns]

print(
    "Numeric feature count:",
    X.shape[1]
)


# =========================================================
# 5. HANDLE INFINITY
# =========================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# =========================================================
# 6. HANDLE MISSING VALUES
# =========================================================

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)


# =========================================================
# 7. ENCODE TARGET LABELS
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")

for number, name in enumerate(
    label_encoder.classes_
):
    print(number, "=", name)


# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================================
# 9. DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
}


# =========================================================
# 10. TRAIN MODELS
# =========================================================

results = {}

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


# =========================================================
# 11. SELECT BEST MODEL
# =========================================================

best_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_name]["model"]

print("\n================================")
print("BEST MODEL")
print("================================")

print(best_name)

print(
    "F1 Score:",
    results[best_name]["f1"]
)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 13. SAVE EVERYTHING
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

joblib.dump(
    imputer,
    "models/imputer.pkl"
)

joblib.dump(
    list(numeric_columns),
    "models/features.pkl"
)


print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    "models/best_model.pkl"
)