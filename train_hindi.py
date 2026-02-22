"""
train_hindi.py — Training pipeline for Hindi medical dataset
Fully cleaned, rare-disease safe, XGBoost-compatible version
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_PATH = "data/updated_result_with_AI_HINDI.csv"
MODEL_DIR = "model/hindi"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("📂 Loading Hindi medical dataset...")
df = pd.read_csv(DATA_PATH)

df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].copy()
df.columns = ['Disease', 'symptoms']
df = df.dropna()

print(f"   Rows: {len(df)}")
print(f"   Diseases: {df['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — BUILD SYMPTOM MATRIX
# ─────────────────────────────────────────────────────────────
print("\n🔧 Extracting symptoms...")

symptom_lists = df['symptoms'].str.split(',').apply(
    lambda x: [s.strip() for s in x if s.strip()]
)

all_symptoms = sorted(set(
    symptom for sublist in symptom_lists for symptom in sublist
))

print(f"   Total unique symptoms: {len(all_symptoms)}")

# Build binary matrix
X = pd.DataFrame(0, index=range(len(df)), columns=all_symptoms)

for i, symptoms in enumerate(symptom_lists):
    X.loc[i, symptoms] = 1

X["Disease"] = df["Disease"].values

print(f"   Matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — REMOVE RARE DISEASES
# ─────────────────────────────────────────────────────────────
print("\n🧹 Removing rare diseases...")

MIN_SAMPLES_PER_CLASS = 5

disease_counts = X["Disease"].value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_SAMPLES_PER_CLASS].index

X = X[X["Disease"].isin(valid_diseases)].reset_index(drop=True)

print(f"   After filtering:")
print(f"   Rows: {len(X)}")
print(f"   Diseases: {X['Disease'].nunique()}")

# Separate features and labels
y = X["Disease"]
X = X.drop("Disease", axis=1)

# ─────────────────────────────────────────────────────────────
# STEP 4 — ENCODE LABELS (AFTER FILTERING)
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"   ✅ Encoded {len(le.classes_)} diseases")

# ─────────────────────────────────────────────────────────────
# STEP 5 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
symptom_list = X.columns.tolist()

joblib.dump(symptom_list, f"{MODEL_DIR}/symptom_list.pkl")
joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

with open(f"{MODEL_DIR}/symptom_list.json", "w", encoding='utf-8') as f:
    json.dump(symptom_list, f, indent=2, ensure_ascii=False)

disease_list = le.classes_.tolist()
with open(f"{MODEL_DIR}/disease_list.json", "w", encoding='utf-8') as f:
    json.dump(disease_list, f, indent=2, ensure_ascii=False)

print("   ✅ Saved artifacts")

# ─────────────────────────────────────────────────────────────
# STEP 6 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# 🔥 IMPORTANT FIX FOR XGBOOST + PANDAS
X_train = X_train.values
X_test = X_test.values

# ─────────────────────────────────────────────────────────────
# STEP 7 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n🏋️  Training models...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"   Training {name}...")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))

    results[name] = {
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std":  round(cv_scores.std(), 4),
        "test_acc": round(test_acc, 4)
    }

    trained_models[name] = model

    print(f"   {name}: CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test={test_acc:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — SELECT BEST MODEL
# ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = trained_models[best_name]

print(f"\n✅ Best model: {best_name}")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

metadata = {
    "language": "Hindi",
    "best_model": best_name,
    "results": results,
    "n_symptoms": len(symptom_list),
    "n_diseases": len(disease_list),
}

with open(f"{MODEL_DIR}/metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# STEP 9 — CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n📋 Classification Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("\n✅ Hindi model trained and saved to /model/hindi")
# ─────────────────────────────────────────────────────────────
# EXTRA DIAGNOSTICS (ADD THIS)
# ─────────────────────────────────────────────────────────────

# Get predictions from best model
y_pred = best_model.predict(X_test)

# Save per-fold CV scores (not just mean/std)
fold_scores_dict = {}
for name, model in trained_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    fold_scores_dict[name] = scores.tolist()

# Classification report
from sklearn.metrics import classification_report
report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0,
    output_dict=True
)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.save(f"{MODEL_DIR}/confusion_matrix.npy", cm)