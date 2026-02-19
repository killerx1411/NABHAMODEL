"""
train.py — Full production training pipeline
Adapted for BERT-based Hindi/English medical dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
DATA_PATH  = "data/updated_result_with_BERT.csv"
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("📂 Loading BERT medical dataset...")
df = pd.read_csv(DATA_PATH)

# Use pseudonymized columns
df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].copy()
df.columns = ['Disease', 'symptoms']

df = df.dropna()

print(f"   Rows: {len(df)}")
print(f"   Diseases: {df['Disease'].nunique()}")

# Clean text
df["Disease"] = df["Disease"].str.strip().str.title()
df["symptoms"] = df["symptoms"].str.strip().str.lower()

# ─────────────────────────────────────────────────────────────
# STEP 2 — BUILD BINARY SYMPTOM MATRIX
# ─────────────────────────────────────────────────────────────
print("\n🔧 Building binary symptom matrix...")

symptom_lists = df['symptoms'].str.split(',').apply(
    lambda x: [s.strip() for s in x if s.strip()]
)

all_symptoms = sorted(set(
    symptom for sublist in symptom_lists for symptom in sublist
))

print(f"   Unique symptoms: {len(all_symptoms)}")

# Create binary matrix efficiently
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i, symptoms in enumerate(symptom_lists):
    X.loc[i, symptoms] = 1

X["Disease"] = df["Disease"]

print(f"   Matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — FEATURES & LABELS
# ─────────────────────────────────────────────────────────────
y = X["Disease"]
X = X.drop("Disease", axis=1)

symptom_list = X.columns.tolist()
joblib.dump(symptom_list, f"{MODEL_DIR}/symptom_list.pkl")

with open(f"{MODEL_DIR}/symptom_list.json", "w", encoding="utf-8") as f:
    json.dump(symptom_list, f, indent=2, ensure_ascii=False)

print(f"   ✅ Saved symptom list ({len(symptom_list)} symptoms)")

# ─────────────────────────────────────────────────────────────
# STEP 4 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

disease_list = le.classes_.tolist()
with open(f"{MODEL_DIR}/disease_list.json", "w", encoding="utf-8") as f:
    json.dump(disease_list, f, indent=2, ensure_ascii=False)

print(f"   ✅ Saved label encoder ({len(disease_list)} diseases)")
# ─────────────────────────────────────────────────────────────
# REMOVE RARE DISEASES
# ─────────────────────────────────────────────────────────────
print("\n🧹 Removing rare diseases...")

MIN_SAMPLES_PER_CLASS = 5

disease_counts = y.value_counts()
valid_diseases = disease_counts[disease_counts >= MIN_SAMPLES_PER_CLASS].index

df_filtered = df[df["Disease"].isin(valid_diseases)].copy()

print(f"   After filtering:")
print(f"   Rows: {len(df_filtered)}")
print(f"   Diseases: {df_filtered['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────
# REBUILD MATRIX AFTER FILTERING
# ─────────────────────────────────────────────────────────────
symptom_lists = df_filtered['symptoms'].str.split(',').apply(
    lambda x: [s.strip() for s in x if s.strip()]
)

X = pd.DataFrame(0, index=df_filtered.index, columns=symptom_list)

for i, symptoms in zip(df_filtered.index, symptom_lists):
    X.loc[i, symptoms] = 1

y = df_filtered["Disease"]

# 🔥 IMPORTANT: Re-fit encoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n🏋️  Training models...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"   Training {name}...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
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
# STEP 7 — SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = trained_models[best_name]

print(f"\n✅ Best model: {best_name}")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

metadata = {
    "best_model": best_name,
    "results": results,
    "n_symptoms": len(symptom_list),
    "n_diseases": len(disease_list),
    "data_source": "BERT-enhanced medical dataset"
}

with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────
# STEP 8 — CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n📋 Classification Report (Best Model):")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("\n✅ Training complete. All artifacts saved in /model")
