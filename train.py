"""
train.py — Training pipeline for Columbia Disease-Symptom Knowledge Base
Converts disease-symptom associations into synthetic patient cases for training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
DATA_PATH  = "data/dataset.csv"
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLES_PER_DISEASE = 100  # How many synthetic patient cases to generate per disease
MIN_SYMPTOMS = 3           # Minimum symptoms per patient
MAX_SYMPTOMS = 8           # Maximum symptoms per patient

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
print("📂 Loading Columbia Disease-Symptom Knowledge Base...")
df = pd.read_csv(DATA_PATH)

# Clean disease names and remove rows with missing symptoms
df["Disease"] = df["Disease"].str.strip().str.replace("_", " ").str.title()
df = df.dropna(subset=["Symptom"])  # Remove rows with NaN symptoms
df["Symptom"] = df["Symptom"].str.strip().str.replace("_", " ").str.lower()

print(f"   Raw KB: {len(df)} disease-symptom associations")
print(f"   Diseases: {df['Disease'].nunique()}")
print(f"   Symptoms: {df['Symptom'].nunique()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — BUILD DISEASE → SYMPTOM MAPPING
# ─────────────────────────────────────────────────────────────
print("\n🔧 Building disease-symptom mapping...")

# Group symptoms by disease
disease_symptoms = df.groupby("Disease")["Symptom"].apply(list).to_dict()

# Get all unique symptoms (guaranteed no NaN now)
all_symptoms = sorted(df["Symptom"].unique())
print(f"   Total unique symptoms: {len(all_symptoms)}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — SYNTHESIZE PATIENT CASES
# For each disease, generate synthetic patients with random
# subsets of that disease's symptoms
# ─────────────────────────────────────────────────────────────
print(f"\n🧬 Synthesizing {SAMPLES_PER_DISEASE} patient cases per disease...")

synthetic_patients = []

for disease, symptoms in disease_symptoms.items():
    # Skip diseases with too few symptoms
    if len(symptoms) < MIN_SYMPTOMS:
        print(f"   ⚠️  Skipping {disease} (only {len(symptoms)} symptoms)")
        continue
    
    for _ in range(SAMPLES_PER_DISEASE):
        # Randomly select 3-8 symptoms for this patient
        n_symptoms = np.random.randint(MIN_SYMPTOMS, min(MAX_SYMPTOMS, len(symptoms)) + 1)
        patient_symptoms = np.random.choice(symptoms, size=n_symptoms, replace=False)
        
        # Create binary vector
        patient_vector = {symptom: 1 if symptom in patient_symptoms else 0 
                          for symptom in all_symptoms}
        patient_vector["Disease"] = disease
        
        synthetic_patients.append(patient_vector)

# Convert to DataFrame
patient_df = pd.DataFrame(synthetic_patients)

print(f"   ✅ Generated {len(patient_df)} synthetic patient cases")
print(f"   ✅ Covering {patient_df['Disease'].nunique()} diseases")

# ─────────────────────────────────────────────────────────────
# STEP 4 — FEATURES & LABELS
# ─────────────────────────────────────────────────────────────
X = patient_df.drop("Disease", axis=1)
y = patient_df["Disease"]

symptom_list = X.columns.tolist()
joblib.dump(symptom_list, f"{MODEL_DIR}/symptom_list.pkl")

with open(f"{MODEL_DIR}/symptom_list.json", "w") as f:
    json.dump(symptom_list, f, indent=2)

print(f"\n   ✅ Saved symptom list ({len(symptom_list)} symptoms)")

# ─────────────────────────────────────────────────────────────
# STEP 5 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

disease_list = le.classes_.tolist()
with open(f"{MODEL_DIR}/disease_list.json", "w") as f:
    json.dump(disease_list, f, indent=2)

print(f"   ✅ Saved label encoder ({len(le.classes_)} diseases)")

# ─────────────────────────────────────────────────────────────
# STEP 6 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — TRAIN & COMPARE MODELS
# ─────────────────────────────────────────────────────────────
print("\n🏋️  Training models with 5-fold cross-validation...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
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
# STEP 8 — PICK BEST MODEL
# ─────────────────────────────────────────────────────────────
print("\n📊 Model Comparison:")
print(f"{'Model':<25} {'CV Accuracy':<15} {'Std Dev':<12} {'Test Accuracy'}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<25} {r['cv_mean']:<15} {r['cv_std']:<12} {r['test_acc']}")

best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = trained_models[best_name]

print(f"\n✅ Best model: {best_name}")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

# Save model metadata
metadata = {
    "best_model": best_name,
    "results": results,
    "n_symptoms": len(symptom_list),
    "n_diseases": len(disease_list),
    "data_source": "Columbia University NYPH Disease-Symptom Knowledge Base",
    "synthetic_samples_per_disease": SAMPLES_PER_DISEASE,
    "total_synthetic_patients": len(patient_df)
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ─────────────────────────────────────────────────────────────
# STEP 9 — CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n📋 Classification Report (Best Model on Test Set):")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("\n✅ All artifacts saved to /model")
print("   - best_model.pkl")
print("   - label_encoder.pkl")
print("   - symptom_list.pkl")
print("   - symptom_list.json")
print("   - disease_list.json")
print("   - metadata.json")
print("\n⚠️  NOTE: This model was trained on SYNTHETIC patient cases")
print("   generated from the Columbia disease-symptom knowledge base.")
print("   100 cases per disease were synthesized by randomly sampling")
print("   3-8 symptoms from each disease's known symptom list.")
print("\n🚀 Ready. Run: uvicorn app.main:app --reload")