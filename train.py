"""
train.py — Full production training pipeline
Adapted for Columbia Disease-Symptom Knowledge Base
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
DATA_PATH  = "data/dataset.csv"
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION FOR SYNTHETIC PATIENT GENERATION
# ─────────────────────────────────────────────────────────────
SAMPLES_PER_DISEASE = 100  # How many synthetic patient cases to generate per disease
MIN_SYMPTOMS = 3           # Minimum symptoms per patient
MAX_SYMPTOMS = 8           # Maximum symptoms per patient

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN COLUMBIA KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
print("📂 Loading Columbia Disease-Symptom Knowledge Base...")
kb_df = pd.read_csv(DATA_PATH)

# Remove rows with missing symptoms
kb_df = kb_df.dropna(subset=["Symptom"])

# Clean disease names and symptoms
kb_df["Disease"] = kb_df["Disease"].str.strip().str.replace("_", " ").str.title()
kb_df["Symptom"] = kb_df["Symptom"].str.strip().str.replace("_", " ").str.lower()

print(f"   Knowledge Base: {len(kb_df)} disease-symptom associations")
print(f"   Diseases: {kb_df['Disease'].nunique()} | Symptoms: {kb_df['Symptom'].nunique()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — SYNTHESIZE PATIENT CASES FROM KNOWLEDGE BASE
# Columbia data is disease→symptom mappings, not patient cases
# We generate synthetic patients by randomly sampling symptoms per disease
# ─────────────────────────────────────────────────────────────
print(f"\n🧬 Synthesizing {SAMPLES_PER_DISEASE} patient cases per disease...")

# Group symptoms by disease
disease_symptoms = kb_df.groupby("Disease")["Symptom"].apply(list).to_dict()
all_symptoms = sorted(kb_df["Symptom"].unique())

# Generate synthetic patients
synthetic_patients = []
for disease, symptoms in disease_symptoms.items():
    if len(symptoms) < MIN_SYMPTOMS:
        print(f"   ⚠️  Skipping {disease} (only {len(symptoms)} symptoms)")
        continue
    
    for _ in range(SAMPLES_PER_DISEASE):
        # Randomly select 3-8 symptoms for this patient
        n_symptoms = np.random.randint(MIN_SYMPTOMS, min(MAX_SYMPTOMS, len(symptoms)) + 1)
        patient_symptoms = np.random.choice(symptoms, size=n_symptoms, replace=False)
        
        # Create patient row
        patient_row = {"Disease": disease}
        for i, symptom in enumerate(patient_symptoms, 1):
            patient_row[f"Symptom_{i}"] = symptom
        
        synthetic_patients.append(patient_row)

df = pd.DataFrame(synthetic_patients)
print(f"   ✅ Generated {len(df)} synthetic patients")
print(f"   ✅ Covering {df['Disease'].nunique()} diseases")

# ─────────────────────────────────────────────────────────────
# STEP 3 — BUILD BINARY SYMPTOM MATRIX
# ─────────────────────────────────────────────────────────────
print("\n🔧 Building binary symptom matrix...")

symptom_cols = [col for col in df.columns if col.startswith("Symptom")]

df = df.reset_index().rename(columns={"index": "PatientID"})

melted = df.melt(
    id_vars=["PatientID", "Disease"],
    value_vars=symptom_cols,
    value_name="Symptom"
)
melted = melted.dropna(subset=["Symptom"])
melted["Symptom"] = melted["Symptom"].str.strip()
melted["Present"] = 1

binary_df = melted.pivot_table(
    index="PatientID",
    columns="Symptom",
    values="Present",
    fill_value=0
)

binary_df["Disease"] = df.set_index("PatientID")["Disease"]

print(f"   Matrix shape: {binary_df.shape}")
print(f"   Symptoms: {binary_df.shape[1] - 1}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — FEATURES & LABELS
# ─────────────────────────────────────────────────────────────
X = binary_df.drop("Disease", axis=1)
y = binary_df["Disease"]

# Save the exact symptom column order — API must use this exact order
symptom_list = X.columns.tolist()
joblib.dump(symptom_list, f"{MODEL_DIR}/symptom_list.pkl")

# Also export as JSON for any frontend that needs it
with open(f"{MODEL_DIR}/symptom_list.json", "w") as f:
    json.dump(symptom_list, f, indent=2)

print(f"   ✅ Saved symptom list ({len(symptom_list)} symptoms)")

# ─────────────────────────────────────────────────────────────
# STEP 5 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

# Export disease list as JSON
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
# Cross-validated so we don't just trust one split
# ─────────────────────────────────────────────────────────────
print("\n🏋️  Training models with 5-fold cross-validation...\n")

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
        n_estimators=100,
        learning_rate=0.15,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.6,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
   "Gradient Boosting": HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.15,
    max_depth=4,
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
print("\n⚠️  NOTE: Model trained on synthetic patient cases generated from")
print("   Columbia NYPH hospital discharge summaries (3,363 records).")
print("\n🚀 Ready. Run: uvicorn app.main:app --reload")