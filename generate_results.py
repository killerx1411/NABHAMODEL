"""
generate_results.py
Creates tables + graphs for multilingual disease prediction results
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LANGUAGES = {
    "English": "model",
    "Hindi": "model/hindi",
    "Punjabi": "model/punjabi"
}

# ----------------------------------------------------
# FUNCTION: Load Model + Metadata
# ----------------------------------------------------
def load_language_model(path):
    model = joblib.load(os.path.join(path, "best_model.pkl"))
    le = joblib.load(os.path.join(path, "label_encoder.pkl"))
    symptom_list = joblib.load(os.path.join(path, "symptom_list.pkl"))
    metadata = json.load(open(os.path.join(path, "metadata.json"), encoding="utf-8"))
    return model, le, symptom_list, metadata


# ----------------------------------------------------
# 1️⃣ CREATE PERFORMANCE TABLE
# ----------------------------------------------------
rows = []

for lang, path in LANGUAGES.items():
    if not os.path.exists(path):
        continue

    model, le, symptom_list, metadata = load_language_model(path)

    rows.append({
        "Language": lang,
        "Best Model": metadata["best_model"],
        "Diseases": metadata["n_diseases"],
        "Symptoms": metadata["n_symptoms"],
        "CV Accuracy": metadata["cv_mean"],
        "CV Std": metadata["cv_std"],
        "Test Accuracy": metadata["test_accuracy"]
    })

df_results = pd.DataFrame(rows)
df_results.to_csv(f"{RESULTS_DIR}/model_performance_table.csv", index=False)

print("\n📊 Saved performance table.")

# ----------------------------------------------------
# 2️⃣ BAR CHART — Accuracy Comparison
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.bar(df_results["Language"], df_results["Test Accuracy"])
plt.title("Test Accuracy Across Languages")
plt.ylabel("Accuracy")
plt.xlabel("Language")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png")
plt.close()

print("📈 Saved accuracy comparison graph.")

# ----------------------------------------------------
# 3️⃣ CV STABILITY PLOT
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.bar(df_results["Language"], df_results["CV Std"])
plt.title("Cross-Validation Standard Deviation")
plt.ylabel("CV Std")
plt.xlabel("Language")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cv_stability.png")
plt.close()

print("📉 Saved CV stability graph.")

# ----------------------------------------------------
# 4️⃣ CONFUSION MATRIX (English example)
# ----------------------------------------------------
english_path = LANGUAGES["English"]
model, le, symptom_list, metadata = load_language_model(english_path)

# Load training data again (modify path if needed)
df = pd.read_csv("data/updated_result_with_BERT.csv")
df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].dropna()
df.columns = ["Disease", "symptoms"]

# Build feature matrix same way as train.py
symptom_lists = df['symptoms'].str.split(',').apply(
    lambda x: [s.strip() for s in x if s.strip()]
)

all_symptoms = symptom_list
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i, symptoms in enumerate(symptom_lists):
    for s in symptoms:
        if s in X.columns:
            X.loc[i, s] = 1

y = le.transform(df["Disease"])

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix (English Model)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_english.png")
plt.close()

print("🔥 Saved confusion matrix.")

# ----------------------------------------------------
# 5️⃣ CLASSIFICATION REPORT TABLE
# ----------------------------------------------------
report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f"{RESULTS_DIR}/classification_report_english.csv")

print("📋 Saved classification report.")

print("\n✅ ALL RESULTS GENERATED IN /results FOLDER")