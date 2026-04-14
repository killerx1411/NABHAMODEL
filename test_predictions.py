"""
test_predictions.py
Run after training: python test_predictions.py
Verifies the model predicts different diseases for different symptom profiles.
"""
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "model"
model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
symptom_list = joblib.load(f"{MODEL_DIR}/symptom_list.pkl")


def predict(symptoms):
    vec = np.zeros(len(symptom_list))
    for s in symptoms:
        if s in symptom_list:
            vec[symptom_list.index(s)] = 1
    df = pd.DataFrame([vec], columns=symptom_list)
    probs = model.predict_proba(df)[0]
    top3 = np.argsort(probs)[::-1][:3]
    return [(le.inverse_transform([i])[0], round(float(probs[i]) * 100, 1)) for i in top3]


# These symptom sets are completely disjoint between classes —
# if the model can't predict different diseases for these, the bias is still present.
test_cases = {
    "AVN patient": ["pain in hip area", "hip stiffness", "aching in buttocks",
                    "discomfort in groin", "restricted hip movement"],
    "OA patient": ["joint stiffness", "aching joints", "inflammation around joints",
                   "swelling in joints", "weakness in muscles"],
    "Fracture patient": ["tenderness in bones", "decreased bone density", "weak bones",
                         "signs of fractures", "stiffness in bones"],
    "Other (spine)": ["lower back pain", "nerve compression", "spinal pain",
                      "lumbar stiffness"],
}

print("\n🔍 Prediction Sanity Check — each case should predict a DIFFERENT disease")
print("=" * 65)
predicted_top = []
for label, symptoms in test_cases.items():
    results = predict(symptoms)
    top = results[0][0]
    predicted_top.append(top)
    print(f"\n  {label}")
    for rank, (disease, conf) in enumerate(results, 1):
        marker = " ◀ TOP" if rank == 1 else ""
        print(f"    {rank}. {disease:<35} {conf:>6.1f}%{marker}")

print(f"\n{'=' * 65}")
unique_preds = set(predicted_top)
if len(unique_preds) == 4:
    print(" PASS — All 4 cases predicted different diseases. Bias is fixed.")
elif len(unique_preds) == 3:
    print(f"⚠️  PARTIAL — Only 3 distinct predictions: {unique_preds}")
    print("   Re-check the undersampling step and class_weight settings.")
else:
    print(f"❌ FAIL — Only {len(unique_preds)} distinct predictions: {unique_preds}")
    print("   AVN bias is still present. Increase undersampling or reduce avn_cap.")
