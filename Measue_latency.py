# Inference latency — run this in your project
import time, joblib, numpy as np, pandas as pd

model        = joblib.load("model/best_model.pkl")
symptom_list = joblib.load("model/symptom_list.pkl")

vector = pd.DataFrame([np.zeros(len(symptom_list))], columns=symptom_list)

times = []
for _ in range(1000):
    t0 = time.perf_counter()
    model.predict_proba(vector)
    times.append((time.perf_counter() - t0) * 1000)

print(f"Median: {np.median(times):.2f} ms")
print(f"p95:    {np.percentile(times, 95):.2f} ms")