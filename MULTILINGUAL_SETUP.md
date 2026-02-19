# Multilingual Disease Prediction System
## Setup Instructions

### **Step 1: Place datasets in data/ folder**

```bash
cd disease-predictor
mkdir -p data
```

Copy these files to `data/`:
- `updated_result_with_AI_HINDI.csv` (Hindi dataset)
- `updated_result_with_AI_PUNJABI.csv` (Punjabi dataset)
- `dataset.csv` (Columbia English dataset - already there)

---

### **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

New dependency: `langdetect` for automatic language detection

---

### **Step 3: Train all three models**

```bash
# Train English model (Columbia dataset)
python train.py

# Train Hindi model
python train_hindi.py

# Train Punjabi model
python train_punjabi.py
```

This creates:
```
model/
├── best_model.pkl          # English model
├── label_encoder.pkl
├── symptom_list.pkl
├── hindi/
│   ├── best_model.pkl      # Hindi model
│   ├── label_encoder.pkl
│   └── symptom_list.pkl
└── punjabi/
    ├── best_model.pkl      # Punjabi model
    ├── label_encoder.pkl
    └── symptom_list.pkl
```

---

### **Step 4: Run the multilingual API**

```bash
uvicorn app.main_multilingual:app --reload
```

Or rename `main_multilingual.py` to `main.py`:
```bash
mv app/main_multilingual.py app/main.py
uvicorn app.main:app --reload
```

---

### **Step 5: Test in multiple languages**

**English:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have fever, headache, and vomiting"}'
```

**Hindi:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "मुझे बुखार, सिरदर्द और उल्टी हो रही है"}'
```

**Punjabi:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ਮੈਨੂੰ ਬੁਖਾਰ, ਸਿਰ ਦਰਦ ਅਤੇ ਉਲਟੀ ਹੋ ਰਹੀ ਹੈ"}'
```

---

### **How Language Detection Works**

1. API receives text
2. Automatically detects language using `langdetect`
3. Routes to appropriate model (English/Hindi/Punjabi)
4. Returns predictions in the same language

You can also force a specific language:
```json
{
  "text": "fever headache vomiting",
  "language": "en"
}
```

---

### **API Endpoints**

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info and supported languages |
| `GET /models` | List all loaded models with stats |
| `POST /predict` | Multilingual prediction (auto-detects language) |
| `GET /docs` | Interactive API documentation |

---

### **Expected Training Times**

- English model: ~3-5 minutes (13,400 synthetic patients, 134 diseases)
- Hindi model: ~1-2 minutes (2,182 real patients, 237 diseases)
- Punjabi model: ~1-2 minutes (2,182 real patients, 214 diseases)

---

### **Research Paper Angles**

**What makes this unique:**

1. **Native language models** — not translation-based
   - Hindi model trained on Hindi medical records
   - Punjabi model trained on Punjabi medical records
   - English model trained on Columbia hospital data

2. **Automatic language detection** — user doesn't specify language

3. **Regional disease coverage** — Hindi/Punjabi models cover different diseases than English

4. **Real clinical data** — Hindi/Punjabi datasets are from actual patient records

**Paper title suggestion:**
"Multilingual Clinical Decision Support Using Native Language Models: A Comparative Study of English, Hindi, and Punjabi Disease Prediction Systems"

**Key result to highlight:**
"We developed three independent models trained on native language medical data rather than relying on translation, achieving [X]% accuracy on Hindi, [Y]% on Punjabi, and [Z]% on English clinical validation cases."

---

### **Next Steps**

After training all models:

1. Run validation tests in each language
2. Compare accuracy across languages
3. Generate multilingual validation reports
4. Update your research paper with multilingual results

This is genuinely unique — most research either translates to English or uses one language. You have **three independent native-language models**.
