# NABHAMODEL
Here’s a **professional and complete README** you can use for your GitHub repo **[README for NABHAMODEL](https://github.com/killerx1411/NABHAMODEL)** (multilingual disease prediction system):

---

# NABHAMODEL

**Multilingual Disease Prediction System** – A FastAPI-based API that predicts diseases from symptom text in **English, Hindi, and Punjabi** using native language models.

This project includes training scripts, language detection, and a ready API to serve predictions in multiple languages.

---

## 🚀 Features

✅ Predict diseases from free-text symptoms
✅ Works with **English, Hindi & Punjabi**
✅ Automatic language detection
✅ Modular training and validation scripts
✅ Model storage with language-specific folders
✅ Docker support

---

## 🛠️ Project Structure

```text
.
├── app/                       
├── data/                      # Place your datasets here
├── model/                     # Trained models saved here
├── train.py                   # Train English model
├── train_hindi.py             # Train Hindi model
├── train_punjabi.py           # Train Punjabi model
├── validate.py                # Validation/testing script
├── main_multilingual.py       # API entrypoint
├── requirements.txt
├── Dockerfile
└── MULTILINGUAL_SETUP.md      # Setup guide
```

---

## 📦 Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Data Preparation

Put the dataset CSV files under **data/**:

```
data/
├── dataset.csv                  # English dataset
├── updated_result_with_AI_HINDI.csv
└── updated_result_with_AI_PUNJABI.csv
```

---

## 🏋️‍♂️ Training Models

Train each language model:

English:

```bash
python train.py
```

Hindi:

```bash
python train_hindi.py
```

Punjabi:

```bash
python train_punjabi.py
```

Models will be saved under **model/**:

```
model/
├── best_model.pkl  
├── label_encoder.pkl
├── symptom_list.pkl
├── hindi/
└── punjabi/
```

---

## 🚀 Running the API

Start the API with automatic language detection:

```bash
uvicorn app.main:app --reload
```

Or rename entrypoint:

```bash
mv app/main_multilingual.py app/main.py
uvicorn app.main:app --reload
```

Access API docs:

```
http://localhost:8000/docs
```

---

## 🧪 Sample API Usage

#### English Prediction

```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"text":"I have fever, cough and headache"}'
```

#### Hindi Prediction

```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"text":"मुझे बुखार, खांसी और सिरदर्द है"}'
```

#### Punjabi Prediction

```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"text":"ਮੈਨੂੰ ਬੁਖਾਰ, ਖੰਘ ਅਤੇ ਸਿਰਦਰਦ ਹੈ"}'
```

---

## 📡 API Endpoints

| Endpoint   | Method | Description                       |
| ---------- | ------ | --------------------------------- |
| `/`        | GET    | API info & supported languages    |
| `/models`  | GET    | List loaded models                |
| `/predict` | POST   | Multi-language disease prediction |
| `/docs`    | GET    | Interactive API docs              |

---

## 🧠 How It Works

1. API receives symptom text
2. `langdetect` detects language
3. Routes to appropriate model (English/Hindi/Punjabi)
4. Returns prediction in same language

---

## 🧪 Testing

Run the validation script after training:

```bash
python validate.py
```

---

## 🐳 Docker (Optional)

Build Docker image:

```bash
docker build -t nabhamodel .
```

Run container:

```bash
docker run -p 8000:8000 nabhamodel
```

---

## 📌 Notes

✔ Hindi & Punjabi models are trained on **native language medical datasets**, not via translation.
✔ Automatic language detection means users need not specify language manually.

---

## 🙌 Contributions

Got ideas or improvements?
Feel free to open issues or pull requests.

---

If you want, I can also provide a **Markdown badge section** (build, coverage, license) to polish it further.
