# 🫀 Heart Disease Predictor

> Predict your risk of heart disease instantly — enter clinical parameters and get a real-time prediction powered by a K-Nearest Neighbours model with StandardScaler preprocessing, deployed on Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KNN-orange?style=flat)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat&logo=streamlit)](https://streamlit.io)

---

## 🔍 What Is This?

**Heart Disease Predictor** is a clinical risk assessment web app. A user enters their medical parameters — age, blood pressure, cholesterol, ECG results, chest pain type, and more — and the app instantly predicts whether they are at **high risk** or **low risk** of heart disease using a trained KNN classifier.

---

## ✨ Features

- **11 Clinical Inputs** — Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak, ST Slope
- **KNN Classifier** — Trained K-Nearest Neighbours model serialized with joblib
- **StandardScaler Preprocessing** — Input features scaled at prediction time using the same scaler fitted during training
- **One-Hot Encoding Alignment** — Categorical inputs are one-hot encoded and aligned to the exact column order used during training via `columns.pkl`
- **Instant Result** — High Risk ⚠️ or Low Risk ✅ displayed immediately on prediction
- **Interactive UI** — Sliders, dropdowns, and number inputs for all clinical fields

---

## 🏗️ How It Works

```
User Input (11 clinical features)
    ↓
One-Hot Encode categorical fields
(Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
    ↓
Align columns to training schema
(columns.pkl — fills missing dummies with 0)
    ↓
StandardScaler (scaler.pkl)
    ↓
KNN Classifier (Knn_heart_model.pkl)
    ↓
Prediction: High Risk ⚠️ / Low Risk ✅
```

---

## 🩺 Input Features

| Feature | Type | Range / Options |
|---|---|---|
| Age | Slider | 18 – 100 |
| Sex | Dropdown | M / F |
| Chest Pain Type | Dropdown | ATA / NAP / TA / ASY |
| Resting Blood Pressure | Number | 80 – 200 mm Hg |
| Cholesterol | Number | 100 – 600 mg/dL |
| Fasting Blood Sugar > 120 mg/dL | Dropdown | 0 / 1 |
| Resting ECG | Dropdown | Normal / ST / LVH |
| Max Heart Rate | Slider | 60 – 220 |
| Exercise-Induced Angina | Dropdown | Y / N |
| Oldpeak (ST Depression) | Slider | 0.0 – 6.0 |
| ST Slope | Dropdown | Up / Flat / Down |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model | K-Nearest Neighbours (`KNeighborsClassifier`) |
| Preprocessing | `StandardScaler` |
| Column Alignment | `columns.pkl` (joblib) |
| Model Serialization | Joblib |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
Heart_disease/
├── app.py                  # Streamlit web application
├── Knn_heart_model.pkl     # Trained KNN classifier
├── scaler.pkl              # Fitted StandardScaler
├── columns.pkl             # Expected column order from training
└── requirements.txt
```

---

## 🚀 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/harshit8204/Heart_disease.git
cd Heart_disease
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit scikit-learn pandas joblib
```

### 4. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚙️ Key Implementation Detail

Categorical inputs are manually one-hot encoded at prediction time and aligned to the exact column schema used during training:

```python
raw_input = {
    'Age': age,
    'Sex_' + sex: 1,
    'ChestPainType_' + chest_pain: 1,
    ...
}
input_df = pd.DataFrame([raw_input])

# Fill missing dummy columns with 0
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
```

This ensures the model always receives the same number and order of features it was trained on — a common source of bugs in deployed ML apps.

---

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
joblib
```

---

## 👤 Author

**Harshit Pal** — [@harshit8204](https://github.com/harshit8204)
