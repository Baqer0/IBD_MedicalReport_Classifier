# IBD Medical Report Classifier

A Flask web application that classifies medical reports as **IBD (Inflammatory Bowel Disease)** or **Non-IBD** using a trained Support Vector Machine (SVM) model.

Built as part of the MyCCAngel project — an AI-powered healthcare platform for IBD patients. The classifier gates platform access by verifying that new users have a genuine IBD diagnosis before they can join the community.

---

## How It Works

1. User uploads a medical report (PDF or image)
2. Text is extracted via `pdfplumber` (PDFs) or `pytesseract` OCR (images)
3. Text is cleaned and vectorized using a TF-IDF vectorizer
4. An SVM classifier predicts IBD or Non-IBD
5. Result is displayed on screen

---

## Model

- **Algorithm:** Support Vector Machine (SVM, linear kernel)
- **Vectorizer:** TF-IDF (unigrams, bigrams, trigrams)
- **Dataset:** 200 manually collected and labeled medical reports
- **Test accuracy:** ~91% (held-out test split, 80/20)
- **Class imbalance:** handled with SMOTE oversampling during training

---

## Project Structure

```
├── app.py                  # Flask application
├── train_model.py          # Model training script
├── svm_clf.pkl             # Trained SVM classifier
├── tfidf.pkl               # Fitted TF-IDF vectorizer
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── templates/
│   └── index.html          # Web interface
└── uploads/                # Temporary upload storage (auto-created)
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ibd-classifier.git
cd ibd-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Tesseract (for image OCR)
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 4. Run the app
python app.py

# 5. Open in browser
# http://localhost:5000
```

---

## Deploying to Render

The `render.yaml` file is pre-configured. Connect the repo to Render and it will:
- Install Python dependencies
- Install Tesseract OCR as a system package
- Start the app with gunicorn

---

## Retraining the Model

To retrain on new data, place your dataset at `Data/Medical_reports(IBD-NonIBD).xlsx` and run:

```bash
python train_model.py
```

This will overwrite `svm_clf.pkl` and `tfidf.pkl`.
