from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import re
import nltk
import joblib
import pdfplumber
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up paths for model and vectorizer
MODEL_PATH = './svm_clf.pkl'
TFIDF_PATH = './tfidf.pkl'

# Load the saved SVM model and TF-IDF vectorizer
svm_clf = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

# Set Tesseract binary path if necessary (adjust for Render)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None  # Default value for result to handle GET requests

    if request.method == 'POST':
        try:
            report = request.files['report']
            if not report:
                return jsonify({'error': 'No file uploaded'})

            if not report.filename:
                return jsonify({'error': 'No file selected'})

            # Save file securely
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], report.filename)
            report.save(report_path)

            # Extract text based on file type
            if report.filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(report_path)
            elif report.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = extract_text_from_image(report_path)
            else:
                return jsonify({'error': 'Invalid file type'})

            # Clean and transform the text
            cleaned_text = cleaner(text)
            app.logger.debug(f"Extracted text: {cleaned_text}")

            # Transform the cleaned text using the TF-IDF vectorizer
            transformed_text = tfidf.transform([cleaned_text])

            # Predict using the SVM model
            prediction = svm_clf.predict(transformed_text)
            result = 'IBD' if prediction[0] == 'yes' else 'Non-IBD'
        
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during processing: {str(e)}'})

    return render_template('index.html', prediction=result)

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {str(e)}")
    return text

def extract_text_from_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        app.logger.error(f"Error extracting text from image: {str(e)}")
        text = ""
    return text

def cleaner(report):
    soup = BeautifulSoup(report, 'lxml')
    text = soup.get_text()
    text = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
