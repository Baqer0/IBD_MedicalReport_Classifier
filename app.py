from flask import Flask, render_template, request, jsonify
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
import os

# Load the saved SVM model and TF-IDF vectorizer
svm_clf = joblib.load('./svm_clf.pkl')  # Load the SVM model
tfidf = joblib.load('./tfidf.pkl')      # Load the TF-IDF vectorizer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None  # Default value for result to handle GET requests
    
    if request.method == 'POST':
        report = request.files['report']
        if report.filename == '':
            return jsonify({'error': 'No selected file'})

        # Ensure the report directory exists
        if not os.path.exists('./report/'):
            os.makedirs('./report/')

        report_path = "./report/" + report.filename
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
        print("Extracted text:", text)  # Debugging info
        print("Cleaned text:", cleaned_text)  # Debugging info

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        transformed_text = tfidf.transform([cleaned_text])

        # Predict using the SVM model
        prediction = svm_clf.predict(transformed_text)
        print(prediction)
        
        # Return the result based on the SVM model's prediction
        result = 'IBD' if prediction[0] == 'yes' else 'Non-IBD'
    
    return render_template('index.html', prediction=result)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from image using OCR
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to clean the text data
def cleaner(report):
    soup = BeautifulSoup(report, 'lxml')  # Remove HTML entities
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", souped)  # Remove @mentions, URLs, etc.
    re2 = re.sub("[^A-Za-z]+", " ", re1)  # Remove non-alphabetic characters

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return " ".join(lemmas)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
