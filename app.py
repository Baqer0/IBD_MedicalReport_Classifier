from flask import Flask, render_template, request, jsonify
import os
import re
import nltk
import joblib
import pdfplumber
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

for resource, path in [
    ('punkt',     'tokenizers/punkt'),
    ('stopwords', 'corpora/stopwords'),
    ('wordnet',   'corpora/wordnet'),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Load model & vectorizer ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
svm_clf = joblib.load(os.path.join(BASE_DIR, 'svm_clf.pkl'))
tfidf   = joblib.load(os.path.join(BASE_DIR, 'tfidf.pkl'))

# ── Tesseract path ────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        app.logger.error(f"PDF extraction error: {e}")
    return text

def extract_text_from_image(file_path):
    try:
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        app.logger.error(f"Image extraction error: {e}")
        return ""

# ── Text cleaning ─────────────────────────────────────────────────────────────
def cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    text = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None

    if request.method == 'POST':
        try:
            report = request.files.get('report')
            if not report or not report.filename:
                return jsonify({'error': 'No file uploaded'})

            filename = report.filename.lower()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], report.filename)
            report.save(save_path)

            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(save_path)
            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                text = extract_text_from_image(save_path)
            else:
                return jsonify({'error': 'Unsupported file type. Please upload a PDF or image.'})

            cleaned = cleaner(text)
            transformed = tfidf.transform([cleaned])
            prediction = svm_clf.predict(transformed)
            result = 'IBD' if prediction[0] == 'yes' else 'Non-IBD'

        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return jsonify({'error': f'Error during processing: {str(e)}'})

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
