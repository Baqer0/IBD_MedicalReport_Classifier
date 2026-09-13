from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, re, nltk, joblib, pdfplumber
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)
for resource, path in [('punkt','tokenizers/punkt'),('punkt_tab','tokenizers/punkt_tab'),('stopwords','corpora/stopwords'),('wordnet','corpora/wordnet')]:
    try: nltk.data.find(path)
    except LookupError: nltk.download(resource, download_dir=NLTK_DIR, quiet=True)
STOP_WORDS = set(stopwords.words('english')) - {'no','not','nor','never','without','against','before','after'}
LEMMATIZER = WordNetLemmatizer()
app = Flask(__name__)
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = os.path.join(BASE_DIR, 'svm_clf.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf.pkl')
if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
    raise FileNotFoundError('svm_clf.pkl or tfidf.pkl is missing. Run train_model.py first.')
svm_clf = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')

def extract_text_from_pdf(path):
    parts=[]
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t=page.extract_text()
                if t: parts.append(t)
    except Exception as e: app.logger.error(f'PDF extraction error: {e}')
    return '\n'.join(parts)

def extract_text_from_image(path):
    try: return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        app.logger.error(f'Image extraction error: {e}')
        return ''

def cleaner(text):
    text=BeautifulSoup(str(text),'lxml').get_text(' ')
    text=re.sub(r'(@|https?://|www\.|\\x)\S*',' ',text)
    text=re.sub(r'[^A-Za-z]+',' ',text)
    tokens=nltk.word_tokenize(text.lower())
    return ' '.join(LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t)>1)

@app.route('/health')
def health():
    return jsonify({'status':'ok','model_classes':[str(x) for x in svm_clf.classes_]})

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        save_path=None
        try:
            report=request.files.get('report')
            if not report or not report.filename: return jsonify({'error':'No file uploaded'}),400
            filename=secure_filename(report.filename).lower()
            if not filename.endswith(('.pdf','.png','.jpg','.jpeg')): return jsonify({'error':'Unsupported file type. Please upload a PDF or image.'}),400
            save_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            report.save(save_path)
            text=extract_text_from_pdf(save_path) if filename.endswith('.pdf') else extract_text_from_image(save_path)
            if not text.strip(): return jsonify({'error':'No readable text could be extracted from the uploaded file.'}),400
            cleaned=cleaner(text)
            if not cleaned.strip(): return jsonify({'error':'No usable text remained after preprocessing.'}),400
            transformed=tfidf.transform([cleaned])
            prediction=svm_clf.predict(transformed)[0]
            probabilities={}
            if hasattr(svm_clf,'predict_proba'):
                probs=svm_clf.predict_proba(transformed)[0]
                probabilities={str(c):round(float(p),4) for c,p in zip(svm_clf.classes_,probs)}
            app.logger.info(f'Prediction={prediction}; probabilities={probabilities}; classes={svm_clf.classes_}')
            return render_template('index.html', prediction=str(prediction), class_probabilities=probabilities, confidence=max(probabilities.values()) if probabilities else None)
        except Exception as e:
            app.logger.error(f'Prediction error: {e}', exc_info=True)
            return jsonify({'error':f'Error during processing: {e}'}),500
        finally:
            if save_path and os.path.exists(save_path):
                try: os.remove(save_path)
                except Exception: pass
    return render_template('index.html', prediction=None, class_probabilities={}, confidence=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT',5000)), debug=False)
