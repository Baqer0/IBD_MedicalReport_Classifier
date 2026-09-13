import os
import re
import joblib
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Medical_reports(IBD-NonIBD).xlsx')
NLTK_DIR = os.path.join(BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DIR)

for resource, path in [('punkt','tokenizers/punkt'),('punkt_tab','tokenizers/punkt_tab'),('stopwords','corpora/stopwords'),('wordnet','corpora/wordnet')]:
    try: nltk.data.find(path)
    except LookupError: nltk.download(resource, download_dir=NLTK_DIR, quiet=True)

STOP_WORDS = set(stopwords.words('english')) - {'no','not','nor','never','without','against','before','after'}
LEMMATIZER = WordNetLemmatizer()

def cleaner(text):
    text = BeautifulSoup(str(text), 'lxml').get_text(' ')
    text = re.sub(r'(@|https?://|www\.|\\x)\S*', ' ', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'Training data not found: {DATA_PATH}')
    report = pd.read_excel(DATA_PATH)
    missing = {'Report','IBD'} - set(report.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    report = report[['Report','IBD']].dropna()
    report['Cleaned_Report'] = report['Report'].astype(str).apply(cleaner)
    report = report[report['Cleaned_Report'].str.len() > 0]
    print('\nClass distribution:')
    print(report['IBD'].value_counts())
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        report['Cleaned_Report'], report['IBD'], test_size=0.20, random_state=42, stratify=report['IBD'])
    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1,2), sublinear_tf=True, max_features=50000)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('\nModel classes:', clf.classes_)
    print('Testing accuracy:', round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y_test, pred))
    joblib.dump(tfidf, os.path.join(BASE_DIR, 'tfidf.pkl'), compress=3)
    joblib.dump(clf, os.path.join(BASE_DIR, 'svm_clf.pkl'), compress=3)
    print('\nSaved tfidf.pkl and svm_clf.pkl')

if __name__ == '__main__': main()
