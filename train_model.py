import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import joblib

Download required NLTK resources

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

Text cleaning function

def cleaner(report):
soup = BeautifulSoup(report, 'lxml')
text = soup.get_text()

text = re.sub(
    r"(@|http://|https://|www|\\x)\S*",
    " ",
    text
)

text = re.sub(
    "[^A-Za-z]+",
    " ",
    text
)

tokens = nltk.word_tokenize(text)

tokens = [token.lower() for token in tokens]

stop_words = set(stopwords.words('english'))

tokens = [
    token for token in tokens
    if token not in stop_words
]

lemmatizer = WordNetLemmatizer()

tokens = [
    lemmatizer.lemmatize(token)
    for token in tokens
]

return " ".join(tokens)

Load dataset

data_path = "./Data/Medical_reports(IBD-NonIBD).xlsx"

report = pd.read_excel(data_path)

report = report.dropna()

print("\nOriginal dataset:")
print(report['IBD'].value_counts())

Clean reports

report['Cleaned_Report'] = report['Report'].apply(cleaner)

report = report[
report['Cleaned_Report'].map(len) > 0
]

Prepare features and labels

data = report['Cleaned_Report']
Y = report['IBD']

print("\nClass distribution:")
print(Y.value_counts())

TF-IDF

tfidf = TfidfVectorizer(
min_df=0.00015,
ngram_range=(1, 3)
)

data_tfidf = tfidf.fit_transform(data)

joblib.dump(tfidf, 'tfidf.pkl')

Train/test split

X_train, X_test, y_train, y_test = train_test_split(
data_tfidf,
Y,
test_size=0.2,
random_state=42,
stratify=Y
)

print("\nTraining class distribution before SMOTE:")
print(y_train.value_counts())

Handle class imbalance

sm = SMOTE(random_state=42)

X_train_sm, y_train_sm = sm.fit_resample(
X_train,
y_train
)

print("\nTraining class distribution after SMOTE:")
print(y_train_sm.value_counts())

Train SVM

svm_clf = svm.SVC(
kernel='linear',
C=1
)

svm_clf.fit(
X_train_sm,
y_train_sm
)

Save model

joblib.dump(
svm_clf,
'svm_clf.pkl'
)

Evaluation

train_predictions = svm_clf.predict(X_train_sm)
test_predictions = svm_clf.predict(X_test)

print("\nModel classes:")
print(svm_clf.classes_)

print("\nTraining accuracy:")
print(round(
accuracy_score(y_train_sm, train_predictions),
4
))

print("\nTesting accuracy:")
print(round(
accuracy_score(y_test, test_predictions),
4
))

print("\nClassification report:")
print(
classification_report(
y_test,
test_predictions
)
)

print("\nConfusion matrix:")
print(
confusion_matrix(
y_test,
test_predictions
)
)

print("\nModel and TF-IDF vectorizer saved successfully.")
