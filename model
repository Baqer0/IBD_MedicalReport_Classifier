import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from imblearn.over_sampling import SMOTE
import joblib

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#------------------------------------------------------------------------------------------------------
# Function to clean the medical reports
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

#------------------------------------------------------------------------------------------------------
# Load the dataset
data_path = "./Data/Medical_reports(IBD-NonIBD).xlsx"  # Use relative path
report = pd.read_excel(data_path)

# Drop missing values
report = report.dropna()

# Clean the 'Report' column
report['Cleaned_Report'] = report['Report'].apply(cleaner)
# Remove rows with cleaned reports of length 0
report = report[report['Cleaned_Report'].map(len) > 0]

# Remove the original 'Report' column as it's no longer needed
report.drop(['Report'], axis=1, inplace=True)

#------------------------------------------------------------------------------------------------------
# Prepare the dataset for model training
data = report['Cleaned_Report']  # Features
Y = report['IBD']  # Target column

# TF-IDF Vectorization
tfidf = TfidfVectorizer(min_df=0.00015, ngram_range=(1, 3))
tfidf.fit(data)  # Fit the vectorizer on the dataset
data_tfidf = tfidf.transform(data)  # Transform the text into TF-IDF values

# Save TF-IDF vectorizer for later use
joblib.dump(tfidf, 'tfidf.pkl')

#------------------------------------------------------------------------------------------------------
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_tfidf, Y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

#------------------------------------------------------------------------------------------------------
# Train a Naive Bayes classifier
svm_clf = svm.SVC(kernel='linear', C=1)

# Fit the model on the oversampled training data
svm_clf.fit(X_train_sm, y_train_sm)

# Save the trained model for later use
joblib.dump(svm_clf, 'svm_clf.pkl')

#------------------------------------------------------------------------------------------------------
# Evaluate the model performance
train_accuracy = svm_clf.score(X_train_sm, y_train_sm)
test_accuracy = svm_clf.score(X_test, y_test)

# Print training and testing accuracy
print("Training accuracy:", round(train_accuracy, 2))
print("Testing accuracy:", round(test_accuracy, 2))
