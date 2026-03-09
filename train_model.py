# ------------------------------------
# 1. Import Libraries
# ------------------------------------
import pandas as pd
import numpy as np
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle

nltk.download('stopwords')

# ------------------------------------
# 2. Load Dataset
# ------------------------------------
df = pd.read_csv("fake_job_postings.csv")

print("Dataset shape:", df.shape)

# ------------------------------------
# 3. Data Cleaning
# ------------------------------------
df = df.fillna("")

df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

X = df["text"]
y = df["fraudulent"]

# ------------------------------------
# 4. Text Preprocessing
# ------------------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

X = X.apply(preprocess)

# ------------------------------------
# 5. Convert Text → Numbers (TF-IDF)
# ------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X_vector = vectorizer.fit_transform(X)

# ------------------------------------
# 6. Train Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# ------------------------------------
# 7. Train Random Forest Model
# ------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------
# 8. Model Evaluation
# ------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------
# 9. Test Custom Job Post
# ------------------------------------
def predict_job(text):

    text = preprocess(text)

    vector = vectorizer.transform([text])

    result = model.predict(vector)

    if result[0] == 1:
        print("⚠️ Fake Job Post")
    else:
        print("✅ Real Job Post")

# Example
test_job = """
Earn $3000 weekly working from home.
No experience needed.
Apply immediately.
"""

predict_job(test_job)

# ------------------------------------
# 10. Save Model
# ------------------------------------
pickle.dump(model, open("rf_fake_job_model.pkl", "wb"))
pickle.dump(vectorizer, open("rf_vectorizer.pkl", "wb"))

print("\nRandom Forest Model Saved")