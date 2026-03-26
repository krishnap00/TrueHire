import pickle
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Load model + vectorizer
tfidf_model = pickle.load(open("tfidf_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------- Detection helpers ---------- #

def detect_fake_company(text):
    words = ["unknown company", "confidential company", "startup soon"]
    return int(any(w in text.lower() for w in words))

def detect_suspicious_email(text):
    emails = re.findall(r'\S+@\S+', text)
    for email in emails:
        if any(d in email for d in ["gmail.com", "yahoo.com", "outlook.com"]):
            return 1
    return 0

def detect_high_salary(text):
    salaries = re.findall(r'\$?\d+', text)
    for s in salaries:
        try:
            if int(s.replace("$", "")) > 10000:
                return 1
        except:
            continue
    return 0

def detect_weird_format(text):
    if text.count("!!!") > 0 or text.isupper():
        return 1
    return 0

def detect_mismatch(text):
    text = text.lower()
    tech_roles = ["software engineer", "developer", "data scientist"]
    non_tech = ["history", "arts", "biology", "commerce"]

    if any(r in text for r in tech_roles) and any(d in text for d in non_tech):
        return 1
    return 0

# ---------- MAIN FUNCTION ---------- #

def predict_job(text):
    cleaned = preprocess(text)

    tfidf_vec = tfidf_vectorizer.transform([cleaned])

    text_length = len(text)
    num_exclamations = text.count("!")
    has_money = int(any(w in text.lower() for w in ["salary", "$", "earn", "income"]))
    has_urgent = int(any(w in text.lower() for w in ["urgent", "immediate"]))

    extra = np.array([[text_length, num_exclamations, has_money, has_urgent]])
    tfidf_vec = hstack((tfidf_vec, extra))

    prob_fake = tfidf_model.predict_proba(tfidf_vec)[0][1]

    # Boost logic
    if detect_fake_company(text):
        prob_fake += 0.05
    if detect_suspicious_email(text):
        prob_fake += 0.07
    if detect_high_salary(text):
        prob_fake += 0.08
    if detect_weird_format(text):
        prob_fake += 0.05
    if detect_mismatch(text):
        prob_fake += 0.08

    prob_fake = min(prob_fake, 1.0)

    fake = prob_fake * 100
    real = 100 - fake

    label = "FAKE JOB" if fake > 50 else "REAL JOB"

    return real, fake, label