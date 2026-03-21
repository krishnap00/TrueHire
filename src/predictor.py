import pickle
from pydoc import text
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load model + vectorizer
# ----------------------------
model = pickle.load(open("tfidf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ----------------------------
# Preprocessing
# ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
# ----------------------------
# Domain Mismatch Detection
# ----------------------------
def detect_mismatch(text):
    text = text.lower()

    tech_roles = [
        "software engineer", "developer", "data scientist",
        "backend engineer", "frontend developer", "ai engineer"
    ]

    non_tech_degrees = [
        "bsc maths", "history", "arts", "biology",
        "commerce", "ba", "bcom"
    ]

    strong_tech_degrees = [
        "computer science", "btech", "b.e", "engineering", "it"
    ]

    role_flag = any(role in text for role in tech_roles)
    weak_degree_flag = any(deg in text for deg in non_tech_degrees)
    strong_degree_flag = any(deg in text for deg in strong_tech_degrees)

    if role_flag and weak_degree_flag and not strong_degree_flag:
        return 1

    return 0
# ----------------------------
# Fake Company Detection
# ----------------------------
def detect_fake_company(text):
    text = text.lower()

    famous_companies = [
        "google", "microsoft", "amazon", "infosys",
        "tcs", "wipro", "meta", "apple"
    ]

    suspicious_roles = [
        "data entry", "typing job", "form filling",
        "no experience", "work from home"
    ]

    company_flag = any(comp in text for comp in famous_companies)
    role_flag = any(role in text for role in suspicious_roles)

    if company_flag and role_flag:
        return 1

    return 0
# ----------------------------
# Prediction Function
# ----------------------------
def predict_job(text):

    cleaned = preprocess(text)

    tfidf_vec = vectorizer.transform([cleaned])

    text_length = len(text)
    num_exclamations = text.count("!")
    has_money_words = int(any(word in text.lower() for word in ["salary", "$", "earn", "income", "paid"]))
    has_urgent_words = int(any(word in text.lower() for word in ["urgent", "immediate", "limited", "hurry"]))

    extra_features = np.array([[text_length, num_exclamations, has_money_words, has_urgent_words]])

    tfidf_vec = hstack((tfidf_vec, extra_features))

    prob_fake = model.predict_proba(tfidf_vec)[0][1]
    fake_company = detect_fake_company(text)

    if fake_company:
        prob_fake = min(prob_fake + 0.30, 1.0)
    # 🔥 Apply mismatch logic
    mismatch = detect_mismatch(text)

    if mismatch:
        prob_fake = min(prob_fake + 0.40, 1.0)

    reasons = []

    if mismatch:
        reasons.append("Mismatch between job role and qualification")

    if has_money_words:
        reasons.append("Contains salary-related terms")

    if has_urgent_words:
        reasons.append("Uses urgency words (e.g., urgent, immediate)")

    if fake_company:
        reasons.append("Famous company used with suspicious job role")
    
    if num_exclamations > 3:
        reasons.append("Excessive use of exclamation marks")

    return {
        "real_probability": round((1 - prob_fake) * 100, 2),
        "fake_probability": round(prob_fake * 100, 2),
        "reasons": reasons
    }