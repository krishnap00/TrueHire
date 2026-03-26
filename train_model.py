# ============================================================
#  Fake Job Post Detection System - Updated Training Script
#  Fixes: class_weight, more features, SMOTE, threshold tuning
# ============================================================

# ------------------------------------
# 1. Import Libraries
# ------------------------------------
from pydoc import text
from xml.parsers.expat import model

import pandas as pd
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

# from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE

nltk.download('stopwords')
nltk.download('wordnet')

# ------------------------------------
# 2. Load Dataset
# ------------------------------------
df = pd.read_csv("fake_job_postings.csv")

print("Dataset shape:", df.shape)
print("Fake vs Real counts:\n", df["fraudulent"].value_counts())

# ------------------------------------
# 3. Data Cleaning + More Features
# ------------------------------------
df = df.fillna("")

# ✅ FIX: Added more columns for richer text signals
df["text"] = (
    df["title"] + " " +
    df["location"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"] + " " +
    df["employment_type"] + " " +
    df["required_experience"]
)
# 🔥 NEW FEATURES
df["text_length"] = df["text"].apply(len)
df["num_exclamations"] = df["text"].apply(lambda x: x.count("!"))

# Keyword-based features
money_keywords = ["salary", "$", "earn", "income", "paid"]
urgent_keywords = ["urgent", "immediate", "limited", "hurry"]

df["has_money_words"] = df["text"].apply(
    lambda x: int(any(word in x.lower() for word in money_keywords))
)

df["has_urgent_words"] = df["text"].apply(
    lambda x: int(any(word in x.lower() for word in urgent_keywords))
)

X_text = df["text"]

# Additional numeric features
X_extra = df[[
    "text_length",
    "num_exclamations",
    "has_money_words",
    "has_urgent_words"
]]
y = df["fraudulent"]

# ------------------------------------
# 4. Text Preprocessing
# ------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

print("\nPreprocessing text... (this may take a minute)")
X_text = X_text.apply(preprocess)
print("Preprocessing done!")
# print("\nLoading embedding model...")
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# print("Embedding model loaded!")

# ------------------------------------
# 5. Convert Text → Numbers (HYBRID)
# ------------------------------------

# 🔵 TF-IDF (Model 1 - Fast)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=5,
    max_df=0.9
)

X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# 🟣 Embeddings (Model 2 - Smart)
#print("\nGenerating embeddings...")
#X_embed = embedder.encode(
#    X_text.tolist(),
#    batch_size=64,
#    show_progress_bar=True
#)
#print("Embeddings generated!")

# Combine with extra features
import numpy as np
from scipy.sparse import hstack

X_tfidf_final = hstack((X_tfidf, X_extra))
#X_embed_final = np.hstack((X_embed, X_extra.values))
# ------------------------------------
# 6. Train Test Split
# ------------------------------------
# Split TF-IDF
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf_final, y, test_size=0.2, random_state=42, stratify=y
)

# Split Embeddings (same split)
#X_train_embed, X_test_embed, _, _ = train_test_split(
#    X_embed_final, y, test_size=0.2, random_state=42, stratify=y
#)

print("\nBefore SMOTE:")
print("  Real jobs in train:", sum(y_train == 0))
print("  Fake jobs in train:", sum(y_train == 1))

# ------------------------------------
# 7. ✅ FIX: Apply SMOTE to balance classes
# ------------------------------------
# print("\nApplying SMOTE to balance the dataset...")
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# print("After SMOTE:")
print("  Real jobs in train:", sum(y_train == 0))
print("  Fake jobs in train:", sum(y_train == 1))

# ------------------------------------
# 8. Train XGBoost Model  ✅ UPDATED
# ------------------------------------
from xgboost import XGBClassifier

print("\nTraining XGBoost model...")

# Calculate scale_pos_weight to handle class imbalance
# (replaces class_weight='balanced' from LogisticRegression)
neg = sum(y_train == 0)
pos = sum(y_train == 1)
scale = neg / pos  # e.g. ~14 if dataset is heavily imbalanced

tfidf_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,   # handles class imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist'        # faster training
)

tfidf_model.fit(X_train_tfidf, y_train)
print("XGBoost model trained!")
# ------------------------------------
# 9. Model Evaluation
# ------------------------------------
y_pred = tfidf_model.predict(X_test_tfidf)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n  True Real  (correct): {cm[0][0]}")
print(f"  False Fake (wrong):   {cm[0][1]}")
print(f"  Missed Fake (wrong):  {cm[1][0]}  ← want this LOW")
print(f"  True Fake  (correct): {cm[1][1]}  ← want this HIGH")

# ------------------------------------
# 10. ✅ FIX: Predict with probability threshold
# ------------------------------------
# ------------------------------------
# 🔥 NEW: Domain Mismatch Detection
# ------------------------------------
def detect_mismatch(text):
    text = text.lower()
    
    tech_roles = [
        "software engineer", "developer", "data scientist",
        "backend engineer", "frontend developer"
    ]
    
    non_tech_degrees = [
        "bsc maths", "history", "arts", "biology",
        "commerce", "ba", "bcom"
    ]
    
    role_flag = any(role in text for role in tech_roles)
    degree_flag = any(deg in text for deg in non_tech_degrees)
    
    if role_flag and degree_flag:
        return 1  # mismatch detected
    return 0
def detect_fake_company(text):
    suspicious_words = [
        "unknown company", "startup soon", "new company",
        "confidential company", "growing company"
    ]
    
    return int(any(word in text.lower() for word in suspicious_words))
import re

def detect_suspicious_email(text):
    emails = re.findall(r'\S+@\S+', text)
    
    suspicious_domains = ["gmail.com", "yahoo.com", "outlook.com"]
    
    for email in emails:
        if any(domain in email for domain in suspicious_domains):
            return 1
    return 0
def detect_high_salary(text):
    import re
    salaries = re.findall(r'\$?\d+', text)
    
    for sal in salaries:
        try:
            if int(sal.replace("$", "")) > 10000:
                return 1
        except:
            continue
    return 0
def detect_weird_format(text):
    if text.count("!!!") > 0:
        return 1
    if text.isupper():
        return 1
    return 0
def predict_job(text, threshold=0.45):
    fake_company = detect_fake_company(text)
    suspicious_email = detect_suspicious_email(text)
    high_salary = detect_high_salary(text)
    weird_format = detect_weird_format(text)

    # Scam keyword boost
    scam_keywords = [
        "registration fee", "pay fee", "processing fee",
        "earn from home", "no experience needed", "guaranteed income",
        "limited slots", "work from home earn", "daily income",
        "whatsapp", "send personal details", "wire transfer"
    ]
    
    text_lower = text.lower()
    keyword_hits = sum(1 for kw in scam_keywords if kw in text_lower)
    
    cleaned = preprocess(text)
    # TF-IDF vector
    tfidf_vec = tfidf_vectorizer.transform([cleaned])

    # Embedding vector
    #embed_vec = np.array(embedder.encode([cleaned]))
    
    # Create same extra features for input
    text_length = len(text)
    num_exclamations = text.count("!")

    has_money_words = int(any(word in text.lower() for word in ["salary", "$", "earn", "income", "paid"]))
    has_urgent_words = int(any(word in text.lower() for word in ["urgent", "immediate", "limited", "hurry"]))

    import numpy as np
    extra_features = np.array([[text_length, num_exclamations, has_money_words, has_urgent_words]])
    from scipy.sparse import hstack

    # Combine TF-IDF with extra features
    tfidf_vec = hstack((tfidf_vec, extra_features))

    # Combine embeddings with extra features
    #embed_vec = np.hstack((embed_vec, extra_features))
    #tfidf_prob = tfidf_model.predict_proba(tfidf_vec)[0][1]
    #embed_prob = embed_model.predict_proba(embed_vec)[0][1]

    #prob_fake = 0.7 * tfidf_prob + 0.3 * embed_prob
    tfidf_prob = tfidf_model.predict_proba(tfidf_vec)[0][1]
    prob_fake = tfidf_prob

    if fake_company:
        prob_fake = min(prob_fake + 0.05, 1.0)

    if suspicious_email:
        prob_fake = min(prob_fake + 0.07, 1.0)

    if high_salary:
        prob_fake = min(prob_fake + 0.08, 1.0)

    if weird_format:
        prob_fake = min(prob_fake + 0.05, 1.0)
    # 🔥 Apply domain mismatch logic
    mismatch = detect_mismatch(text)

    if mismatch:
        prob_fake = min(prob_fake + 0.08, 1.0)
      # Boost probability if scam keywords found
    if keyword_hits >= 2:
        prob_fake = min(prob_fake + 0.15, 1.0)
    elif keyword_hits == 1:
        prob_fake = min(prob_fake + 0.05, 1.0)
    
    print(f"\n--- Prediction Result ---")
    print(f"Fake Probability : {prob_fake:.2%}")
    print(f"Scam Keywords Found: {keyword_hits}")
    
    if prob_fake > threshold:
        print(f"⚠️  FAKE Job Post Detected!")
    elif prob_fake > 0.5:
        print("⚠️ Suspicious Job Post (Review Recommended)")
    else:
        print(f"✅ Likely Real Job Post")
    
    return prob_fake

# ------------------------------------
# 11. Test with example job posts
# ------------------------------------
print("\n" + "="*50)
print("TESTING CUSTOM JOB POSTS")
print("="*50)

# Test 1 - Obvious scam
test1 = """
Earn $4000 weekly working from home.
No experience needed. Limited slots available.
Click the link and pay registration fee to start today!
Guaranteed daily income. No skills required.
"""
print("\nTest 1 - Obvious Scam:")
predict_job(test1)

# Test 2 - Subtle scam impersonating real company
test2 = """
Google is hiring remote data entry operators.
No qualifications required. Salary $200 per day.
Send your personal details and pay $50 processing fee.
Work from home. Immediate joining. Urgent hiring.
"""
print("\nTest 2 - Subtle Scam (fake company):")
predict_job(test2)

# Test 3 - Real-looking legitimate job
test3 = """
Software Engineer at Microsoft. 3 plus years Python experience required.
Bachelor degree in Computer Science preferred.
Competitive salary and benefits. Apply through official company website.
Strong problem solving skills. Experience with cloud platforms a plus.
"""
print("\nTest 3 - Legitimate Job Post:")
predict_job(test3)

# Test 4 - Another real job
test4 = """
mechanical engineer required. high salary. qualification: 10th pass
"""
print("\nTest 4 - Legitimate Job Post 2:")
predict_job(test4)

# ------------------------------------
# 12. Save Model & Vectorizer
# ------------------------------------
#pickle.dump(tfidf_model, open("tfidf_model.pkl", "wb"))
#pickle.dump(embed_model, open("embed_model.pkl", "wb"))
pickle.dump(tfidf_model, open("tfidf_model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\n" + "="*50)
print("✅ Model and Vectorizer saved successfully!")
print("   Files: tfidf_model.pkl, tfidf_vectorizer.pkl")
print("="*50)