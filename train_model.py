# ============================================================
#  Fake Job Post Detection System - Updated Training Script
#  Fixes: class_weight, more features, SMOTE, threshold tuning
# ============================================================

# ------------------------------------
# 1. Import Libraries
# ------------------------------------
import pandas as pd
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

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
X = X.apply(preprocess)
print("Preprocessing done!")

# ------------------------------------
# 5. Convert Text → Numbers (TF-IDF)
# ------------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=5,
    max_df=0.9
)
from scipy.sparse import hstack

X_vector = vectorizer.fit_transform(X_text)

# Combine TF-IDF + extra features
X_final = hstack((X_vector, X_extra))

# ------------------------------------
# 6. Train Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("\nBefore SMOTE:")
print("  Real jobs in train:", sum(y_train == 0))
print("  Fake jobs in train:", sum(y_train == 1))

# ------------------------------------
# 7. ✅ FIX: Apply SMOTE to balance classes
# ------------------------------------
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("  Real jobs in train:", sum(y_train == 0))
print("  Fake jobs in train:", sum(y_train == 1))

# ------------------------------------
# 8. Train Logistic Regression Model  ✅ UPDATED
# ------------------------------------
from sklearn.linear_model import LogisticRegression

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'   # handles imbalance better
)

model.fit(X_train, y_train)
print("Training complete!")

# ------------------------------------
# 9. Model Evaluation
# ------------------------------------
y_pred = model.predict(X_test)

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
def predict_job(text, threshold=0.35):
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
    vector = vectorizer.transform([cleaned])

    # Create same extra features for input
    text_length = len(text)
    num_exclamations = text.count("!")

    has_money_words = int(any(word in text.lower() for word in ["salary", "$", "earn", "income", "paid"]))
    has_urgent_words = int(any(word in text.lower() for word in ["urgent", "immediate", "limited", "hurry"]))

    import numpy as np
    extra_features = np.array([[text_length, num_exclamations, has_money_words, has_urgent_words]])

    from scipy.sparse import hstack
    vector = hstack((vector, extra_features))
    prob_fake = model.predict_proba(vector)[0][1]
    # 🔥 Apply domain mismatch logic
    mismatch = detect_mismatch(text)

    if mismatch:
        prob_fake = min(prob_fake + 0.15, 1.0)
      # Boost probability if scam keywords found
    if keyword_hits >= 2:
        prob_fake = min(prob_fake + 0.2, 1.0)  # boost by 20%
    elif keyword_hits == 1:
        prob_fake = min(prob_fake + 0.1, 1.0)  # boost by 10%
    
    print(f"\n--- Prediction Result ---")
    print(f"Fake Probability : {prob_fake:.2%}")
    print(f"Scam Keywords Found: {keyword_hits}")
    
    if prob_fake > threshold:
        print(f"⚠️  FAKE Job Post Detected!")
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
predict_job(test1, threshold=0.35)

# Test 2 - Subtle scam impersonating real company
test2 = """
Google is hiring remote data entry operators.
No qualifications required. Salary $200 per day.
Send your personal details and pay $50 processing fee.
Work from home. Immediate joining. Urgent hiring.
"""
print("\nTest 2 - Subtle Scam (fake company):")
predict_job(test2, threshold=0.35)

# Test 3 - Real-looking legitimate job
test3 = """
Software Engineer at Microsoft. 3 plus years Python experience required.
Bachelor degree in Computer Science preferred.
Competitive salary and benefits. Apply through official company website.
Strong problem solving skills. Experience with cloud platforms a plus.
"""
print("\nTest 3 - Legitimate Job Post:")
predict_job(test3, threshold=0.35)

# Test 4 - Another real job
test4 = """
Data Analyst position at Infosys, Bangalore.
2 years experience in SQL and Excel required.
MBA or relevant degree. 5 day work week.
Health insurance provided. PF and gratuity as per norms.
"""
print("\nTest 4 - Legitimate Job Post 2:")
predict_job(test4, threshold=0.35)

# ------------------------------------
# 12. Save Model & Vectorizer
# ------------------------------------
pickle.dump(model, open("rf_fake_job_model.pkl", "wb"))
pickle.dump(vectorizer, open("rf_vectorizer.pkl", "wb"))

print("\n" + "="*50)
print("✅ Model and Vectorizer saved successfully!")
print("   Files: rf_fake_job_model.pkl, rf_vectorizer.pkl")
print("="*50)
