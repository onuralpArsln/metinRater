import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import os

def load_texts(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# --- 1. LOAD DATA & PREPARE LABELS ---
successful_texts = load_texts("successful.txt")
unsuccessful_texts = load_texts("unsuccessful.txt")
texts_to_test = load_texts("test_texts.txt")

if not successful_texts or not unsuccessful_texts:
    print("Error: Need texts to continue.")
    exit()

all_known_texts = successful_texts + unsuccessful_texts
# Create labels: 1 for Successful, 0 for Unsuccessful
y_train = [1] * len(successful_texts) + [0] * len(unsuccessful_texts)

# --- 2. VECTORIZATION ---
print("Initializing TF-IDF Vectorizer...")
try:
    turkish_stop_words = stopwords.words('turkish')
except LookupError:
    nltk.download('stopwords')
    turkish_stop_words = stopwords.words('turkish')

vectorizer = TfidfVectorizer(stop_words=turkish_stop_words, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(all_known_texts)

# --- 3. TRAIN CLASSIFIER ---
print("Training Logistic Regression Model...")
# C=10 is regularization strength. Higher means it trusts the sparse training data more.
model = LogisticRegression(class_weight='balanced', C=10)
model.fit(X_train, y_train)

# --- 4. CLASSIFY NEW TEXTS ---
print("\n--- RESULTS (APPROACH 2: CLASSIFIER) ---")
X_test = vectorizer.transform(texts_to_test)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

for i, text in enumerate(texts_to_test):
    # prob[0] is probability of class 0 (Unsuccessful)
    # prob[1] is probability of class 1 (Successful)
    prob_unsucc, prob_succ = probabilities[i]
    
    if predictions[i] == 1:
        result = "SUCCESSFUL"
        confidence = prob_succ * 100
    else:
        result = "UNSUCCESSFUL"
        confidence = prob_unsucc * 100

    print(f"\nText: '{text}'")
    print(f"Classification: {result} (Confidence: {confidence:.1f}%)")
    print(f"Raw Probabilities -> Success: {prob_succ:.3f} | Unsuccess: {prob_unsucc:.3f}")

# Optional: Show what words the model thinks are most important overall
print("\nModel's Most Important Keywords:")
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

# Top 5 successful words
top_succ_indices = np.argsort(coefs)[-5:][::-1]
print("Strongest Indicators of Success:")
for idx in top_succ_indices:
    if coefs[idx] > 0:
        print(f"  - '{feature_names[idx]}' (Weight +{coefs[idx]:.2f})")

# Top 5 unsuccessful words
top_unsucc_indices = np.argsort(coefs)[:5]
print("Strongest Indicators of Failure:")
for idx in top_unsucc_indices:
    if coefs[idx] < 0:
        print(f"  - '{feature_names[idx]}' (Weight {coefs[idx]:.2f})")
