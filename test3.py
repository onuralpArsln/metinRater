import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import os
import matplotlib.pyplot as plt

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

os.makedirs("kategori", exist_ok=True)

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

# --- 5. VISUALIZATIONS ---
print("\nGenerating visualizations for test3...")
plt.figure(figsize=(10, 6))
# Get top 10 positive and top 10 negative
top_pos_idx = np.argsort(coefs)[-10:]
top_neg_idx = np.argsort(coefs)[:10]

combined_idx = np.concatenate([top_neg_idx, top_pos_idx])
colors = ['red' if coefs[i] < 0 else 'green' for i in combined_idx]
features = [feature_names[i] for i in combined_idx]
importances = [coefs[i] for i in combined_idx]

plt.barh(features, importances, color=colors)
plt.title("Top 10 Positive and Negative Features (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("kategori/3_feature_importance.png")
print("Saved: kategori/3_feature_importance.png")

# Probability Histogram
succ_probs = probabilities[:, 1]
plt.figure(figsize=(8, 5))
plt.hist(succ_probs, bins=10, color='blue', alpha=0.7, range=(0,1))
plt.title("Prediction Confidence for 'Successful' Class")
plt.xlabel("Probability of being 'Successful' (0 to 1)")
plt.ylabel("Number of Texts")
plt.tight_layout()
plt.savefig("kategori/3_confidence_histogram.png")
print("Saved: kategori/3_confidence_histogram.png")

# --- 6. GENERATE TEXT REPORT ---
print("\nGenerating Report...")
report = []
report.append("="*50)
report.append("TEST 3 - REPORT SUMMARY")
report.append("="*50)
report.append("APPROACH:")
report.append("- Method: Logistic Regression Classifier trained on Enhanced TF-IDF features")
report.append("- Parameters: TF-IDF(turkish stopwords, n-grams 1-2), LR(class_weight='balanced', C=10)")
report.append("- Mechanics: Learns which specific n-grams predict success vs. failure using logistic regression weights. Outputs mathematical probabilities.")
report.append("\nGLOBAL MODEL PROPERTIES:")
report.append("Strongest Indicators of Success (Top 5 Positive Coefficients):")
for idx in top_succ_indices[:5]:
    if coefs[idx] > 0:
        report.append(f"  - '{feature_names[idx]}' (Weight +{coefs[idx]:.2f})")

report.append("Strongest Indicators of Failure (Top 5 Negative Coefficients):")
for idx in top_unsucc_indices[:5]:
    if coefs[idx] < 0:
        report.append(f"  - '{feature_names[idx]}' (Weight {coefs[idx]:.2f})")

report.append("\nRESULTS FOR TEST TEXTS:")

for i, text in enumerate(texts_to_test):
    prob_unsucc, prob_succ = probabilities[i]
    if predictions[i] == 1:
        result = "SUCCESSFUL"
        confidence = prob_succ * 100
    else:
        result = "UNSUCCESSFUL"
        confidence = prob_unsucc * 100
        
    report.append(f"\n[Test Text {i+1}]")
    report.append(f"Content: '{text}'")
    report.append(f"Classification: {result} (Confidence: {confidence:.1f}%)")
    report.append(f"Raw Probabilities -> Success: {prob_succ:.3f} | Failure: {prob_unsucc:.3f}")

report.append("\nVISUALIZATIONS GENERATED:")
report.append("- kategori/3_feature_importance.png: Horizontal bar chart showing the top 10 positive and negative logistic regression features.")
report.append("- kategori/3_confidence_histogram.png: Histogram showing the distribution of the model's confidence scores.")
report.append("\n")

os.makedirs("kategori", exist_ok=True)
with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(report))
print("Report appended to kategori/rapor.txt")
