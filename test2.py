import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import os

def load_texts(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# --- 1. LOAD DATA ---
successful_texts = load_texts("successful.txt")
unsuccessful_texts = load_texts("unsuccessful.txt")
texts_to_test = load_texts("test_texts.txt")

if not successful_texts or not unsuccessful_texts:
    print("Error: Need texts to continue.")
    exit()

os.makedirs("kategori", exist_ok=True)

all_known_texts = successful_texts + unsuccessful_texts

# --- 2. ENHANCED VECTORIZATION ---
print("Initializing Enhanced TF-IDF Vectorizer...")
try:
    turkish_stop_words = stopwords.words('turkish')
except LookupError:
    nltk.download('stopwords')
    turkish_stop_words = stopwords.words('turkish')

# Key changes: Turkish stopwords + ngram_range(1, 2) looks at single words AND pairs
vectorizer = TfidfVectorizer(stop_words=turkish_stop_words, ngram_range=(1, 2))
vectorizer.fit(all_known_texts)

succ_vectors = vectorizer.transform(successful_texts)
unsucc_vectors = vectorizer.transform(unsuccessful_texts)

succ_profile = np.asarray(np.mean(succ_vectors, axis=0))
unsucc_profile = np.asarray(np.mean(unsucc_vectors, axis=0))

# --- 3. CLASSIFICATION ---
print("\n--- RESULTS (APPROACH 1: PREPROCESSING) ---")
new_vectors = vectorizer.transform(texts_to_test)
feature_names = vectorizer.get_feature_names_out()

for i, text in enumerate(texts_to_test):
    succ_score = cosine_similarity(new_vectors[i], succ_profile)[0][0]
    unsucc_score = cosine_similarity(new_vectors[i], unsucc_profile)[0][0]
    
    if succ_score > unsucc_score:
        result = "SUCCESSFUL"
    elif unsucc_score > succ_score:
        result = "UNSUCCESSFUL"
    else:
        result = "NEUTRAL"
        
    print(f"\nText: '{text}'")
    print(f"Classification: {result}")
    print(f"Success Score: {succ_score:.3f} | Unsuccess Score: {unsucc_score:.3f}")
    
    # What triggered this? Show top non-zero features matching the text
    vector = np.asarray(new_vectors[i].todense()).flatten()
    top_indices = np.argsort(vector)[::-1]
    print("Key trigrams/words found in this text:")
    found_any = False
    for idx in top_indices[:5]:
        if vector[idx] > 0:
            print(f"  - '{feature_names[idx]}': score {vector[idx]:.3f}")
            found_any = True
    if not found_any:
        print("  - (No matching vocabulary found)")

# --- 4. PCA VISUALIZATION ---
print("\nGenerating PCA visualization...")
all_vectors = vectorizer.transform(all_known_texts + texts_to_test).toarray()
pca = PCA(n_components=2)
coords = pca.fit_transform(all_vectors)

plt.figure(figsize=(10, 7))
n_succ = len(successful_texts)
plt.scatter(coords[:n_succ, 0], coords[:n_succ, 1], c='green', label='Success', s=100)
n_unsucc = len(unsuccessful_texts)
plt.scatter(coords[n_succ:n_succ+n_unsucc, 0], coords[n_succ:n_succ+n_unsucc, 1], c='red', label='Unsuccess', s=100)
plt.scatter(coords[n_succ+n_unsucc:, 0], coords[n_succ+n_unsucc:, 1], c='blue', label='Test Texts', s=150, marker='X')

for i, text in enumerate(texts_to_test):
    plt.annotate(f"Test {i+1}", (coords[n_succ+n_unsucc+i, 0], coords[n_succ+n_unsucc+i, 1]))

plt.title("Approach 1 Embedding Clusters")
plt.legend()
plt.savefig("kategori/2_pca.png")
print("Saved: test2_pca.png")

# Save PCA coordinates to text file
pca_data = []
pca_data.append("SUCCESSFUL TEXTS (GREEN):")
for i in range(n_succ):
    pca_data.append(f"X: {coords[i, 0]:.3f}, Y: {coords[i, 1]:.3f} | Text: {successful_texts[i]}")

pca_data.append("\nUNSUCCESSFUL TEXTS (RED):")
for i in range(n_unsucc):
    idx = n_succ + i
    pca_data.append(f"X: {coords[idx, 0]:.3f}, Y: {coords[idx, 1]:.3f} | Text: {unsuccessful_texts[i]}")

pca_data.append("\nTEST TEXTS (BLUE):")
for i in range(len(texts_to_test)):
    idx = n_succ + n_unsucc + i
    pca_data.append(f"X: {coords[idx, 0]:.3f}, Y: {coords[idx, 1]:.3f} | Text: {texts_to_test[i]}")

with open("kategori/2_pca_data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(pca_data))
print("Saved: kategori/2_pca_data.txt")

# --- 5. GENERATE TEXT REPORT ---
print("\nGenerating Report...")
report = []
report.append("="*50)
report.append("TEST 2 - REPORT SUMMARY")
report.append("="*50)
report.append("APPROACH:")
report.append("- Method: Enhanced TF-IDF Vectorizer (TfidfVectorizer)")
report.append("- Parameters: stop_words='turkish' (Uses Turkish stopwords), n-grams: 1-2 (unigrams and bigrams)")
report.append("- Mechanics: Calculates 'average' vector profiles for Known Success and Known Unsuccess. Uses Cosine Similarity to compare Test Texts to these profiles.")
report.append("\nRESULTS FOR TEST TEXTS:")

for i, text in enumerate(texts_to_test):
    succ_score = cosine_similarity(new_vectors[i], succ_profile)[0][0]
    unsucc_score = cosine_similarity(new_vectors[i], unsucc_profile)[0][0]
    
    if succ_score > unsucc_score:
        result = "SUCCESSFUL"
    elif unsucc_score > succ_score:
        result = "UNSUCCESSFUL"
    else:
        result = "NEUTRAL"
        
    report.append(f"\n[Test Text {i+1}]")
    report.append(f"Content: '{text}'")
    report.append(f"Classification: {result}")
    report.append(f"Scores -> Similarity to Success: {succ_score:.3f} | Similarity to Failure: {unsucc_score:.3f}")
    
    report.append("Key matching n-grams found in this text:")
    vector = np.asarray(new_vectors[i].todense()).flatten()
    top_indices = np.argsort(vector)[::-1]
    found_any = False
    for idx in top_indices[:5]:
        if vector[idx] > 0:
            report.append(f"  - '{feature_names[idx]}': TF-IDF score {vector[idx]:.3f}")
            found_any = True
    if not found_any:
        report.append("  - (No matching vocabulary found)")

report.append("\nVISUALIZATIONS GENERATED:")
report.append("- kategori/2_pca.png: PCA scatter plot of vectors using the enhanced TF-IDF model.")
report.append("\n")

os.makedirs("kategori", exist_ok=True)
with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(report))
print("Report appended to kategori/rapor.txt")
