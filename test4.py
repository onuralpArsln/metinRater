import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

all_known_texts = successful_texts + unsuccessful_texts

# --- 2. SEMANTIC EMBEDDINGS ---
print("Loading Pretrained Multilingual NLP Model (This may take a moment the first time)...")
# Using a lightweight, fast, multilingual model perfect for Turkish
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Generating sentence embeddings based on meaning...")
succ_vectors = model.encode(successful_texts)
unsucc_vectors = model.encode(unsuccessful_texts)

# Calculate "Average Meaning" profile
succ_profile = np.asarray(np.mean(succ_vectors, axis=0)).reshape(1, -1)
unsucc_profile = np.asarray(np.mean(unsucc_vectors, axis=0)).reshape(1, -1)

# --- 3. CLASSIFICATION ---
print("\n--- RESULTS (APPROACH 3: SEMANTIC MEANING) ---")
test_vectors = model.encode(texts_to_test)

for i, text in enumerate(texts_to_test):
    # Encodings are already normalized, but cosine similarity works out of the box
    succ_score = cosine_similarity(test_vectors[i].reshape(1, -1), succ_profile)[0][0]
    unsucc_score = cosine_similarity(test_vectors[i].reshape(1, -1), unsucc_profile)[0][0]
    
    if succ_score > unsucc_score:
        result = "SUCCESSFUL"
    elif unsucc_score > succ_score:
        result = "UNSUCCESSFUL"
    else:
        result = "NEUTRAL"
        
    print(f"\nText: '{text}'")
    print(f"Classification: {result}")
    print(f"Semantic Similarity -> to Success: {succ_score:.3f} | to Failure: {unsucc_score:.3f}")

# --- 4. PCA VISUALIZATION ---
print("\nGenerating PCA visualization for Sentence Embeddings...")
all_vectors = model.encode(all_known_texts + texts_to_test)
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

plt.title("Approach 3: Semantic Embeddings (PCA)")
plt.legend()
plt.savefig("test4_pca.png")
print("Saved: test4_pca.png")
