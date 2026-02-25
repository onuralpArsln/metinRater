import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

os.makedirs("kategori", exist_ok=True)

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
plt.savefig("kategori/4_pca.png")
print("Saved: test4_pca.png")

# --- 5. VISUALIZATION 2: t-SNE Transform ---
print("\nGenerating t-SNE visualization for Sentence Embeddings...")
# Perplexity should be smaller than the number of samples
perplexity_val = min(5, max(1, len(all_vectors) - 1))
tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
coords_tsne = tsne.fit_transform(all_vectors)

plt.figure(figsize=(10, 7))
plt.scatter(coords_tsne[:n_succ, 0], coords_tsne[:n_succ, 1], c='green', label='Success', s=100)
plt.scatter(coords_tsne[n_succ:n_succ+n_unsucc, 0], coords_tsne[n_succ:n_succ+n_unsucc, 1], c='red', label='Unsuccess', s=100)
plt.scatter(coords_tsne[n_succ+n_unsucc:, 0], coords_tsne[n_succ+n_unsucc:, 1], c='blue', label='Test Texts', s=150, marker='X')

for i, text in enumerate(texts_to_test):
    plt.annotate(f"Test {i+1}", (coords_tsne[n_succ+n_unsucc+i, 0], coords_tsne[n_succ+n_unsucc+i, 1]))

plt.title("Approach 4: Semantic Embeddings (t-SNE)")
plt.legend()
plt.savefig("kategori/4_tsne.png")
print("Saved: kategori/4_tsne.png")

# --- 6. VISUALIZATION 3: Cosine Similarity Heatmap ---
print("\nGenerating Cosine Similarity Heatmap...")
sim_matrix = cosine_similarity(test_vectors, np.vstack([succ_profile, unsucc_profile]))

plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks([0, 1], ['Success Profile', 'Unsuccess Profile'])
plt.yticks(range(len(texts_to_test)), [f"Test {i+1}" for i in range(len(texts_to_test))])

# Add text annotations on the heatmap
for i in range(len(texts_to_test)):
    for j in range(2):
        val = sim_matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        plt.text(j, i, f"{val:.3f}", ha='center', va='center', color=color)

plt.title("Cosine Similarity to Profiles (Test Texts)")
plt.tight_layout()
plt.savefig("kategori/4_similarity_heatmap.png")
print("Saved: kategori/4_similarity_heatmap.png")

# --- 7. GENERATE TEXT REPORT ---
print("\nGenerating Report...")
report = []
report.append("="*50)
report.append("TEST 4 - REPORT SUMMARY")
report.append("="*50)
report.append("APPROACH:")
report.append("- Method: Pre-trained Multilingual Sentence Transformer Model")
report.append("- Parameters: Model = 'paraphrase-multilingual-MiniLM-L12-v2' (Dense semantic embeddings)")
report.append("- Mechanics: Converts sentences into high-dimensional vectors based on semantic meaning, capturing context regardless of the exact wording. Uses Cosine Similarity to compare Test Texts to the average Meaning Profiles of Success/Failure.")
report.append("\nRESULTS FOR TEST TEXTS:")

for i, text in enumerate(texts_to_test):
    succ_score = cosine_similarity(test_vectors[i].reshape(1, -1), succ_profile)[0][0]
    unsucc_score = cosine_similarity(test_vectors[i].reshape(1, -1), unsucc_profile)[0][0]
    
    if succ_score > unsucc_score:
        result = "SUCCESSFUL"
    elif unsucc_score > succ_score:
        result = "UNSUCCESSFUL"
    else:
        result = "NEUTRAL"
        
    report.append(f"\n[Test Text {i+1}]")
    report.append(f"Content: '{text}'")
    report.append(f"Classification: {result}")
    report.append(f"Semantic Distances -> Similarity to Success: {succ_score:.3f} | Similarity to Failure: {unsucc_score:.3f}")

report.append("\nVISUALIZATIONS GENERATED:")
report.append("- kategori/4_pca.png: PCA scatter plot of sentence embeddings.")
report.append("- kategori/4_tsne.png: t-SNE scatter plot of sentence embeddings (better at separating semantic clusters).")
report.append("- kategori/4_similarity_heatmap.png: Visual grid of the exact cosine similarity scores.")
report.append("\n")

os.makedirs("kategori", exist_ok=True)
with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(report))
print("Report appended to kategori/rapor.txt")

