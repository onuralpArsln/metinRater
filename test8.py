import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_texts(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("========================================")
    print("TEST 8 - HIGH-DIMENSIONAL SEMANTIC SVM")
    print("========================================")
    
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")
    texts_to_test = load_texts("test_texts.txt")

    if not successful_texts or not unsuccessful_texts:
        print("Error: Need both successful and unsuccessful texts to continue.")
        exit()

    all_known_texts = successful_texts + unsuccessful_texts
    y_train = [1] * len(successful_texts) + [0] * len(unsuccessful_texts)

    print("Loading Pretrained Multilingual NLP Model (This may take a moment the first time)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("Converting all texts into 384-Dimensional Semantic Embeddings...")
    X_train = model.encode(all_known_texts)
    X_test = model.encode(texts_to_test)

    print("Training Support Vector Machine (SVM) on Semantic Space...")
    # SVM is excellent at drawing complex boundaries in high-dimensional space
    svm_clf = SVC(kernel='linear', probability=True, class_weight='balanced', C=1.0)
    svm_clf.fit(X_train, y_train)

    predictions = svm_clf.predict(X_test)
    probabilities = svm_clf.predict_proba(X_test)

    print("\n--- RESULTS (APPROACH 8: SEMANTIC SVM) ---")
    for i, text in enumerate(texts_to_test):
        prob_unsucc, prob_succ = probabilities[i]
        
        if predictions[i] == 1:
            result = "SUCCESSFUL"
            confidence = prob_succ * 100
        else:
            result = "UNSUCCESSFUL"
            confidence = prob_unsucc * 100
            
        print(f"\nText: '{text}'")
        print(f"Semantic Classification: {result} (Confidence: {confidence:.1f}%)")

    print("\nGenerating t-SNE visualization for Test 8...")
    os.makedirs("kategori", exist_ok=True)
    
    # We visualize the semantic groupings in 2D using t-SNE.
    # t-SNE preserves local neighborhood relationships.
    all_vectors = np.vstack((X_train, X_test))
    
    # Auto-adjust perplexity for small datasets to avoid warnings/errors
    n_samples = len(all_vectors)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(all_vectors)

    plt.figure(figsize=(10, 7))
    n_succ = len(successful_texts)
    n_unsucc = len(unsuccessful_texts)
    
    plt.scatter(coords[:n_succ, 0], coords[:n_succ, 1], c='green', label='Success cluster', s=100)
    plt.scatter(coords[n_succ:n_succ+n_unsucc, 0], coords[n_succ:n_succ+n_unsucc, 1], c='red', label='Unsuccess cluster', s=100)
    plt.scatter(coords[n_succ+n_unsucc:, 0], coords[n_succ+n_unsucc:, 1], c='blue', label='Test Texts', s=150, marker='X')

    for i, text in enumerate(texts_to_test):
        plt.annotate(f"Test {i+1}", (coords[n_succ+n_unsucc+i, 0], coords[n_succ+n_unsucc+i, 1]))

    plt.title("Approach 8: SVM clusters in Semantic Space (t-SNE mapped)")
    plt.legend()
    plt.savefig("kategori/8_tsne_svm.png")
    print("Saved: kategori/8_tsne_svm.png")

    report = []
    report.append("\n" + "="*50)
    report.append("TEST 8 - REPORT SUMMARY")
    report.append("="*50)
    report.append("YAKLAŞIM (Test 8 - Semantik SVM):")
    report.append("* Odak Noktası: Sadece cümlenin GENEL ANLAMINA ve BAĞLAMINA bakar, hassas bir sınır çizer.")
    report.append("* Nasıl Çalışır: Metinleri 384 boyutlu vektörlere çevirip 'Destek Vektör Makineleri' (SVM) ile kesin bir matematiksel sınır çeker.")
    report.append("* Neye Bakmaz: Kelimelerin frekansına veya sırasına.")
    report.append("* Sonuç Ne İfade Eder: SVM modelinin sınırlarının hangi tarafına düştüğünü gösteren Olasılık Skorudur (Confidence %).")
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
        report.append(f"Semantic SVM Classification: {result} (Confidence: {confidence:.1f}%)")

    with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("Report appended to kategori/rapor.txt")


if __name__ == "__main__":
    main()
