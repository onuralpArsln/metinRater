import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def load_texts(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def run_test(texts_to_score):
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")
    if not successful_texts or not unsuccessful_texts:
        return [0.0] * len(texts_to_score)
        
    all_known_texts = successful_texts + unsuccessful_texts
    y_train = [1] * len(successful_texts) + [0] * len(unsuccessful_texts)
    
    vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(3, 5),
        lowercase=False
    )
    X_train = vectorizer.fit_transform(all_known_texts)
    
    model = LogisticRegression(class_weight='balanced', C=10)
    model.fit(X_train, y_train)

    X_test = vectorizer.transform(texts_to_score)
    probabilities = model.predict_proba(X_test)
    
    return [prob[1] for prob in probabilities]

def main():
    # --- 1. LOAD DATA & PREPARE LABELS ---
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")
    texts_to_test = load_texts("test_texts.txt")

    if not successful_texts or not unsuccessful_texts:
        print("Error: Need texts to continue.")
        exit()

    os.makedirs("kategori", exist_ok=True)

    all_known_texts = successful_texts + unsuccessful_texts
    y_train = [1] * len(successful_texts) + [0] * len(unsuccessful_texts)

    # --- 2. ENHANCED VECTORIZATION (CHARACTER LEVEL N-GRAMS) ---
    print("Initializing Character-Level TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(3, 5),
        lowercase=False
    )
    X_train = vectorizer.fit_transform(all_known_texts)

    # --- 3. TRAIN CLASSIFIER ---
    print("Training Logistic Regression Model on Character N-grams...")
    model = LogisticRegression(class_weight='balanced', C=10)
    model.fit(X_train, y_train)

    # --- 4. CLASSIFY NEW TEXTS ---
    print("\n--- RESULTS (APPROACH 5: CHARACTER N-GRAMS) ---")
    X_test = vectorizer.transform(texts_to_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    for i, text in enumerate(texts_to_test):
        prob_unsucc, prob_succ = probabilities[i]
        if predictions[i] == 1:
            result = "SUCCESSFUL"
            confidence = prob_succ * 100
        else:
            result = "UNSUCCESSFUL"
            confidence = prob_unsucc * 100

        print(f"\nText: '{text}'")
        print(f"Classification: {result} (Confidence: {confidence:.1f}%)")

    print("\nModel's Most Important Keywords (Character Chunks):")
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_succ_indices = np.argsort(coefs)[-5:][::-1]
    print("Strongest Indicators of Success:")
    for idx in top_succ_indices:
        if coefs[idx] > 0:
            print(f"  - '{feature_names[idx]}' (Weight +{coefs[idx]:.2f})")

    top_unsucc_indices = np.argsort(coefs)[:5]
    print("Strongest Indicators of Failure:")
    for idx in top_unsucc_indices:
        if coefs[idx] < 0:
            print(f"  - '{feature_names[idx]}' (Weight {coefs[idx]:.2f})")

    # --- 5. VISUALIZATIONS ---
    print("\nGenerating visualizations for test5...")
    plt.figure(figsize=(10, 6))
    top_pos_idx = np.argsort(coefs)[-10:]
    top_neg_idx = np.argsort(coefs)[:10]

    combined_idx = np.concatenate([top_neg_idx, top_pos_idx])
    colors = ['red' if coefs[i] < 0 else 'green' for i in combined_idx]
    features = [feature_names[i] for i in combined_idx]
    importances = [coefs[i] for i in combined_idx]

    plt.barh(features, importances, color=colors)
    plt.title("Top Character N-Grams (Punctuation & Caps Included)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("kategori/5_feature_importance.png")
    print("Saved: kategori/5_feature_importance.png")

    print("Generating PCA visualization...")
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

    plt.title("Approach 5 Embedding Clusters (Character N-Grams)")
    plt.legend()
    plt.savefig("kategori/5_pca.png")
    print("Saved: test5_pca.png")

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

    with open("kategori/5_pca_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pca_data))
    print("Saved: kategori/5_pca_data.txt")

    # --- 6. GENERATE TEXT REPORT ---
    print("\nGenerating Report...")
    report = []
    report.append("\n" + "="*50)
    report.append("TEST 5 - REPORT SUMMARY")
    report.append("="*50)
    report.append("APPROACH:")
    report.append("- Method: Character-level TF-IDF Vectorizer + Logistic Regression")
    report.append("- Parameters: analyzer='char', ngram_range=(3, 5), lowercase=False")
    report.append("- Mechanics: Evaluates combinations of 3-5 characters instead of full words, preserving capital letters and punctuation. Useful for catching spam patterns (like '!!!' or 'ALL CAPS').")
    
    report.append("\nGLOBAL MODEL PROPERTIES:")
    report.append("Strongest Indicators of Success (Top 5 Positive Character Chunks):")
    for idx in top_succ_indices[:5]:
        if coefs[idx] > 0:
            report.append(f"  - '{feature_names[idx]}' (Weight +{coefs[idx]:.2f})")

    report.append("Strongest Indicators of Failure (Top 5 Negative Character Chunks):")
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

    with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("Report appended to kategori/rapor.txt")

if __name__ == "__main__":
    main()
