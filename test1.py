import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords


import os


def load_texts(filename):
    """Reads a file line by line and returns a list of non-empty strings."""
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def run_test(texts_to_score):
    """Programmatic entry point for Test 7. Returns a list of similarity scores."""
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")

    if not successful_texts or not unsuccessful_texts:
        return [0.0] * len(texts_to_score)

    all_known_texts = successful_texts + unsuccessful_texts

    try:
        turkish_stop_words = stopwords.words('turkish')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        turkish_stop_words = stopwords.words('turkish')

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(all_known_texts)

    succ_vectors = vectorizer.transform(successful_texts)
    succ_profile = np.asarray(np.mean(succ_vectors, axis=0))

    new_vectors = vectorizer.transform(texts_to_score)
    scores = []
    for i in range(len(texts_to_score)):
        succ_score = cosine_similarity(new_vectors[i], succ_profile)[0][0]
        scores.append(succ_score)
    return scores


def main():
    # --- 1. LOAD YOUR DATA ---
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")
    texts_to_test = load_texts("test_texts.txt")

    if not successful_texts or not unsuccessful_texts:
        print("Error: Need both successful and unsuccessful texts to continue.")
        exit()

    os.makedirs("kategori", exist_ok=True)
    # --- 3. THE LOGIC ---
    # Combine known texts to teach the vectorizer our vocabulary
    all_known_texts = successful_texts + unsuccessful_texts

    # TfidfVectorizer converts text to numbers, giving more weight to unique/important words
    # 1. Download the stopwords corpus (you only need to run this once)
    nltk.download('stopwords')
    # 2. Load the Turkish stop words into a list
    turkish_stop_words = stopwords.words('turkish')
    # 3. Pass the list to your vectorizer
    # vectorizer = TfidfVectorizer(stop_words=turkish_stop_words)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(all_known_texts)

    # Convert our known texts into mathematical vectors
    succ_vectors = vectorizer.transform(successful_texts)
    unsucc_vectors = vectorizer.transform(unsuccessful_texts)

    # Calculate the "average" vector and force it to be a standard array
    succ_profile = np.asarray(np.mean(succ_vectors, axis=0))
    unsucc_profile = np.asarray(np.mean(unsucc_vectors, axis=0))

    # --- 4. CLASSIFY NEW TEXTS ---
    print("--- RESULTS ---")
    new_vectors = vectorizer.transform(texts_to_test)

    for i, text in enumerate(texts_to_test):
        # Calculate similarity score (0.0 to 1.0) against both profiles
        succ_score = cosine_similarity(new_vectors[i], succ_profile)[0][0]
        unsucc_score = cosine_similarity(new_vectors[i], unsucc_profile)[0][0]

        # Decide the winner
        if succ_score > unsucc_score:
            result = "SUCCESSFUL"
        elif unsucc_score > succ_score:
            result = "UNSUCCESSFUL"
        else:
            result = "NEUTRAL / UNKNOWN"

        print(f"\nText: '{text}'")
        print(f"Classification: {result}")
        print(
            f"(Scores -> Success: {succ_score:.3f} | Unsuccess: {unsucc_score:.3f})")

    # --- 5. VISUALIZATION 1: PCA Scatter Plot ---
    print("\nGenerating PCA visualization...")
    all_vectors = vectorizer.transform(
        all_known_texts + texts_to_test).toarray()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vectors)

    plt.figure(figsize=(10, 7))

    # Plot Known Successful
    n_succ = len(successful_texts)
    plt.scatter(coords[:n_succ, 0], coords[:n_succ, 1],
                c='green', label='Known Success', s=100, alpha=0.6)

    # Plot Known Unsuccessful
    n_unsucc = len(unsuccessful_texts)
    plt.scatter(coords[n_succ:n_succ+n_unsucc, 0], coords[n_succ:n_succ +
                n_unsucc, 1], c='red', label='Known Unsuccess', s=100, alpha=0.6)

    # Plot Test Texts
    plt.scatter(coords[n_succ+n_unsucc:, 0], coords[n_succ+n_unsucc:,
                1], c='blue', label='Test Texts', s=150, marker='X')

    for i, text in enumerate(texts_to_test):
        plt.annotate(
            f"Test {i+1}", (coords[n_succ+n_unsucc+i, 0], coords[n_succ+n_unsucc+i, 1]))

    plt.title("Text Embedding Clusters (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("kategori/1_pca_visualization.png")
    print("Saved: pca_visualization.png")

    # Save PCA coordinates to text file
    pca_data = []
    pca_data.append("SUCCESSFUL TEXTS (GREEN):")
    for i in range(n_succ):
        pca_data.append(
            f"X: {coords[i, 0]:.3f}, Y: {coords[i, 1]:.3f} | Text: {successful_texts[i]}")

    pca_data.append("\nUNSUCCESSFUL TEXTS (RED):")
    for i in range(n_unsucc):
        idx = n_succ + i
        pca_data.append(
            f"X: {coords[idx, 0]:.3f}, Y: {coords[idx, 1]:.3f} | Text: {unsuccessful_texts[i]}")

    pca_data.append("\nTEST TEXTS (BLUE):")
    for i in range(len(texts_to_test)):
        idx = n_succ + n_unsucc + i
        pca_data.append(
            f"X: {coords[idx, 0]:.3f}, Y: {coords[idx, 1]:.3f} | Text: {texts_to_test[i]}")

    with open("kategori/1_pca_data.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pca_data))
    print("Saved: kategori/1_pca_data.txt")

    # --- 5B. VISUALIZATION: Similarity Bar Chart ---
    print("Generating Closeness Bar Chart...")
    test_labels = [f"Test {i+1}" for i in range(len(texts_to_test))]
    succ_scores = [cosine_similarity(v, succ_profile)[0][0]
                   for v in new_vectors]
    unsucc_scores = [cosine_similarity(v, unsucc_profile)[
        0][0] for v in new_vectors]

    x = np.arange(len(test_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, succ_scores, width,
           label='Similarity to SUCCESS', color='green', alpha=0.7)
    ax.bar(x + width/2, unsucc_scores, width,
           label='Similarity to FAILURE', color='red', alpha=0.7)

    ax.set_ylabel('Cosine Similarity Score')
    ax.set_title('Test Text Closeness to Word Profiles (Test 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(test_labels)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("kategori/1_similarity_bars.png")
    print("Saved: kategori/1_similarity_bars.png")

    # --- 6. VISUALIZATION 2: Feature Importance ---
    print("Generating Feature Importance visualization...")
    feature_names = vectorizer.get_feature_names_out()

    def plot_top_words(profile, title, filename, color):
        # Flatten profile to 1D
        scores = profile.flatten()
        # Get top 10 indices
        top_indices = np.argsort(scores)[-10:]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        plt.figure(figsize=(8, 6))
        plt.barh(top_words, top_scores, color=color)
        plt.title(title)
        plt.xlabel("TF-IDF Score")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved: {filename}")

    plot_top_words(succ_profile, "Top Words - Successful Profile",
                   "kategori/1_top_words_success.png", "green")
    plot_top_words(unsucc_profile, "Top Words - Unsuccessful Profile",
                   "kategori/1_top_words_unsuccess.png", "red")

    # --- 7. GENERATE TEXT REPORT ---
    print("\nGenerating Report...")
    report = []
    report.append("="*50)
    report.append("TEST 1 - REPORT SUMMARY")
    report.append("="*50)
    report.append("YAKLAŞIM (Test 1):")
    report.append(
        "* Odak Noktası: Sadece kelimelerin sıklığına (frekansına) bakar.")
    report.append("* Nasıl Çalışır: Hangi kelimenin kaç defa geçtiğini sayar.")
    report.append(
        "* Neye Bakmaz: Anlama, kelime sırasına, büyük/küçük harfe, noktalama işaretlerine.")
    report.append(
        "* Sonuç Ne İfade Eder: Eski metinlerle içerdiği ortak kelime sayısının ve sıklığının yüzdesidir.")
    report.append("\nRESULTS FOR TEST TEXTS:")

    for i, text in enumerate(texts_to_test):
        succ_score = cosine_similarity(new_vectors[i], succ_profile)[0][0]
        unsucc_score = cosine_similarity(new_vectors[i], unsucc_profile)[0][0]

        if succ_score > unsucc_score:
            result = "SUCCESSFUL"
        elif unsucc_score > succ_score:
            result = "UNSUCCESSFUL"
        else:
            result = "NEUTRAL / UNKNOWN"

        report.append(f"\n[Test Text {i+1}]")
        report.append(f"Content: '{text}'")
        report.append(f"Classification: {result}")
        report.append(
            f"Scores -> Similarity to Success: {succ_score:.3f} | Similarity to Failure: {unsucc_score:.3f}")

    report.append("\nVISUALIZATIONS GENERATED:")
    report.append(
        "- kategori/1_pca_visualization.png: PCA scatter plot mapping the text neighborhood.")
    report.append(
        "- kategori/1_similarity_bars.png: 1D closeness chart comparing success vs failure similarity.")
    report.append(
        "- kategori/1_top_words_success.png: Top 10 words contributing to the Success profile.")
    report.append(
        "- kategori/1_top_words_unsuccess.png: Top 10 words contributing to the Unsuccess profile.")
    report.append("\n")

    os.makedirs("kategori", exist_ok=True)
    print("\n".join(report))


if __name__ == "__main__":
    main()
