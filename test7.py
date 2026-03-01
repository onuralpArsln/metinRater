import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Import the refactored test files
import test1
import test2
import test3
import test4
import test5
import test6

def load_texts(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Returning empty list.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("========================================")
    print("TEST 7 - ENSEMBLE MASTER CLASSIFIER")
    print("========================================")
    
    successful_texts = load_texts("successful.txt")
    unsuccessful_texts = load_texts("unsuccessful.txt")
    texts_to_test = load_texts("test_texts.txt")

    if not successful_texts or not unsuccessful_texts:
        print("Error: Need both successful and unsuccessful texts to continue.")
        exit()

    all_known_texts = successful_texts + unsuccessful_texts
    combined_texts = all_known_texts + texts_to_test
    
    y_train = [1] * len(successful_texts) + [0] * len(unsuccessful_texts)

    print("\nRunning Test 1 (Basic Word Frequency)...")
    s1 = test1.run_test(combined_texts)
    
    print("Running Test 2 (Word & Bigram Frequencies)...")
    s2 = test2.run_test(combined_texts)
    
    print("Running Test 3 (AI Keyword Weighting)...")
    s3 = test3.run_test(combined_texts)
    
    print("Running Test 4 (Deep Semantic Analysis - Please wait)...")
    s4 = test4.run_test(combined_texts)
    
    print("Running Test 5 (Character Patterns)...")
    s5 = test5.run_test(combined_texts)
    
    print("Running Test 6 (Custom Punctuation Tokens)...")
    s6 = test6.run_test(combined_texts)

    # Build Feature Matrix
    print("\nAggregating scores into feature matrix...")
    X = np.column_stack([s1, s2, s3, s4, s5, s6])
    
    # Split back into training and testing chunks
    num_train = len(all_known_texts)
    X_train = X[:num_train]
    X_test = X[num_train:]

    print("Training the Master Ensemble Classifier (Logistic Regression)...")
    # Using Logistic Regression. C=1 to avoid overfitting perfectly on itself.
    ensemble = LogisticRegression(class_weight='balanced', C=1.0)
    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)

    print("\n--- RESULTS FOR TEST TEXTS (ENSEMBLE) ---")
    for i, text in enumerate(texts_to_test):
        prob_unsucc, prob_succ = probabilities[i]
        
        if predictions[i] == 1:
            result = "SUCCESSFUL"
            confidence = prob_succ * 100
        else:
            result = "UNSUCCESSFUL"
            confidence = prob_unsucc * 100
            
        print(f"\nText: '{text}'")
        print(f"Master Classification: {result} (Confidence: {confidence:.1f}%)")
        print(f"Underlying Test Scores -> T1:{s1[i+num_train]:.2f}, T2:{s2[i+num_train]:.2f}, T3:{s3[i+num_train]:.2f}, T4:{s4[i+num_train]:.2f}, T5:{s5[i+num_train]:.2f}, T6:{s6[i+num_train]:.2f}")

    # Generate Feature Importance for the Ensemble
    print("\nGenerating visualizations...")
    coefs = ensemble.coef_[0]
    test_names = ['Test 1 (TF-IDF)', 'Test 2 (Bigrams)', 'Test 3 (LR Keywords)', 'Test 4 (Semantics)', 'Test 5 (Char N-grams)', 'Test 6 (Punctuation)']
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in coefs]
    plt.barh(test_names, coefs, color=colors)
    plt.title("Test 7: Which Tests are the Most Trustworthy?")
    plt.xlabel("Ensemble Classifier Weight (Higher = More Trusted)")
    plt.tight_layout()
    plt.savefig("kategori/7_ensemble_weights.png")
    print("Saved: kategori/7_ensemble_weights.png")

    # Generate Report
    report = []
    report.append("\n" + "="*50)
    report.append("TEST 7 - ENSEMBLE MASTER CLASSIFIER")
    report.append("="*50)
        report.append("YAKLAŞIM (Test 7 - Master Ensemble):")
    report.append("* Odak Noktası: Doğrudan metinlere bakmaz. Test 1'den 6'ya kadar olan sonuçları birleştirir.")
    report.append("* Nasıl Çalışır: Hangi testin daha güvenilir sonuçlar verdiğini öğrenen bir 'Meta' Yapay Zeka kullanır.")
    report.append("* Sonuç Ne İfade Eder: Tüm algoritmaların ortaklaşa ürettiği nihai Güven Skorudur (Confidence %).")
    report.append("\nWEIGHT OF INDIVIDUAL TESTS:")
    for i in range(len(test_names)):
        report.append(f"  - {test_names[i]}: Weight {coefs[i]:.3f} ({'Trusted' if coefs[i] > 0 else 'Ignored/Reversed'})")
        
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
        report.append(f"Master Classification: {result} (Confidence: {confidence:.1f}%)")

    os.makedirs("kategori", exist_ok=True)
    with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("Report appended to kategori/rapor.txt")


if __name__ == "__main__":
    main()
