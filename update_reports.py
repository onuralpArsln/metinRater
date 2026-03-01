import os
import re

replacements = {
    "test1.py": """    report.append("YAKLAŞIM (Test 1):")
    report.append("* Odak Noktası: Sadece kelimelerin sıklığına (frekansına) bakar.")
    report.append("* Nasıl Çalışır: Hangi kelimenin kaç defa geçtiğini sayar.")
    report.append("* Neye Bakmaz: Anlama, kelime sırasına, büyük/küçük harfe, noktalama işaretlerine.")
    report.append("* Sonuç Ne İfade Eder: Eski metinlerle içerdiği ortak kelime sayısının ve sıklığının yüzdesidir.")""",

    "test2.py": """    report.append("YAKLAŞIM (Test 2):")
    report.append("* Odak Noktası: Hem tek tek kelimelerin sıklığına hem de kısmi kelime sırasına (yan yana gelen 2 kelimeye) bakar.")
    report.append("* Nasıl Çalışır: Türkçe bağlaçları temizler ve ikili kelimelerin yan yana gelme sıklığına bakar.")
    report.append("* Neye Bakmaz: Cümlenin genel anlamı, büyük/küçük harf, noktalama.")
    report.append("* Sonuç Ne İfade Eder: Yeni metnin geçmişteki metinlerle ne kadar fazla ikili kelime kalıbı paylaştığını gösterir.")""",

    "test3.py": """    report.append("YAKLAŞIM (Test 3):")
    report.append("* Odak Noktası: Kelimelerin frekansı önemlidir, kelime sırası önemlidir. Anlam önemli değildir.")
    report.append("* Nasıl Çalışır: Yapay Zeka (Logistic Regression) kullanır. Hangi kelimenin geçmesi başarı şansını ne kadar artırıyor, bunu hesaplar.")
    report.append("* Sonuç Ne İfade Eder: Sadece benzerlik değil, Net Matematiksel Olasılıktır (Confidence %).")""",

    "test4.py": """    report.append("YAKLAŞIM (Test 4):")
    report.append("* Odak Noktası: Sadece cümlenin GENEL ANLAMINA ve BAĞLAMINA bakar. Kelime frekansı, sırası önemsizdir.")
    report.append("* Nasıl Çalışır: Çok dilli bir dil modeli (Sentence Transformer) kullanır.")
    report.append("* Neye Bakmaz: Hangi kelimenin kaç defa geçtiğine (kelime sayımı yapmaz).")
    report.append("* Sonuç Ne İfade Eder: Olasılık değil, Anlamsal Mesafe (Semantic Similarity) verir.")""",

    "test5.py": """    report.append("YAKLAŞIM (Test 5):")
    report.append("* Odak Noktası: Sadece harf parçalarının (3-5 harflik) sıklığına ve harf büyüklüğüne bakar.")
    report.append("* Nasıl Çalışır: Karakter gruplarının geçme sıklığını hesaplar (Örn: 'HAR', 'ARİ', 'İKA').")
    report.append("* Neye Bakmaz: Metnin diline, anlamına, mantığına.")
    report.append("* Sonuç Ne İfade Eder: Bir metnin biçimsel (format) olarak geçmiştekilere ne kadar benzediğini gösterir.")""",

    "test6.py": """    report.append("YAKLAŞIM (Test 6):")
    report.append("* Odak Noktası: Tam kelimelerin ve özel olarak noktalama işareti dizilerinin frekansına bakar.")
    report.append("* Nasıl Çalışır: '???' grubunu, 'ürün' kelimesi gibi tek başına anlamlı bir kelime olarak sayar.")
    report.append("* Sonuç Ne İfade Eder: Yazarın tarzını (noktalama alışkanlıklarını) puanlar.")""",

    "test7.py": """    report.append("YAKLAŞIM (Test 7 - Master Ensemble):")
    report.append("* Odak Noktası: Doğrudan metinlere bakmaz. Test 1'den 6'ya kadar olan sonuçları birleştirir.")
    report.append("* Nasıl Çalışır: Hangi testin daha güvenilir sonuçlar verdiğini öğrenen bir 'Meta' Yapay Zeka kullanır.")
    report.append("* Sonuç Ne İfade Eder: Tüm algoritmaların ortaklaşa ürettiği nihai Güven Skorudur (Confidence %).")""",

    "test8.py": """    report.append("YAKLAŞIM (Test 8 - Semantik SVM):")
    report.append("* Odak Noktası: Sadece cümlenin GENEL ANLAMINA ve BAĞLAMINA bakar, hassas bir sınır çizer.")
    report.append("* Nasıl Çalışır: Metinleri 384 boyutlu vektörlere çevirip 'Destek Vektör Makineleri' (SVM) ile kesin bir matematiksel sınır çeker.")
    report.append("* Neye Bakmaz: Kelimelerin frekansına veya sırasına.")
    report.append("* Sonuç Ne İfade Eder: SVM modelinin sınırlarının hangi tarafına düştüğünü gösteren Olasılık Skorudur (Confidence %).")"""
}

import re

for filename, new_block in replacements.items():
    if not os.path.exists(filename):
        continue
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to find the block starting with report.append("APPROACH:")
    # and ending right before report.append("\nGLOBAL MODEL PROPERTIES:") 
    # OR report.append("\nWEIGHT OF INDIVIDUAL TESTS:") 
    # OR report.append("\nRESULTS FOR TEST TEXTS:")
    
    # Regex pattern to match the APPROACH block
    pattern = r'(report\.append\("APPROACH:"\).*?)(?=report\.append\("\\nGLOBAL MODEL PROPERTIES:"\)|report\.append\("GLOBAL MODEL PROPERTIES:"\)|report\.append\("\\nWEIGHT OF INDIVIDUAL TESTS:"\)|report\.append\("WEIGHT OF INDIVIDUAL TESTS:"\)|report\.append\("\\nRESULTS FOR TEST TEXTS:"\)|report\.append\("RESULTS FOR TEST TEXTS:"\))'
    
    new_content = re.sub(pattern, new_block + '\n    ', content, flags=re.DOTALL)
    
    # Check if we successfully modified anything
    if new_content == content:
        print(f"Failed to replace in {filename}")
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully updated {filename}")
