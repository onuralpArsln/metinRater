import os
import sys
import csv

# Try importing the new google-genai library
try:
    from google import genai
except ImportError:
    print("HATA: 'google-genai' kütüphanesi bulunamadı. Lütfen 'pip install google-genai' komutunu çalıştırın.")
    sys.exit(1)

from rich.console import Console
console = Console()

def send_notification(title, message):
    try:
        os.system(f'notify-send "{title}" "{message}"')
    except:
        pass

from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_actual_key_here":
    print("HATA: .env dosyasında geçerli bir GEMINI_API_KEY bulunamadı!")
    sys.exit(1)

# Initialize the client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"HATA: Gemini istemcisi başlatılamadı: {e}")
    sys.exit(1)

def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def load_scores_csv(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, mode='r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ""

def main():
    send_notification("MetinRater", "Starting Stage 2: Gemini Analysis...")
    console.print("[bold yellow]--- Stage 2: Gemini 'Kör Test' Analizi Başlatılıyor... ---[/bold yellow]")

    # Load data
    grup_a = read_file("grup1.txt")
    grup_b = read_file("grup2.txt")
    test_texts = read_file("test_texts.txt")
    scores_csv = load_scores_csv("target_scores_summary.csv")
    
    if not grup_a or not grup_b:
        print("HATA: Analiz için grup dosyaları (grup1.txt, grup2.txt) bulunamadı.")
        return

    # Test Descriptions for Gemini
    test_descs = """
Sana ileteceğim skorlar şu 8 testin sonucudur:
1. Test 1 (TF-IDF): Kelime kullanım sıklığı ve nadirliği. Metnin kelime haznesinin Grup A veya B ile olan uyumuna bakar. 
2. Test 2 (Bigram): Kelime ikilileri (kalıplar). Kelimelerin yan yana geliş sırasına odaklanır.
3. Test 3 (LR Keywords): Başarıyı tetikleyen kritik anahtar kelimelerin matematiksel ağırlıklandırılması.
4. Test 4 (Semantics): Metnin genel anlamı ve bağlamı. Kelimeler farklı olsa bile aynı konuyu ifade edip etmediğine bakar.
5. Test 5 (Char N-grams): Karakter dizilimleri ve ek/kök analizi. Kelime morfolojisindeki benzerliği yakalar.
6. Test 6 (Punctuation): Noktalama işaretleri ve özel karakterlerin (tire, parantez vb.) kullanım tarzı.
7. Test 8 (Semantic SVM): 384 boyutlu semantik uzayda en doğru kümelenmeyi bulan gelişmiş algoritma.
8. Test 7 (Master Ensemble): Tüm diğer testlerin (1-6 ve 8) sonuçlarını oylayarak nihai kararı veren ana kontrolcü Zeka.
"""

    # Create the prompt (Turkish, Detailed Analysis)
    prompt = f"""
Sana iki farklı ürün ismi grubu gönderiyorum: Grup A (Stil 1) ve Grup B (Stil 2). 
Ayrıca, test ettiğimiz metinlerin 8 farklı algoritmadan aldığı skorlar aşağıdadır.

---
GRUP A ÖRNEKLERİ (Stil 1):
{grup_a}

---
GRUP B ÖRNEKLERİ (Stil 2):
{grup_b}

---
TEST EDİLEN METİNLER VE SKORLARI (CSV):
{scores_csv}

---
{test_descs}

---
GÖREVLERİN (Lütfen Türkçe yanıtla):
1. Grup A ve Grup B arasındaki dilbilimsel ve yapısal farkları analiz et (tarz, tonlama, uzunluk, teknik detay vb.).
2. Skor tablosundaki her bir test için ayrı ayrı küçük çıkarımlar yap. (Örneğin: "Test 4 skoru düşük çünkü anlam olarak farklı," veya "Test 6 skoru yüksek çünkü noktalama kullanımı Grup A ile aynı").
3. Tüm test sonuçlarını birleştirerek "Total Durum" için genel bir çıkarım yap.
4. Test metinlerini, 'Grup A' nın tarzına (genetiğine) daha uygun hale getirecek 3 adet yeni ve iyileştirilmiş başlık öner.

Yanıtını detaylı ama anlaşılır bir şekilde ver. CSV'deki 'S' (Success) Grup A'yı, 'U' (Unsuccess) Grup B'yi temsil eder.
"""

    # Choose model from .env or fallback
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    try:
        with console.status("[bold green]Waiting for Gemini analysis...[/bold green]", spinner="dots"):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
        os.makedirs("kategori", exist_ok=True)
        with open("kategori/gemini_yaniti.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        console.print("✅ [bold green]Gemini detaylı analizi başarıyla tamamlandı.[/bold green]")
        send_notification("MetinRater", "Gemini Analysis Complete.")
    except Exception as e:
        print(f"❌ Gemini API hatası: {e}")

if __name__ == "__main__":
    main()
