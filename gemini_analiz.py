import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_actual_key_here":
    print("HATA: .env dosyasında geçerli bir GEMINI_API_KEY bulunamadı!")
    exit(1)

genai.configure(api_key=api_key)

def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def main():
    print("Gemini 'Kör Test' Analizi Başlatılıyor...")

    # Load data
    grup_a = read_file("grup1.txt")
    grup_b = read_file("grup2.txt")
    test_texts = read_file("test_texts.txt")
    
    if not grup_a or not grup_b:
        print("HATA: Analiz için grup dosyaları bulunamadı.")
        return

    # Create the prompt (Turkish, Blind Study)
    prompt = f"""
Sana iki farklı ürün ismi grubu gönderiyorum: Grup A ve Grup B. 
Ayrıca, test etmek istediğimiz yeni bir metin listesi var.

---
GRUP A ÖRNEKLERİ:
{grup_a}

---
GRUP B ÖRNEKLERİ:
{grup_b}

---
TEST EDİLECEK METİNLER:
{test_texts}

---
GÖREVLERİN (Lütfen Türkçe yanıtla):
1. Grup A ve Grup B arasındaki dilbilimsel ve yapısal farkları analiz et (Hangi grup daha teknik, hangisi daha duygusal, kelime uzunlukları, büyük harf kullanımı vb.).
2. Test edilecek metinlerin neden bu gruplardan birine daha yakın olduğunu (skorlarımıza göre öyle görünüyor) mantıksal olarak açıkla.
3. Test metinlerini, 'Grup A' nın tarzına (genetiğine) daha uygun hale getirecek 3 adet yeni ve iyileştirilmiş başlık öner.
4. Lütfen "Başarılı" veya "Başarısız" kelimelerini kullanma, sadece "Grup A Tarzı" veya "Grup B Tarzı" şeklinde tarafsız bir analiz yap.

Yanıtını detaylı ama anlaşılır bir şekilde ver.
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        os.makedirs("kategori", exist_ok=True)
        with open("kategori/gemini_yaniti.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Gemini analizi tamamlandı ve kategori/gemini_yaniti.txt dosyasına kaydedildi.")
    except Exception as e:
        print(f"Gemini API hatası: {e}")

if __name__ == "__main__":
    main()
