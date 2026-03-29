#!/bin/bash

echo "🚀 MetinRater Installation Started..."

# 1. Check for Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ HATA: python3 bulunamadı. Lütfen Python3 kurun."
    exit 1
fi

# 2. Install/Check Pipenv
if ! command -v pipenv &> /dev/null; then
    echo "📦 Pipenv kuruluyor..."
    python3 -m pip install --user pipenv
fi

# 3. Create Virtual Environment and Install Base Dependencies
echo "🛠️ Bağımlılıklar yükleniyor (Pipenv)..."
python3 -m pipenv install

# 4. Install CPU-Only Torch (Optimized for standard PCs)
echo "⚡ CPU-Optimize Torch kuruluyor..."
python3 -m pipenv run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 5. Correct Gemini Library & Playwright
echo "🧠 Kütüphaneler kontrol ediliyor (Gemini, Playwright, vb.)..."
python3 -m pipenv run pip install google-genai playwright python-dotenv markdown
echo "🌐 Playwright tarayıcı motorları kuruluyor..."
python3 -m pipenv run playwright install

# 6. Create .env if not exists
if [ ! -f .env ]; then
    echo "📝 .env dosyası oluşturuluyor..."
    echo "# MetinRater Configuration" > .env
    echo "GEMINI_API_KEY=YOUR_KEY_HERE" >> .env
    echo "GEMINI_MODEL=gemini-2.5-flash-lite" >> .env
    echo "SCRAPE_ENABLED=false" >> .env
    echo "✅ .env oluşturuldu. (Not: Kendi API anahtarını kullanabilirsin)"
fi

# 7. Make run_batch.sh executable
chmod +x run_batch.sh

echo "------------------------------------------------------------"
echo "✅ KURULUM TAMAMLANDI!"
echo "Şimdi './run_batch.sh' komutuyla veya masaüstü ikonuyla analizi başlatabilirsin."
echo "------------------------------------------------------------"
