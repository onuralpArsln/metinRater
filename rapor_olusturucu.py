import os
import csv
import markdown

def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def generate_headline_section(row):
    """Creates a styled card for each analyzed headline."""
    headline = row.get('Headline', 'Bilinmeyen Başlık')
    verdict = row.get('Final_Verdict', 'N/A')
    confidence = row.get('Final_Confidence', '0')
    
    # Define test score columns
    test_columns = ['Test1_Score', 'Test2_Score', 'Test3_Score', 'Test4_Score', 'Test5_Score', 'Test6_Score', 'Test8_Score', 'Test7_Score']
    test_names = {
        'Test1_Score': 'Kelime Frekansı (TF-IDF)',
        'Test2_Score': 'Bigram (Kelime İkilileri)',
        'Test3_Score': 'AI Anahtar Kelime Ağırlığı',
        'Test4_Score': 'Derin Semantik Analiz',
        'Test5_Score': 'Karakter N-Gram',
        'Test6_Score': 'Noktalama İşaretleri',
        'Test8_Score': 'Semantik SVM (Yüksek Boyutlu)',
        'Test7_Score': 'Master Ensemble (Nihai Karar)'
    }
    
    test_rows_html = ""
    for col in test_columns:
        score = row.get(col, 'N/A')
        # Determine color based on (S) or (U)
        badge_class = "badge-success" if "(S)" in str(score) else "badge-danger" if "(U)" in str(score) else "badge-neutral"
        test_rows_html += f"""
            <div class="test-item">
                <span class="test-name">{test_names.get(col, col)}</span>
                <span class="test-score {badge_class}">{score}</span>
            </div>
        """

    verdict_class = "status-success" if verdict == "SUCCESSFUL" else "status-danger"
    
    return f"""
    <div class="card">
        <div class="card-header">
            <h3 class="headline-title">{headline}</h3>
            <span class="verdict-tag {verdict_class}">{verdict} (%{confidence})</span>
        </div>
        <div class="test-grid">
            {test_rows_html}
        </div>
    </div>
    """

def main():
    print("Rapor Oluşturuluyor...")
    
    # 1. Load Data
    gemini_md = read_file("kategori/gemini_yaniti.txt")
    gemini_html = markdown.markdown(gemini_md) if gemini_md else "Analiz henüz hazır değil."
    
    scores_data = []
    if os.path.exists("target_scores_summary.csv"):
        with open("target_scores_summary.csv", mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores_data.append(row)

    # 2. Build the HTML Structure
    headline_cards = "".join([generate_headline_section(row) for row in scores_data])
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MetinRater Elite Analysis</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --primary: #6366f1;
                --success: #22c55e;
                --danger: #ef4444;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: var(--text);
                margin: 0;
                padding: 40px 20px;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            
            header {{
                text-align: center;
                margin-bottom: 50px;
            }}
            
            h1 {{
                font-size: 3rem;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            .subtitle {{
                color: var(--text-muted);
                font-size: 1.1rem;
                margin-top: 10px;
            }}
            
            .section-title {{
                font-size: 1.5rem;
                font-weight: 700;
                margin: 40px 0 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .card {{
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }}
            
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 25px;
                gap: 20px;
            }}
            
            .headline-title {{
                font-size: 1.25rem;
                margin: 0;
                line-height: 1.4;
            }}
            
            .verdict-tag {{
                padding: 6px 16px;
                border-radius: 100px;
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                white-space: nowrap;
            }}
            
            .status-success {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }}
            .status-danger {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
            
            .test-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 15px;
            }}
            
            .test-item {{
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 12px 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.9rem;
            }}
            
            .test-name {{ color: var(--text-muted); }}
            
            .badge-success {{ color: #4ade80; font-weight: 600; }}
            .badge-danger {{ color: #f87171; font-weight: 600; }}
            .badge-neutral {{ color: var(--text-muted); }}
            
            .ai-insight {{
                line-height: 1.7;
                color: #e2e8f0;
            }}
            
            .ai-insight h1, .ai-insight h2, .ai-insight h3 {{
                color: #818cf8;
                margin-top: 30px;
            }}
            
            .ai-insight code {{
                background: rgba(255, 255, 255, 0.1);
                color: #fce7f3;
                padding: 2px 6px;
                border-radius: 4px;
            }}
            
            footer {{
                text-align: center;
                padding: 40px;
                color: var(--text-muted);
                font-size: 0.9rem;
            }}
            
            @media (max-width: 768px) {{
                .card-header {{ flex-direction: column; }}
                h1 {{ font-size: 2.2rem; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>MetinRater Elite</h1>
                <div class="subtitle">Yapay Zeka Destekli Başlık Analizi & Skor Paneli</div>
            </header>
            
            <div class="section-title">📊 Analiz Edilen Başlıklar</div>
            {headline_cards}
            
            <div class="section-title">🧠 Gemini  Analizi & Öneriler</div>
            <div class="card">
                <div class="ai-insight">
                    {gemini_html}
                </div>
            </div>
            
            <footer>
                &copy; 2026 MetinRater AI Dashboard
            </footer>
        </div>
    </body>
    </html>
    """
    
    # 3. Save the Report
    os.makedirs("kategori", exist_ok=True)
    report_path = "kategori/nihai_rapor.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"\n✅ Elite HTML raporu başarıyla oluşturuldu: {report_path}")

if __name__ == "__main__":
    main()
