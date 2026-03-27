import os
import csv

def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def main():
    print("Nihai Rapor Oluşturuluyor...")
    
    # 1. Load Gemini Response
    gemini_output = read_file("kategori/gemini_yaniti.txt")
    if not gemini_output:
        gemini_output = "❌ Gemini analizi henüz hazır değil."

    # 2. Load Scores from CSV
    scores_data = []
    if os.path.exists("target_scores_summary.csv"):
        with open("target_scores_summary.csv", mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores_data.append(row)

    # 3. Format the Report (Turkish/Cute)
    report = []
    report.append("="*60)
    report.append("  🚀 METİNRATER: TAM KAPSAMLI ANALİZ RAPORU 🚀  ")
    report.append("="*60)
    report.append("\nAnaliz tamamlandı! İşte sonuçlarınızın özeti ve yapay zeka yorumları ✨\n")

    # Section 1: Score Table
    if scores_data:
        report.append("📊 [1] SKOR ÖZETİ (Grup Yakınlıkları)")
        report.append("-" * 30)
        # Assuming we just show the first row or summary
        for row in scores_data:
            report.append(f"📍 Hedef: {row.get('Target', 'Bilinmiyor')}")
            report.append(f"   Final Karar: {row.get('Final_Verdict', '?')}")
            report.append(f"   Güven Skoru: %{row.get('Final_Confidence', '0')}")
            report.append("-" * 20)
    
    report.append("\n" + "*"*40)
    
    # Section 2: Gemini Analysis
    report.append("🧠 [2] YAPAY ZEKA (GEMINI) DERİN ANALİZİ")
    report.append("-" * 40)
    report.append(gemini_output)
    
    report.append("\n" + "="*60)
    report.append("🎯 Bu rapor, MetinRater AI Asistanı tarafından hazırlanmıştır :)")
    report.append("="*60)

    # Save the report
    os.makedirs("kategori", exist_ok=True)
    report_path = "kategori/nihai_rapor.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\n✅ Nihai rapor başarıyla oluşturuldu: {report_path}")
    print("\n" + "="*40)
    print("\n".join(report))
    print("="*40)

if __name__ == "__main__":
    main()
