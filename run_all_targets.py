import os
import subprocess
import glob
import re
import sys
import csv

def run_command(command):
    # Use the current interpreter
    full_command = [sys.executable] + command
    print(f"Running: {' '.join(full_command)}")
    # Use errors='replace' to handle Turkish characters in stdout gracefully
    result = subprocess.run(full_command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"Error running {command[0]}: {result.stderr}")
    return result.stdout if result.stdout is not None else ""

def extract_scores_from_output(output):
    """
    Parses the terminal output of a test script to find the classification score/confidence.
    Returns a dict with found info.
    """
    scores = {}
    
    # Pattern for Test 1, 2, 4 (Similarity based)
    sim_match = re.search(r"Similarity to Success: ([\d.]+)", output)
    if sim_match:
        scores['success_sim'] = sim_match.group(1)
    
    # Pattern for Test 3, 5, 6, 7, 8 (Confidence based)
    conf_match = re.search(r"Classification:.*?\(Confidence: ([\d.]+)%\)", output)
    if not conf_match:
         # Fallback for Test 7 or slight variations
         conf_match = re.search(r"Confidence: ([\d.]+)%", output)
    
    if conf_match:
        scores['confidence'] = conf_match.group(1)
        
    class_match = re.search(r"Classification: (SUCCESSFUL|UNSUCCESSFUL|NEUTRAL)", output)
    if class_match:
        scores['verdict'] = class_match.group(1)
        
    return scores

def main():
    target_files = glob.glob("targets/*.html")
    if not target_files:
        print("No HTML files found in targets/")
        return

    # Clear previous results
    with open("kategori/rapor.txt", "w", encoding="utf-8") as f:
        f.write("--- MULTI-TARGET TEST LOG ---\n")

    # Clear group files
    for f in ["grup1.txt", "grup2.txt"]:
        if os.path.exists(f): 
            os.remove(f)

    all_data = []
    test_scripts = ["test1.py", "test2.py", "test3.py", "test4.py", "test5.py", "test6.py", "test8.py", "test7.py"]

    for target in target_files:
        target_name = os.path.basename(target)
        print(f"\nProcessing: {target_name}")
        
        # 1. Extraction
        run_command(["extractor.py", target])

        # Append to group files immediately after extraction
        try:
            with open("successful.txt", "r", encoding="utf-8") as f:
                succ_lines = f.read().strip()
                if succ_lines:
                    with open("grup1.txt", "a", encoding="utf-8") as g1:
                        g1.write(succ_lines + "\n")
            
            with open("unsuccessful.txt", "r", encoding="utf-8") as f:
                unsucc_lines = f.read().strip()
                if unsucc_lines:
                    with open("grup2.txt", "a", encoding="utf-8") as g2:
                        g2.write(unsucc_lines + "\n")
        except Exception as e:
            print(f"Error appending to group files: {e}")
        
        target_row = {"Target": target_name}
        
        # 2. Run Tests and Extract Scores
        for i, script in enumerate(test_scripts):
            script_num = script.replace("test", "").replace(".py", "")
            if os.path.exists(script):
                output = run_command([script])
                # Save full output to the main report
                with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- TARGET: {target_name} | SCRIPT: {script} ---\n")
                    f.write(output)
                
                # Extract score for CSV
                sdata = extract_scores_from_output(output)
                score_val = sdata.get('confidence') or sdata.get('success_sim') or "N/A"
                verdict = sdata.get('verdict', "?")
                target_row[f"Test{script_num}_Score"] = f"{score_val} ({verdict[0]})"
                
                if script == "test7.py":
                    target_row["Final_Verdict"] = verdict
                    target_row["Final_Confidence"] = score_val

        all_data.append(target_row)

    # 3. Save to CSV
    if all_data:
        keys = all_data[0].keys()
        with open("target_scores_summary.csv", "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_data)
        print("\nStructured data saved to target_scores_summary.csv")

    # 4. Generate Final Turkish Paragraph
    success_count = sum(1 for d in all_data if d.get("Final_Verdict") == "SUCCESSFUL")
    total = len(all_data)
    
    summary_para = f"Tam kapsamlı analiz tamamlandı. 'targets' klasöründeki {total} kategorinin tamamı için 8 farklı test algoritması koşturulmuş ve sonuçlar 'target_scores_summary.csv' dosyasına kaydedilmiştir. "
    summary_para += f"Master Ensemble modeline göre {success_count} kategoride başarı sinyali alınırken, {total - success_count} kategoride metinler yetersiz bulunmuştur. "
    summary_para += "Detaylı skor tablosu, her bir algoritmanın (TF-IDF, Karakter N-Gram, Semantik NLP vb.) ilgili hedef üzerindeki spesifik güven yüzdelerini göstermektedir."
    
    print("\n" + "-"*50)
    print(summary_para)
    print("-" * 50)
    
    with open("aggregator_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_para)

if __name__ == "__main__":
    main()
