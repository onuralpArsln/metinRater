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

    # Clear previous results and logs
    with open("kategori/rapor.txt", "w", encoding="utf-8") as f:
        f.write("--- AGGREGATE BIG-DATA TEST LOG ---\n")

    # 1. PHASE 1: CORPUS BUILDING
    print("PHASE 1: Building Aggregate Corpus from all targets...")
    for f_path in ["grup1.txt", "grup2.txt", "successful.txt", "unsuccessful.txt"]:
        if os.path.exists(f_path): 
            os.remove(f_path)

    for target in target_files:
        target_name = os.path.basename(target)
        print(f"  > Extracting: {target_name}")
        run_command(["extractor.py", target])

        # Aggregate into group files
        try:
            for source, dest in [("successful.txt", "grup1.txt"), ("unsuccessful.txt", "grup2.txt")]:
                if os.path.exists(source):
                    with open(source, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            with open(dest, "a", encoding="utf-8") as d_file:
                                d_file.write(content + "\n")
        except Exception as e:
            print(f"Error during aggregation: {e}")

    # Prepare final training sets for Phase 2
    import shutil
    if os.path.exists("grup1.txt"): shutil.copy("grup1.txt", "successful.txt")
    if os.path.exists("grup2.txt"): shutil.copy("grup2.txt", "unsuccessful.txt")

    # 2. PHASE 2: AGGREGATE TESTING
    print("\nPHASE 2: Testing Headlines against Aggregate Corpus...")
    
    # Read original headlines
    if not os.path.exists("test_texts.txt"):
        print("Error: test_texts.txt not found.")
        return
    with open("test_texts.txt", "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]

    all_data = []
    test_scripts = ["test1.py", "test2.py", "test3.py", "test4.py", "test5.py", "test6.py", "test8.py", "test7.py"]

    for headline in headlines:
        print(f"  > Testing Global Performance: {headline[:50]}...")
        
        # Write only this headline for scripts to read
        with open("test_texts.txt", "w", encoding="utf-8") as f:
            f.write(headline)
        
        row_data = {"Headline": headline, "Target": "AGGREGATE_POOL"}
        
        for i, script in enumerate(test_scripts):
            script_num = script.replace("test", "").replace(".py", "")
            if os.path.exists(script):
                output = run_command([script])
                with open("kategori/rapor.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- HEADLINE: {headline} | SCRIPT: {script} ---\n")
                    f.write(output)
                
                sdata = extract_scores_from_output(output)
                score_val = sdata.get('confidence') or sdata.get('success_sim') or "N/A"
                verdict = sdata.get('verdict', "?")
                row_data[f"Test{script_num}_Score"] = f"{score_val} ({verdict[0]})"
                
                if script == "test7.py":
                    row_data["Final_Verdict"] = verdict
                    row_data["Final_Confidence"] = score_val

        all_data.append(row_data)

    # Restore original test_texts.txt
    with open("test_texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(headlines))

    # 3. OUTPUT GENERATION
    if all_data:
        # CSV
        keys = all_data[0].keys()
        with open("target_scores_summary.csv", "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_data)

        # Vote Summary TXT
        vote_score_cols = ["Test1_Score", "Test2_Score", "Test3_Score", "Test4_Score",
                           "Test5_Score", "Test6_Score", "Test8_Score", "Test7_Score"]
        with open("target_votes_summary.txt", "w", encoding="utf-8") as vf:
            vf.write("headline-Final_Verdict-Final_Confidence-unsucces_votes-succes_votes\n")
            for row in all_data:
                s_votes = sum(1 for col in vote_score_cols if row.get(col, "").endswith("(S)"))
                u_votes = sum(1 for col in vote_score_cols if row.get(col, "").endswith("(U)"))
                verdict = "SUCCESSFUL" if s_votes > u_votes else "UNSUCCESSFUL"
                vf.write(f"{row['Headline']}-{verdict}-{row.get('Final_Confidence','N/A')}-{u_votes}-{s_votes}\n")

    print("\n" + "="*50)
    print("COMPREHENSIVE AGGREGATE ANALYSIS COMPLETE")
    print("="*50)
    for d in all_data:
        print(f"TITLE: {d['Headline'][:40]}...")
        print(f"RESULT: {d.get('Final_Verdict')} | Confidence: {d.get('Final_Confidence')}%")
        print("-" * 30)


if __name__ == "__main__":
    main()
