import csv
import os

def main():
    csv_path = "target_scores_summary.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    summary_lines = []
    summary_lines.append("target-Final_Verdict-Final_Confidence-unsucces_votes-succes_votes")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row['Target']
            verdict = row['Final_Verdict']
            confidence = row['Final_Confidence']
            
            # Count votes from Test1 to Test6 and Test8 (Test 7 is the ensemble itself)
            # Test columns look like "Test1_Score", etc.
            s_votes = 0
            u_votes = 0
            
            for key in row:
                if "Test" in key and "Score" in key and key != "Test7_Score":
                    val = row[key]
                    if "(S)" in val:
                        s_votes += 1
                    elif "(U)" in val:
                        u_votes += 1
            
            line = f"{target}-{verdict}-{confidence}-{u_votes}-{s_votes}"
            summary_lines.append(line)

    output_path = "target_votes_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
