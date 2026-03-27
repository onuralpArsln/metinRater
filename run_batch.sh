#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project directory
cd "$SCRIPT_DIR" || { echo "Error: Could not navigate to $SCRIPT_DIR"; exit 1; }

# Run the batch analysis using pipenv
echo "Starting batch analysis in $SCRIPT_DIR..."
python3 -m pipenv run python run_all_targets.py

echo "📊 Stage 2: Gemini AI Analysis (Blind Study)..."
python3 -m pipenv run python gemini_analiz.py

echo "✨ Stage 3: Generating Final 'Cute' Turkish Report..."
python3 -m pipenv run python rapor_olusturucu.py

echo "Analysis complete. Press enter to close."
read -r
