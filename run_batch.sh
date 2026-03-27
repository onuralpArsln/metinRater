#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project directory
cd "$SCRIPT_DIR" || { echo "Error: Could not navigate to $SCRIPT_DIR"; exit 1; }

# Force CPU mode for Torch/Transformers to avoid CUDA errors
export CUDA_VISIBLE_DEVICES=""
export TRANSFORMERS_VERBOSITY=error

# Determine the correct python3 to call pipenv (prefer system python to avoid virtualenv conflicts)
SYSTEM_PYTHON=$(which python3)
if [[ "$SYSTEM_PYTHON" == *"/virtualenvs/"* ]]; then
    SYSTEM_PYTHON="/usr/bin/python3"
fi

# PHASE 1: Run the batch analysis
echo "Starting batch analysis in $SCRIPT_DIR..."
$SYSTEM_PYTHON -m pipenv run python run_all_targets.py

# PHASE 2: Gemini AI Analysis
echo "📊 Stage 2: Gemini AI Analysis (Blind Study)..."
$SYSTEM_PYTHON -m pipenv run python gemini_analiz.py

# PHASE 3: Generate the Final Elite HTML Report
echo "✨ Stage 3: Generating Final HTML Report..."
$SYSTEM_PYTHON -m pipenv run python rapor_olusturucu.py

echo "------------------------------------------------------------"
echo "✅ Analysis complete! Opening report: kategori/nihai_rapor.html"
echo "------------------------------------------------------------"

# Attempt to open the report automatically (Linux standard)
if command -v xdg-open > /dev/null; then
    xdg-open "kategori/nihai_rapor.html"
fi

echo "Press enter to close."
read -r
