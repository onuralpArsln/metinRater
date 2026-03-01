#!/bin/bash

echo "Extracting from target.html"
python3 -m pipenv run python extractor.py

echo "Starting all tests..."

echo "==============="
echo "Running test1.py"
echo "==============="
python3 -m pipenv run python test1.py

echo -e "\n==============="
echo "Running test2.py"
echo "==============="
python3 -m pipenv run python test2.py

echo -e "\n==============="
echo "Running test3.py"
echo "==============="
python3 -m pipenv run python test3.py

echo -e "\n==============="
echo "Running test4.py"
echo "==============="
python3 -m pipenv run python test4.py

echo -e "\n==============="
echo "Running test5.py"
echo "==============="
python3 -m pipenv run python test5.py

echo -e "\n==============="
echo "Running test6.py"
echo "==============="
python3 -m pipenv run python test6.py

echo -e "\n==============="
echo "Running test7.py (Ensemble Model)"
echo "==============="
python3 -m pipenv run python test7.py

echo -e "\n==============="
echo "Running test8.py (Semantic SVM Model)"
echo "==============="
python3 -m pipenv run python test8.py

echo -e "\nAll tests finished!"
