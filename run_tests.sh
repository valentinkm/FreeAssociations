#!/bin/bash

# A robust test script for all functionalities of the main.py pipeline.
# It ensures all commands are run sequentially and stops if any command fails.
set -e

# --- Configuration ---
# You can change this to test with a different primary model
TEST_MODEL="gpt-4o"
echo "--- Starting Full Test Suite for model: $TEST_MODEL ---"
echo "----------------------------------------------------"

# --- Test 1: Extract Profiles ---
echo "\n[1/6] Testing: extract-profiles"
python main.py extract-profiles --top-countries=3 -v

# --- Test 2: prompt sweep and steering comparison ---
echo "\n[2/6] Testing: generate --type=prompt-sweep"
python main.py generate --type=prompt-sweep --model=$TEST_MODEL --ncues=2 --nsets=1 -v

echo "\n[3/6] Testing: generate --type=yoked"
python main.py generate --type=yoked --model=$TEST_MODEL --ncues=2 --nsets=5 -v

# --- Test 3: Generalize Commands (includes lexicon generation) ---
echo "\n[4/6] Testing: generalize --type=spp"
python main.py generalize --type=spp --model=$TEST_MODEL --ncues=5 --nsets=2 -v

echo "\n[5/6] Testing: generalize --type=3tt"
python main.py generalize --type=3tt --model=$TEST_MODEL --ncues=5 --nsets=2 -v

# --- Test 4: Evaluate Commands ---
echo "\n[6/6] Testing: evaluate command"
echo "--> Evaluating prompt sweep..."
python main.py evaluate --type=prompt-sweep -v

echo "--> Evaluating yoked steering..."
python main.py evaluate --type=yoked-steering -v

echo "--> Evaluating model alignment..."
python main.py evaluate --type=model-alignment --models $TEST_MODEL llama3 gemini-2.5-flash --ncues=5 --nsets=2 -v

# echo "--> Evaluating model comparison on SPP..."
# python main.py evaluate --type=model-comparison --task=spp -v

# echo "--> Evaluating model comparison on 3TT..."
# python main.py evaluate --type=model-comparison --task=3tt -v

# --- Completion ---
echo "\n[6/6] All tests completed successfully."
echo "----------------------------------------------------"

