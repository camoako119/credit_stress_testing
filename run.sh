#!/bin/bash
set -e

echo "Starting Growth-Adjusted Mortgage Credit Stress Testing pipeline"

echo "Step 1: Prepare mortgage and FRED data"
python src/data_ingest.py

echo "Step 2: Run transition matrix and MLlib model"
python src/modelling.py

echo "Step 3: Run Streaming growth simulator"
python python src/growth_streaming.py

echo "Pipeline complete. Check the outputs/ folder for results"

