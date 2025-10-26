#!/bin/bash
# setup.sh

echo "Setting up SDG Climate AI Project..."

# Create directories
mkdir -p sample_data assets

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Generate sample data if not exists
if [ ! -f "sample_data/sample_emissions_data.csv" ]; then
    echo "Generating sample data..."
    python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('sample_data/sample_emissions_data.csv')
print('Sample data created successfully!')
"
fi

echo "Setup complete! You can now run:"
echo "1. python climate_emissions_predictor.py"
echo "2. jupyter notebook climate_analysis.ipynb"