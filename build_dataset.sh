#!/bin/bash

echo "======================================"
echo "California Rockfish Dataset Builder"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Step 1: Download from iNaturalist"
echo "======================================"
echo "Note: Will skip species with existing data"
python scripts/01_download_inaturalist.py

echo ""
echo "======================================"
echo "Step 2: Download from FishBase (rare species)"
echo "======================================"
echo "Note: Will skip species with existing data"
python scripts/02_download_fishbase.py

echo ""
echo "======================================"
echo "Step 3: Preprocess all images"
echo "======================================"
echo "Note: Will skip already processed images"
python scripts/05_preprocess.py

echo ""
echo "======================================"
echo "Step 4: Create train/val/test splits"
echo "======================================"
echo "Note: Will skip if splits already exist"
python scripts/06_create_splits.py

echo ""
echo "✅ Dataset build complete!"
echo ""
echo "Dataset location: data/processed/"
echo "Metadata location: data/metadata/"
echo ""
echo "Next steps:"
echo "1. Review data/metadata/dataset_stats.json"
echo "2. Check data/metadata/class_distribution.json"
echo "3. Start training your model!"
