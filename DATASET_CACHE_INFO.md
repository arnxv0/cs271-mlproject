# Dataset Caching - Smart Download System

All dataset scripts now check if data already exists before downloading or processing. This saves bandwidth and time when re-running scripts!

## How It Works

### 1. Download Scripts (01_download_inaturalist.py, 02_download_fishbase.py)

**Checks:**
- Does the species directory exist? (`data/raw/inaturalist/{species_code}/`)
- Does it have images (`.jpg` files)?
- Is the count >= target images from config.yaml?

**Behavior:**
- ✓ **If yes:** Skips download, loads existing metadata
- ✗ **If no:** Downloads images as usual

**Example output:**
```
✓ Already have 503 images for Yelloweye Rockfish (target: 500)
  Skipping download. Delete data/raw/inaturalist/yelloweye to re-download.
```

### 2. Preprocessing Script (05_preprocess.py)

**Checks:**
- Does `data/processed/merged/{species_code}/` exist?
- Does it contain processed images?

**Behavior:**
- ✓ **If yes:** Skips merging and processing
- ✗ **If no:** Merges sources, cleans, and resizes images

**Example output:**
```
✓ Already processed: 487 images found
  Skipping merge. Delete data/processed/merged/yelloweye to reprocess.
```

### 3. Split Creation Script (06_create_splits.py)

**Checks:**
- Do train/val/test directories exist for this species?
- Do they contain images?

**Behavior:**
- ✓ **If yes:** Skips split creation
- ✗ **If no:** Creates train/val/test splits

**Example output:**
```
✓ Splits already exist: Train=341, Val=73, Test=73
  Skipping. Delete data/processed/train,val,test to recreate.
```

## Force Re-download / Reprocess

If you want to start fresh or re-download specific species:

### Option 1: Delete specific species
```bash
# Re-download yelloweye rockfish
rm -rf data/raw/inaturalist/yelloweye
rm -rf data/metadata/inaturalist_yelloweye.csv

# Reprocess yelloweye rockfish
rm -rf data/processed/merged/yelloweye

# Recreate splits for yelloweye
rm -rf data/processed/train/*/yelloweye
rm -rf data/processed/val/*/yelloweye
rm -rf data/processed/test/*/yelloweye
```

### Option 2: Delete entire dataset
```bash
# Start completely fresh
rm -rf data/raw/
rm -rf data/processed/
rm -rf data/metadata/*.csv
```

### Option 3: Delete just processed data (keep downloads)
```bash
# Keep raw downloads, but reprocess everything
rm -rf data/processed/
```

## Benefits

1. **Faster reruns**: If you already have 25/30 species, only downloads 5 missing ones
2. **Resume capability**: Script crashed? Just run it again
3. **Bandwidth savings**: No re-downloading existing data
4. **Time savings**: Skip preprocessing if already done
5. **Debugging friendly**: Can selectively reprocess problematic species

## Build Script Behavior

When you run `./build_dataset.sh`, it will:

1. Check each species in the config
2. Skip species that already have sufficient images
3. Download only missing species
4. Skip preprocessing for already-processed species
5. Skip split creation if splits already exist

**First run:** Downloads everything (~30 species)
**Second run:** Skips everything (instant!)
**Partial download:** Completes only what's missing

## Checking What You Have

```bash
# Count images per species
for dir in data/raw/inaturalist/*/; do
  echo "$(basename $dir): $(ls -1 $dir | wc -l) images"
done

# Check processed images
ls -lh data/processed/merged/

# Check splits
echo "Train: $(find data/processed/train -name '*.jpg' | wc -l)"
echo "Val: $(find data/processed/val -name '*.jpg' | wc -l)"
echo "Test: $(find data/processed/test -name '*.jpg' | wc -l)"
```

## Summary

You can now safely run the scripts multiple times without worrying about:
- Re-downloading thousands of images
- Wasting bandwidth
- Waiting for preprocessing again

Just run: `./build_dataset.sh` or individual scripts - they'll automatically skip what's already done!
