#!/usr/bin/env python3
"""
Flatten dataset structure for PyTorch ImageFolder

Converts:
  train/legal/black/*.jpg        →  train_flat/black/*.jpg
  train/prohibited/cowcod/*.jpg  →  train_flat/cowcod/*.jpg

This allows ImageFolder to treat each species as a separate class.
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_split(source_dir: Path, dest_dir: Path):
    """Flatten one split (train/val/test)"""

    if not source_dir.exists():
        logger.warning(f"Directory not found: {source_dir}")
        return

    # Remove old flat directory if it exists
    if dest_dir.exists():
        logger.info(f"Removing existing {dest_dir}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True)

    # Iterate through categories (legal, prohibited, restricted)
    species_count = 0
    image_count = 0

    for category_dir in source_dir.iterdir():
        if not category_dir.is_dir():
            continue

        logger.info(f"\nProcessing category: {category_dir.name}")

        # Iterate through species within category
        for species_dir in category_dir.iterdir():
            if not species_dir.is_dir():
                continue

            species_name = species_dir.name
            logger.info(f"  Species: {species_name}")

            # Create species directory in flat structure
            dest_species_dir = dest_dir / species_name
            dest_species_dir.mkdir(exist_ok=True)

            # Copy all images
            images = list(species_dir.glob('*.jpg'))
            for img in images:
                shutil.copy(img, dest_species_dir / img.name)

            logger.info(f"    Copied {len(images)} images")
            species_count += 1
            image_count += len(images)

    logger.info(f"\n✓ {source_dir.name}: {species_count} species, {image_count} images")


def main():
    logger.info("="*60)
    logger.info("FLATTENING DATASET STRUCTURE")
    logger.info("="*60)

    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'

    # Flatten each split
    for split in ['train', 'val', 'test']:
        source = processed_dir / split
        dest = processed_dir / f'{split}_flat'

        logger.info(f"\n{'='*60}")
        logger.info(f"Flattening {split}...")
        logger.info(f"{'='*60}")

        flatten_split(source, dest)

    logger.info("\n" + "="*60)
    logger.info("✅ FLATTENING COMPLETE")
    logger.info("="*60)
    logger.info("\nNew structure:")
    logger.info("  data/processed/train_flat/")
    logger.info("  data/processed/val_flat/")
    logger.info("  data/processed/test_flat/")
    logger.info("\nUpdate your training script to use these directories!")


if __name__ == "__main__":
    main()
