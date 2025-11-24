"""Create train/validation/test splits"""
import sys
from pathlib import Path
import yaml
import shutil
import json
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetSplitter:
    """Create train/val/test splits"""

    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.merged_dir = Path('data/processed/merged')
        self.output_dir = Path('data/processed')

        self.train_ratio = self.config['preprocessing']['train_ratio']
        self.val_ratio = self.config['preprocessing']['val_ratio']
        self.test_ratio = self.config['preprocessing']['test_ratio']

        # Verify ratios sum to 1
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 0.01

    def get_species_category(self, species_code: str) -> str:
        """Determine category (prohibited/restricted/legal) for species"""
        for category in ['prohibited', 'restricted', 'legal']:
            codes = [s['code'] for s in self.config['species'][category]]
            if species_code in codes:
                return category
        return 'unknown'

    def create_splits(self, species_code: str):
        """
        Create train/val/test splits for a species

        Args:
            species_code: Species code
        """
        logger.info(f"\nCreating splits for: {species_code}")

        # Get source directory
        source_dir = self.merged_dir / species_code
        if not source_dir.exists():
            logger.warning(f"Directory not found: {source_dir}")
            return

        # Check if splits already exist
        category = self.get_species_category(species_code)
        train_dir = self.output_dir / 'train' / category / species_code
        val_dir = self.output_dir / 'val' / category / species_code
        test_dir = self.output_dir / 'test' / category / species_code

        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            train_count = len(list(train_dir.glob('*.jpg')))
            val_count = len(list(val_dir.glob('*.jpg')))
            test_count = len(list(test_dir.glob('*.jpg')))
            total = train_count + val_count + test_count

            if total > 0:
                logger.info(f"✓ Splits already exist: Train={train_count}, Val={val_count}, Test={test_count}")
                logger.info(f"  Skipping. Delete data/processed/train,val,test to recreate.")
                return

        # Get all images
        images = list(source_dir.glob('*.jpg'))
        if not images:
            logger.warning(f"No images found for {species_code}")
            return

        logger.info(f"  Total images: {len(images)}")

        # Handle edge case: very few images
        if len(images) < 3:
            logger.warning(f"  Only {len(images)} image(s) - putting all in train set")
            train_imgs = images
            val_imgs = []
            test_imgs = []
        else:
            # Create splits
            train_imgs, temp_imgs = train_test_split(
                images,
                train_size=self.train_ratio,
                random_state=42,
                shuffle=True
            )

            if len(temp_imgs) < 2:
                # Not enough for val/test split
                val_imgs = temp_imgs
                test_imgs = []
            else:
                val_imgs, test_imgs = train_test_split(
                    temp_imgs,
                    train_size=self.val_ratio / (self.val_ratio + self.test_ratio),
                    random_state=42,
                    shuffle=True
                )

        logger.info(f"  Train: {len(train_imgs)}")
        logger.info(f"  Val: {len(val_imgs)}")
        logger.info(f"  Test: {len(test_imgs)}")

        # Determine category
        category = self.get_species_category(species_code)

        # Copy to split directories
        for split_name, imgs in [('train', train_imgs),
                                 ('val', val_imgs),
                                 ('test', test_imgs)]:

            dest_dir = self.output_dir / split_name / category / species_code
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy(img, dest_dir / img.name)

    def generate_metadata(self):
        """Generate dataset metadata files"""
        logger.info("\nGenerating metadata...")

        # Species mapping
        species_map = {}
        class_id = 0

        for category in ['prohibited', 'restricted', 'legal']:
            for species_info in self.config['species'][category]:
                code = species_info['code']

                species_map[code] = {
                    'class_id': class_id,
                    'scientific_name': species_info['scientific'],
                    'common_name': species_info['common'],
                    'category': category,
                    'regulation': self._get_regulation(code, category),
                }
                class_id += 1

        # Save species mapping
        metadata_dir = Path('data/metadata')
        metadata_dir.mkdir(parents=True, exist_ok=True)

        with open(metadata_dir / 'species_mapping.json', 'w') as f:
            json.dump(species_map, f, indent=2)

        logger.info(f"  Saved species mapping ({len(species_map)} species)")

        # Class distribution
        distribution = {}

        for split in ['train', 'val', 'test']:
            distribution[split] = {}
            split_dir = self.output_dir / split

            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if not category_dir.is_dir():
                        continue

                    for species_dir in category_dir.iterdir():
                        if not species_dir.is_dir():
                            continue

                        code = species_dir.name
                        count = len(list(species_dir.glob('*.jpg')))
                        distribution[split][code] = count

        with open(metadata_dir / 'class_distribution.json', 'w') as f:
            json.dump(distribution, f, indent=2)

        logger.info("  Saved class distribution")

        # Dataset statistics
        stats = {
            'total_species': len(species_map),
            'prohibited_species': len([s for s in species_map.values() if s['category'] == 'prohibited']),
            'restricted_species': len([s for s in species_map.values() if s['category'] == 'restricted']),
            'legal_species': len([s for s in species_map.values() if s['category'] == 'legal']),
            'splits': {
                split: sum(distribution[split].values())
                for split in distribution.keys()
            }
        }

        with open(metadata_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("  Saved dataset statistics")

        return stats

    def _get_regulation(self, species_code: str, category: str) -> str:
        """Get regulation text for species"""
        regulations = {
            'prohibited': {
                'yelloweye': 'Prohibited - Zero Take (Overfished)',
                'quillback': 'Prohibited - Zero Take (Overfished since 2023)',
                'bronzespotted': 'Prohibited - Zero Take',
                'cowcod': 'Prohibited - Zero Take (Critically Depleted)',
            },
            'restricted': {
                'vermilion': 'Sub-bag Limit: 2-4 fish (varies by GMA)',
                'copper': 'Sub-bag Limit: 1 fish',
                'canary': 'Restricted - Check current regulations',
            },
            'legal': {}
        }

        if category in regulations and species_code in regulations[category]:
            return regulations[category][species_code]

        if category == 'legal':
            return 'Legal - Within 10 fish rockfish limit'

        return 'See current regulations'

    def create_all_splits(self):
        """Create splits for all species"""
        logger.info("="*60)
        logger.info("CREATING TRAIN/VAL/TEST SPLITS")
        logger.info("="*60)

        # Get all species
        all_species = []
        for category in ['prohibited', 'restricted', 'legal']:
            all_species.extend([s['code'] for s in self.config['species'][category]])

        # Create splits for each species
        for code in all_species:
            self.create_splits(code)

        # Generate metadata
        stats = self.generate_metadata()

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Total species: {stats['total_species']}")
        logger.info(f"  Prohibited: {stats['prohibited_species']}")
        logger.info(f"  Restricted: {stats['restricted_species']}")
        logger.info(f"  Legal: {stats['legal_species']}")
        logger.info(f"\nDataset splits:")
        for split, count in stats['splits'].items():
            logger.info(f"  {split}: {count} images")

        logger.info("\n✅ Dataset ready for training!")
        logger.info(f"\nDataset location: {self.output_dir}")

if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.create_all_splits()
