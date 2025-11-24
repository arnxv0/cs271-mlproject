"""Preprocess and clean all collected images"""
import sys
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm
import logging

sys.path.append(str(Path(__file__).parent))
from utils.image_utils import validate_image, resize_image, get_image_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Clean and preprocess raw images"""

    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed/merged')
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.min_size = self.config['preprocessing']['min_image_size']
        self.target_size = self.config['preprocessing']['target_size']

    def get_all_species_codes(self) -> list:
        """Get list of all species codes from config"""
        codes = []
        for category in ['prohibited', 'restricted', 'legal']:
            codes.extend([s['code'] for s in self.config['species'][category]])
        return codes

    def merge_sources(self, species_code: str) -> int:
        """
        Merge images from all sources for a species

        Args:
            species_code: Species code (e.g., 'yelloweye')

        Returns:
            Number of images merged
        """
        logger.info(f"\nMerging sources for: {species_code}")

        # Output directory
        output_dir = self.processed_dir / species_code
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        existing_images = list(output_dir.glob('*.jpg'))
        if len(existing_images) > 0:
            logger.info(f"✓ Already processed: {len(existing_images)} images found")
            logger.info(f"  Skipping merge. Delete {output_dir} to reprocess.")
            return len(existing_images)

        # Sources to check
        sources = [
            ('inaturalist', self.raw_dir / 'inaturalist' / species_code),
            ('fishbase', self.raw_dir / 'fishbase' / species_code),
            ('image_library', self.raw_dir / 'image_library' / species_code),
            ('noaa', self.raw_dir / 'noaa' / species_code),
            ('fishnet', self.raw_dir / 'fishnet' / species_code),
        ]

        total_count = 0

        for source_name, source_dir in sources:
            if not source_dir.exists():
                logger.info(f"  {source_name}: not found")
                continue

            images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg'))
            count = 0

            for img_file in images:
                # Validate image
                if not validate_image(img_file, self.min_size):
                    continue

                # Create unique filename
                new_filename = f"{species_code}_{source_name}_{total_count:05d}.jpg"
                new_path = output_dir / new_filename

                # Copy file
                shutil.copy(img_file, new_path)
                count += 1
                total_count += 1

            logger.info(f"  {source_name}: {count} images")

        logger.info(f"  TOTAL: {total_count} images")
        return total_count

    def clean_images(self, species_code: str) -> dict:
        """
        Remove invalid/corrupted images

        Args:
            species_code: Species code

        Returns:
            Statistics dict
        """
        logger.info(f"\nCleaning images for: {species_code}")

        species_dir = self.processed_dir / species_code
        if not species_dir.exists():
            logger.warning(f"Directory not found: {species_dir}")
            return {}

        stats = get_image_stats(species_dir)
        logger.info(f"  Total: {stats['total_images']}")
        logger.info(f"  Valid: {stats['valid_images']}")
        logger.info(f"  Invalid: {stats['invalid_images']}")
        logger.info(f"  Avg size: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")

        return stats

    def resize_images(self, species_code: str):
        """
        Resize all images to standard size

        Args:
            species_code: Species code
        """
        logger.info(f"\nResizing images for: {species_code}")

        species_dir = self.processed_dir / species_code
        if not species_dir.exists():
            return

        resized_count = 0
        for img_file in tqdm(list(species_dir.glob('*.jpg')), desc="Resizing"):
            resized = resize_image(img_file, self.target_size, keep_aspect=True)

            if resized is not None:
                import cv2
                cv2.imwrite(str(img_file), resized)
                resized_count += 1

        logger.info(f"  Resized {resized_count} images to max {self.target_size}px")

    def process_all(self):
        """Process all species"""
        species_codes = self.get_all_species_codes()

        logger.info("="*60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("="*60)

        summary = {}

        for code in species_codes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {code}")
            logger.info(f"{'='*60}")

            # Step 1: Merge sources
            count = self.merge_sources(code)

            # Step 2: Clean
            stats = self.clean_images(code)

            # Step 3: Resize
            self.resize_images(code)

            summary[code] = {
                'total_images': count,
                'valid_images': stats.get('valid_images', 0),
            }

        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)

        total = 0
        for code, stats in summary.items():
            logger.info(f"{code}: {stats['valid_images']} images")
            total += stats['valid_images']

        logger.info(f"\nTOTAL: {total} images")
        logger.info("\n✅ Preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    preprocessor.process_all()
