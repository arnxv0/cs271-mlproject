"""Download images from iNaturalist via API"""
import sys
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.download_utils import ImageDownloader, get_image_urls_from_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pyinaturalist import get_observations
except ImportError:
    logger.error("pyinaturalist not installed. Run: pip install pyinaturalist")
    sys.exit(1)

class iNaturalistDownloader:
    """Download fish images from iNaturalist"""

    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path('data/raw/inaturalist')
        self.metadata_dir = Path('data/metadata')
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def download_species(self, species_info: dict) -> pd.DataFrame:
        """
        Download images for a single species

        Args:
            species_info: Species configuration dict

        Returns:
            DataFrame with metadata
        """
        scientific_name = species_info['scientific']
        common_name = species_info['common']
        code = species_info['code']
        target = species_info['target_images']

        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {common_name} ({scientific_name})")
        logger.info(f"Target: {target} images")
        logger.info(f"{'='*60}")

        # Create output directory
        species_dir = self.output_dir / code
        species_dir.mkdir(parents=True, exist_ok=True)

        # Check if data already exists
        metadata_file = self.metadata_dir / f'inaturalist_{code}.csv'
        existing_images = list(species_dir.glob('*.jpg'))

        if metadata_file.exists() and len(existing_images) >= target:
            logger.info(f"✓ Already have {len(existing_images)} images for {common_name} (target: {target})")
            logger.info(f"  Skipping download. Delete {species_dir} to re-download.")
            # Load and return existing metadata
            return pd.read_csv(metadata_file)
        elif len(existing_images) > 0:
            logger.info(f"Found {len(existing_images)} existing images, need {target - len(existing_images)} more")
            # Continue to download more images

        # Get observations from iNaturalist
        try:
            observations = get_observations(
                taxon_name=scientific_name,
                place_id=self.config['data_sources']['inaturalist']['place_id'],
                photos=True,
                quality_grade=self.config['data_sources']['inaturalist']['quality_grade'],
                per_page=self.config['data_sources']['inaturalist']['per_page'],
                pages=self.config['data_sources']['inaturalist']['max_pages']
            )
        except Exception as e:
            logger.error(f"Failed to get observations: {e}")
            return pd.DataFrame()

        # Extract metadata and download images
        metadata_records = []
        downloader = ImageDownloader(species_dir)

        count = 0
        for obs in tqdm(observations.get('results', []), desc=f"Processing {code}"):
            if count >= target:
                break

            obs_id = obs['id']

            for photo_idx, photo in enumerate(obs.get('photos', [])):
                if count >= target:
                    break

                try:
                    # Get medium-sized image
                    img_url = photo.get('url', '').replace('square', 'medium')
                    if not img_url:
                        continue

                    # Create filename
                    filename = f"{code}_{obs_id}_{photo_idx:02d}.jpg"

                    # Download
                    success = downloader.download_image(img_url, filename)

                    if success:
                        # Save metadata
                        metadata_records.append({
                            'filename': filename,
                            'species_code': code,
                            'common_name': common_name,
                            'scientific_name': scientific_name,
                            'source': 'inaturalist',
                            'observation_id': obs_id,
                            'latitude': obs.get('latitude'),
                            'longitude': obs.get('longitude'),
                            'observed_date': obs.get('observed_on'),
                            'license': obs.get('license_code'),
                            'url': img_url,
                        })
                        count += 1

                except Exception as e:
                    logger.error(f"Error processing photo: {e}")
                    continue

        logger.info(f"Downloaded {count} images for {common_name}")

        # Save metadata
        df = pd.DataFrame(metadata_records)
        if not df.empty:
            metadata_file = self.metadata_dir / f'inaturalist_{code}.csv'
            df.to_csv(metadata_file, index=False)
            logger.info(f"Metadata saved to {metadata_file}")

        return df

    def download_all(self):
        """Download all species from config"""
        all_metadata = []

        for category in ['prohibited', 'restricted', 'legal']:
            logger.info(f"\n{'#'*60}")
            logger.info(f"CATEGORY: {category.upper()}")
            logger.info(f"{'#'*60}\n")

            for species_info in self.config['species'][category]:
                df = self.download_species(species_info)
                if not df.empty:
                    df['category'] = category
                    all_metadata.append(df)

        # Combine all metadata
        if all_metadata:
            combined_df = pd.concat(all_metadata, ignore_index=True)
            combined_file = self.metadata_dir / 'inaturalist_all.csv'
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"\n✅ All metadata saved to {combined_file}")

            # Print summary
            logger.info("\n" + "="*60)
            logger.info("DOWNLOAD SUMMARY")
            logger.info("="*60)
            summary = combined_df.groupby('species_code').size()
            for species, count in summary.items():
                logger.info(f"{species}: {count} images")
            logger.info(f"TOTAL: {len(combined_df)} images")

if __name__ == "__main__":
    downloader = iNaturalistDownloader()
    downloader.download_all()
