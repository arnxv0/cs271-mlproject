"""Download images from FishBase for rare species"""
import sys
from pathlib import Path
import yaml
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import logging

sys.path.append(str(Path(__file__).parent))
from utils.download_utils import ImageDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FishBaseDownloader:
    """Download fish images from FishBase"""

    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config['data_sources']['fishbase']['base_url']
        self.output_dir = Path('data/raw/fishbase')
        self.metadata_dir = Path('data/metadata')
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_species_images(self, scientific_name: str) -> list:
        """
        Scrape image URLs from FishBase species page

        Args:
            scientific_name: e.g., "Sebastes gilli"

        Returns:
            List of image URLs
        """
        # Format species name for URL
        species_url_name = scientific_name.replace(' ', '-')
        url = f"{self.base_url}/summary/{species_url_name}"

        logger.info(f"Fetching from: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all image tags
            image_urls = []

            # Method 1: Find images in species photos section
            for img in soup.find_all('img', src=True):
                src = img['src']

                # Filter for actual fish photos
                if any(keyword in src.lower() for keyword in ['photos', 'images', 'pictures']):
                    if not src.startswith('http'):
                        src = self.base_url + src
                    image_urls.append(src)

            # Method 2: Look for links to photo gallery
            photo_links = soup.find_all('a', href=True)
            for link in photo_links:
                href = link['href']
                if 'photo' in href.lower() or 'image' in href.lower():
                    if not href.startswith('http'):
                        href = self.base_url + href

                    # Fetch gallery page
                    try:
                        gallery_response = requests.get(href, timeout=10)
                        gallery_soup = BeautifulSoup(gallery_response.content, 'html.parser')

                        for img in gallery_soup.find_all('img', src=True):
                            src = img['src']
                            if not src.startswith('http'):
                                src = self.base_url + src
                            if src not in image_urls:
                                image_urls.append(src)

                        time.sleep(1)  # Be nice to servers

                    except:
                        continue

            return list(set(image_urls))  # Remove duplicates

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return []

    def download_species(self, species_info: dict) -> pd.DataFrame:
        """Download images for a single species from FishBase"""
        scientific_name = species_info['scientific']
        common_name = species_info['common']
        code = species_info['code']
        target = species_info['target_images']

        logger.info(f"\nDownloading from FishBase: {common_name}")

        # Create output directory
        species_dir = self.output_dir / code
        species_dir.mkdir(parents=True, exist_ok=True)

        # Check if data already exists
        metadata_file = self.metadata_dir / f'fishbase_{code}.csv'
        existing_images = list(species_dir.glob('*.jpg'))

        if metadata_file.exists() and len(existing_images) >= target:
            logger.info(f"✓ Already have {len(existing_images)} images for {common_name} (target: {target})")
            logger.info(f"  Skipping download. Delete {species_dir} to re-download.")
            # Load and return existing metadata
            return pd.read_csv(metadata_file)
        elif len(existing_images) > 0:
            logger.info(f"Found {len(existing_images)} existing images, need {target - len(existing_images)} more")

        # Get image URLs
        image_urls = self.get_species_images(scientific_name)
        logger.info(f"Found {len(image_urls)} images on FishBase")

        if not image_urls:
            logger.warning(f"No images found for {common_name}")
            return pd.DataFrame()

        # Download images
        downloader = ImageDownloader(species_dir, delay=1.0)
        metadata_records = []

        for idx, url in enumerate(tqdm(image_urls[:target], desc=f"Downloading {code}")):
            filename = f"{code}_fishbase_{idx:04d}.jpg"

            success = downloader.download_image(url, filename)

            if success:
                metadata_records.append({
                    'filename': filename,
                    'species_code': code,
                    'common_name': common_name,
                    'scientific_name': scientific_name,
                    'source': 'fishbase',
                    'url': url,
                })

        logger.info(f"Downloaded {len(metadata_records)} images")

        # Save metadata
        df = pd.DataFrame(metadata_records)
        if not df.empty:
            metadata_file = self.metadata_dir / f'fishbase_{code}.csv'
            df.to_csv(metadata_file, index=False)

        return df

    def download_rare_species(self):
        """Download only species that are hard to find (like bronzespotted)"""
        # Focus on bronzespotted which is hardest to find
        rare_species = [
            species for species in self.config['species']['prohibited']
            if species['code'] == 'bronzespotted'
        ]

        all_metadata = []
        for species_info in rare_species:
            df = self.download_species(species_info)
            if not df.empty:
                all_metadata.append(df)

        if all_metadata:
            combined_df = pd.concat(all_metadata, ignore_index=True)
            combined_file = self.metadata_dir / 'fishbase_all.csv'
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"✅ Metadata saved to {combined_file}")

if __name__ == "__main__":
    downloader = FishBaseDownloader()
    downloader.download_rare_species()
