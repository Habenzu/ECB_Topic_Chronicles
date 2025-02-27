import pandas as pd
import numpy as np
from pathlib import Path
import requests
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import time
import random
from bs4 import BeautifulSoup
from pathlib import Path
import re
from tqdm import tqdm
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).resolve().parent

class Metadata(): 
    def __init__(self, path_to_file = Path(r'data\export_datamart.csv')):
        self.path_to_file = path_to_file
        self.metadata_df = pd.read_csv(self.path_to_file)
        self._prepare_metadata()

    def _prepare_metadata(self):
        self.metadata_df['when_speech'] = pd.to_datetime(self.metadata_df['when_speech'], format="%Y-%m-%d")
        self.metadata_df['file_ending'] = self.metadata_df['what_weblink'].str.extract(r'\.(\w+)$')
        self.what_type_dict = {
            'S': 'speech',
            'I': 'interview',
            'P': 'press-conference',
            'B': 'blog-post',
            'E': 'podcast',
        }
        self.metadata_df['what_type_long'] = self.metadata_df['what_type'].apply(lambda x: self.what_type_dict[x])

    def get_metadata(self):
        return self.metadata_df

class DataDownloader(): 
    def __init__(self, output_dir = FILE_DIR / 'data' / 'speeches', metadata_file = Path(r'data\export_datamart.csv')):
        self.output_dir = output_dir
        self.metadata_df = Metadata(metadata_file).get_metadata()
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _download_file(self, url, file_path):
        try:
            # time.sleep(2)
            time.sleep(abs(random.normalvariate()))
            headers = {
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'
                ])
            }
            response = requests.get(url, headers=headers, allow_redirects=True)
            response.raise_for_status()
            
            # check if redirected url is the same otherwise change the file path
            orig_url_suffix = Path(urlparse(url)).suffix 
            redirected_url_suffix = Path(urlparse(response.url).path).suffix
            if orig_url_suffix != redirected_url_suffix:
                file_path = file_path.with_suffix(redirected_url_suffix)
                logger.warning(f"Redirected to {response.url}, saving to {file_path}")

            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded {url} to {file_path}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")


    def download_data(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, row in self.metadata_df.iterrows():
                speech_id = row['speech_id']
                url = row['what_weblink']
                file_ending = row['file_ending']
                if file_ending is np.nan:
                    file_ending = 'pdf'
                date = row['when_speech'].strftime('%Y-%m-%d')
                type_long = row['what_type_long']
                filepath = Path(self.output_dir) / f"{speech_id}_{date}_{type_long}.{file_ending}"
                executor.submit(self._download_file, url, filepath)

class Parser(): 
    def __init__(self, speeches_dir = Path('data/speeches'), out_path = Path('data/speeches/parsed.jsonl'), metadata_file = Path(r'data\export_datamart.csv')):
        self.speeches_dir = speeches_dir
        self.out_path = out_path
        self.classes_to_remove = ["ecb-publicationDate", "ecb-authors", "ecb-pressContentSubtitle"]
        # If the following text is found the parser stops processing the remaining paragraphs
        self.texts_to_remove = [
            "The views expressed in each blog entry are those of the author(s) and do not necessarily represent the views of the European Central Bank and the Eurosystem.",
        ]
        self.div_classes_to_remove = ["related-topics", "title", "address-box", "footnotes", "final hidden"]
        self.metadata_df = Metadata(metadata_file).get_metadata()

    def extract_main_content_from_html(self, html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        main_content = soup.find('main')
        result = {"content": "", "related_topics": []}
    
        # Extract related topics, has to be done before the rest otherwise its eaten from the soup already
        related_topics_div = soup.find('div', class_='related-topics')
        if related_topics_div:
            result["related_topics"] = [li.get_text(strip=True) for li in related_topics_div.find_all('li')]
        if main_content:
            # Remove paragraphs with the specified classes
            for class_to_remove in self.classes_to_remove:
                for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'], class_=class_to_remove):
                    tag.decompose()
        
            # Remove divs with the specified classes
            for div_class_to_remove in self.div_classes_to_remove:
                for div in main_content.find_all('div', class_=div_class_to_remove):
                    div.decompose()
                    
            # Remove paragraphs containing specific text and stop processing further paragraphs
            stop_processing = False
            for p in main_content.find_all('p'):
                if stop_processing:
                    p.decompose()
                elif any(text_to_remove in p.get_text() for text_to_remove in self.texts_to_remove):
                    p.decompose()
                    stop_processing = True
        
            # Extract text from the main content
            content = main_content.get_text(separator='\n', strip=True)
            # Remove multiple whitespaces and linebreaks 
            content = re.sub(r'\s+', ' ', content)
            result["content"] = content
    
        return result
    
    def process_directory(self):
        if self.out_path.exists():
            logger.info(f"File {self.out_path} already exists, skipping processing.")
            return None

        results = []
        html_files = list(Path(self.speeches_dir).rglob("*.html"))
        for index, html_file in enumerate(tqdm(html_files, desc="Processing HTML files"), start=1):
            speech_id, date, type_long = html_file.stem.split('_')
            speech_id = int(speech_id)
            
            result = self.extract_main_content_from_html(html_file)
            result["speech_id"] = speech_id
            result["date"] = date
            result["type_long"] = type_long
            results.append(result)

            if index % 300 == 0:
                with open(self.out_path, "a", encoding="utf-8") as f:
                    for res in results:
                        f.write(json.dumps(res) + "\n")
                logger.info(f"Appended {len(results)} records to {self.out_path}")
                results = []

        if results:
            with open(self.out_path, "a", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
            logger.info(f"Appended {len(results)} remaining records to {self.out_path}")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    if not Path('data/speeches').is_dir() and len(list(Path('data/speeches').glob("*.html"))) >= 3800: 
        data_downloader = DataDownloader(output_dir=FILE_DIR / 'data' / 'speeches')
        data_downloader.download_data()
    
    parser = Parser()
    parser.process_directory()
    pass


