import pandas as pd
import numpy as np
from pathlib import Path
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import time
import random
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import json
from loguru import logger
import sys
import gensim
import nltk
import spacy
from typing import List, Tuple
import pickle

FILE_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).name
logger.remove()
logger.add(sys.stdout, format="{time} | {level} | {module}.{function}:{line} | {message}")
logger.add("log_file.log", format="{time} | {level} | {module}.{function}:{line} | {message}")

class Metadata:
    """
    Handles the loading and preprocessing of metadata from a CSV file.
    """
    def __init__(self, path_to_file: str|Path = r"data\export_datamart.csv") -> None:
        self.path_to_file = Path(path_to_file)
        self.metadata_df = pd.read_csv(self.path_to_file)
        self._prepare_metadata()

    def _prepare_metadata(self) -> None:
        """
        Prepares metadata by formatting dates and extracting file extensions.
        """
        self.metadata_df["when_speech"] = pd.to_datetime(
            self.metadata_df["when_speech"], format="%Y-%m-%d"
        )
        self.metadata_df["file_ending"] = self.metadata_df["what_weblink"].str.extract(
            r"\.(\w+)$"
        )
        self.what_type_dict = {
            "S": "speech",
            "I": "interview",
            "P": "press-conference",
            "B": "blog-post",
            "E": "podcast",
        }
        self.metadata_df["what_type_long"] = self.metadata_df["what_type"].apply(
            lambda x: self.what_type_dict[x]
        )
    
    def get(self, speech_id:int) -> tuple: 
        """
        Returns the metadata for the given speech_id.
        
        Args:
            speech_id (int): The speech ID.
        Returns:
            tuple: A tuple containing the metadata for the speech.
        """
        return self.metadata_df[self.metadata_df["speech_id"] == speech_id].iloc[0].to_dict()

class DataDownloader:
    """
    Downloads the speeches/documents from the url from the metadata and saves it to some directory.
    """
    def __init__(self, output_dir: str|Path =FILE_DIR / "data" / "speeches", metadata_file: str|Path = r"data\export_datamart.csv") -> None:
        self.output_dir = Path(output_dir)
        self.metadata_df = Metadata(metadata_file).metadata_df
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url:str, file_path:Path) -> None:
        """
        Downloads a file from the given URL and saves it to the specified path.
        
        Args:
            url (str): The URL of the file to download.
            file_path (Path): The path to save the downloaded file.
        """
        try:
            time.sleep(abs(random.normalvariate()))
            headers = {
                "User-Agent": random.choice(
                    [
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1 Safari/605.1.15",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
                    ]
                )
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
        """
        Downloads all document files concurrently.
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, row in self.metadata_df.iterrows():
                speech_id = row["speech_id"]
                url = row["what_weblink"]
                file_ending = row["file_ending"]
                if file_ending is np.nan:
                    file_ending = "pdf"
                date = row["when_speech"].strftime("%Y-%m-%d")
                type_long = row["what_type_long"]
                filepath = Path(self.output_dir)/ f"{speech_id}_{date}_{type_long}.{file_ending}"
                executor.submit(self._download_file, url, filepath)

class Parser:
    """
    Parses HTML speech files, extracting main content and saving it as JSON.
    """
    def __init__(self, speeches_dir: str|Path = Path("data/speeches"), out_path: str|Path =Path("data/speeches/parsed.jsonl"), metadata_file: str|Path =Path(r"data\export_datamart.csv")) -> None:
        """
        Initializes the Parser.

        Args:
            speeches_dir (Path): Directory containing speech HTML files.
            out_path (Path): Path for the parsed JSON output.
            metadata_file (Path): Path to metadata file.
        """
        self.speeches_dir = Path(speeches_dir)
        self.out_path = Path(out_path)
        self.classes_to_remove = [
            "ecb-publicationDate",
            "ecb-authors",
            "ecb-pressContentSubtitle",
        ]
        # If the following text is found the parser stops processing the remaining paragraphs
        self.texts_to_remove = [
            "The views expressed in each blog entry are those of the author(s) and do not necessarily represent the views of the European Central Bank and the Eurosystem.",
        ]
        self.div_classes_to_remove = [
            "related-topics",
            "title",
            "address-box",
            "footnotes",
            "final hidden",
        ]
        self.metadata = Metadata(metadata_file)
        logger.info(f"Initialized Parser with speeches_dir={self.speeches_dir}, out_path={self.out_path}")

    def extract_main_content_from_html(self, html_file_path:str|Path) -> dict:
        """
        Extracts the main content from an HTML file.
        
        Args:
            html_file_path (str|Path): Path to the HTML file.
        Returns:
            dict: A dictionary containing the main content and related topics.
        """
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        main_content = soup.find("main")
        result = {"content": "", "related_topics": []}

        # Extract related topics, has to be done before the rest otherwise its eaten from the soup already
        related_topics_div = soup.find("div", class_="related-topics")
        if related_topics_div:
            result["related_topics"] = [
                li.get_text(strip=True) for li in related_topics_div.find_all("li")
            ]
        if main_content:
            # Remove paragraphs with the specified classes
            for class_to_remove in self.classes_to_remove:
                for tag in main_content.find_all(
                    ["p", "h1", "h2", "h3", "h4", "h5", "h6"], class_=class_to_remove
                ):
                    tag.decompose()

            # Remove divs with the specified classes
            for div_class_to_remove in self.div_classes_to_remove:
                for div in main_content.find_all("div", class_=div_class_to_remove):
                    div.decompose()

            # Remove paragraphs containing specific text and stop processing further paragraphs
            stop_processing = False
            for p in main_content.find_all("p"):
                if stop_processing:
                    p.decompose()
                elif any(
                    text_to_remove in p.get_text()
                    for text_to_remove in self.texts_to_remove
                ):
                    p.decompose()
                    stop_processing = True

            content = main_content.get_text(separator="\n", strip=True)
            # Remove multiple whitespaces and linebreaks
            content = re.sub(r"\s+", " ", content)
            result["content"] = content

        return result

    def process_directory(self) -> None:
        """
        Processes all HTML files in the speeches directory and saves the extracted content as JSON.
        """
        logger.info(f"Processing HTML files in {self.speeches_dir}")
        if self.out_path.exists():
            logger.info(f"File {self.out_path} already exists, skipping processing.")
            return None

        results = []
        html_files = list(Path(self.speeches_dir).rglob("*.html"))
        for index, html_file in enumerate(tqdm(html_files, desc="Processing HTML files"), start=1):
            speech_id, date, type_long = html_file.stem.split("_")
            speech_id = int(speech_id)

            # Add metadata information to the results 
            metadata_fields = self.metadata.get(speech_id)

            result = self.extract_main_content_from_html(html_file)
            result["id"] = speech_id
            result["date"] = str(metadata_fields.get("when_speech", None))
            result["author"] = metadata_fields.get("who", None)
            result["title"] = metadata_fields.get("what_title", None)
            result["url"] = metadata_fields.get("what_weblink", None)
            result['language'] = metadata_fields.get("what_language", None)
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
        logger.info(f"Processed {len(html_files)} HTML files, saved to {self.out_path}")

class Preprocessor: 
    def __init__(self, language: str = "english", spacy_model: str = "en_core_web_sm"):
        self.language = language
        self.spacy_nlp = spacy.load(spacy_model, disable=["ner", "parser"])

    def set_stopwords(self, source="spacy", additional_stopwords: List[str]|Path|str = []):
            if source == "spacy":
                logger.info("Using spaCy stopwords.")
                self.stopwords = set(self.spacy_nlp.Defaults.stop_words)
            elif source == "nltk":
                logger.info("Using NLTK stopwords.")
                self.stopwords = set(nltk.corpus.stopwords.words("english"))
            elif source == "both":
                logger.info("Using spaCy and NLTK stopwords.")
                self.stopwords = set(self.spacy_nlp.Defaults.stop_words).union(set(nltk.corpus.stopwords.words("english")))
            else:
                raise ValueError("Invalid source. Choose from 'spacy', 'nltk', 'both'")

            if additional_stopwords:
                if isinstance(additional_stopwords, (Path, str)):
                    stopword_path = Path(additional_stopwords)
                    additional_stopwords = stopword_path.read_text().splitlines()
                self.stopwords = self.stopwords.union(set(additional_stopwords))
                logger.info(f"Added {len(additional_stopwords)} additional stopwords.")
            logger.info(f"Total stopwords: {len(self.stopwords)}")

    def tokenize(self, text: str) -> List[str]: 
        return [x for x in gensim.utils.tokenize(text, lower=True)]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [x for x in tokens if x not in self.stopwords]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        wnl = nltk.stem.WordNetLemmatizer()
        return [wnl.lemmatize(x) for x in tokens]

    def preprocess_document(self, text:str) -> List[str]: 
        text = text.lower()
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens
    
    def preprocess_dataset(self, parsed:Path|str|pd.DataFrame, output_dir:Path|str, save = True) -> None:
        if isinstance(parsed, pd.DataFrame):
            logger.info("Preprocessing DataFrame.")
            parsed_df = parsed
        else: 
            logger.info(f"Preprocessing dataset from {parsed}")
            parsed = Path(parsed)
            parsed_df = pd.read_json(parsed, lines=True)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        parsed_df["tokens"] = parsed_df["content"].apply(self.preprocess_document)
        self.processed_documents = parsed
        
        if save:
            parsed_df.to_json(output_dir/"preprocessed_data.jsonl", lines=True, orient="records")
            logger.info(f"Preprocessed dataset saved to {output_dir}/preprocessed_data.jsonl")

    def build_dictionary_and_corpus(self, output_dir:Path|str, save = True) -> None: 
        if not hasattr(self, "processed_documents"):
            raise ValueError("No preprocessed documents found. Run preprocess_dataset() first.")
        
        logger.info("Building dictionary and corpus.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dictionary = gensim.corpora.Dictionary(self.processed_documents["tokens"])
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_documents["tokens"]]
        tfidf_model = gensim.models.TfidfModel(self.bow_corpus)
        self.tfidf_corpus = tfidf_model[self.bow_corpus]

        if save: 
            self.dictionary.save(str(output_dir / "dictionary"))
            with open(output_dir / "bow_corpus.pkl", "wb") as f:
                pickle.dump(self.bow_corpus, f)
            with open(output_dir / "tfidf_corpus.pkl", "wb") as f:
                pickle.dump(self.tfidf_corpus, f)
            logger.info(f"Dictionary, BoW and TF-IDF corpus built and saved to {output_dir}.")

    def load_df_dict_corpus(self,preprocessed_out_dir:Path|str) -> Tuple[pd.DataFrame, gensim.corpora.Dictionary, List[List[Tuple[int, int]]], gensim.interfaces.TransformedCorpus]:
        preprocessed_out_dir = Path(preprocessed_out_dir)
        preprocessed_data = pd.read_json(preprocessed_out_dir / "preprocessed_data.jsonl", lines=True)
        dictionary = gensim.corpora.Dictionary.load(str(preprocessed_out_dir / "dictionary"))
        with open(preprocessed_out_dir / "bow_corpus.pkl", "rb") as f:
            bow_corpus = pickle.load(f)
        with open(preprocessed_out_dir / "tfidf_corpus.pkl", "rb") as f:
            tfidf_corpus = pickle.load(f)
        return preprocessed_data, dictionary, bow_corpus, tfidf_corpus

if __name__ == "__main__":
    # Configurations
    paths = {
        "metadata": Path(r"data\export_datamart.csv"),
        "all_speeches": Path(r"data\all_ECB_speeches.csv"),
        "html_dir": Path(r"data\speeches"),
        "parsed_file": Path(r"data\speeches\parsed.jsonl"),
        "preprocessed_dir": Path(r"data\preprocessed_en"),
    }

    logger.info("Starting data gathering process.")
    logger.info(f"Paths: {paths}")

    # Check and download speeches if necessary
    if not paths["html_dir"].is_dir() or len(list(paths["html_dir"].glob("*.html"))) < 3800:
        logger.info("Downloading speeches: directory missing or insufficient files.")
        DataDownloader(output_dir=paths["html_dir"]).download_data()

    # Check and parse speeches if necessary
    if not paths["parsed_file"].is_file():
        logger.info("Parsed output file missing. Starting parsing process.")
        Parser().process_directory()
    else:
        logger.info("Parsed output file exists. Validating content.")
        nrows, ncols = pd.read_json(paths["parsed_file"], lines=True).shape
        if nrows < 3800 or ncols != 9:
            logger.info("Parsed output file incomplete. Re-parsing required.")
            Parser().process_directory()
        else:
            logger.info("Parsed output file is complete.")
    
    # Preprocess the parsed JSON file # TODO: Include other languages if necessary
    logger.info("Preprocessing parsed JSON file.")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    preprocessor = Preprocessor(language="english", spacy_model="en_core_web_sm")
    preprocessor.set_stopwords("both")
    # Removing non english documents from the dataset
    en_parsed = pd.read_json(paths["parsed_file"], lines=True)
    en_parsed = en_parsed[en_parsed["language"] == "EN"]
    preprocessor.preprocess_dataset(en_parsed, paths['preprocessed_dir'])
    preprocessor.build_dictionary_and_corpus(paths['preprocessed_dir'])
    pass