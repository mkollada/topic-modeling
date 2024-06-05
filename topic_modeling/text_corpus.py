import os
import json
import logging
from tqdm import tqdm
from gensim import corpora
from data_preprocessing import get_text_from_pdf, get_text_from_txt, preprocess_text
import psutil
from typing import List, Dict, Optional, Generator
from data_preprocessing import log_skipped_file

MEMORY_THRESHOLD = 80

def is_memory_usage_high(threshold: int = MEMORY_THRESHOLD) -> bool:
    """
    Checks if the memory usage is above the specified threshold.
    
    Args:
        threshold (int): The memory usage threshold percentage.
    
    Returns:
        bool: True if memory usage is above the threshold, False otherwise.
    """
    memory_info = psutil.virtual_memory()
    return memory_info.percent > threshold

class TextCorpus:
    def __init__(self, directory: str, no_below: int = 1, no_above: float = 0.9, max_files: Optional[int] = None, cache_file: Optional[str] = None, log_filename: str = 'outputs/skipped_files.csv'):
        """
        Args:
            directory (str): The directory containing the text files.
            no_below (int): Minimum number of documents a token must appear in.
            no_above (float): Maximum proportion of documents a token can appear in.
            max_files (Optional[int]): Maximum number of files to process.
            cache_file (Optional[str]): Path to a file where the cache should be saved or loaded from.
        """
        self.directory = directory
        self.no_below = no_below
        self.no_above = no_above
        self.filepaths = []
        self.skipped_filepaths = []
        self.preprocessed_texts: Dict[str, List[str]] = {}
        self.cache_file = cache_file
        self.log_filename = log_filename

        # Load the cache if the file exists
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                self.preprocessed_texts = cache.get('preprocessed_texts', {})
                self.skipped_filepaths = cache.get('skipped_filepaths', [])
                logging.info(f"Loaded cache from {cache_file}")

        self.dictionary = corpora.Dictionary()

        all_filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        if max_files is not None:
            all_filepaths = all_filepaths[:max_files]

        new_filepaths = [filepath for filepath in all_filepaths if filepath not in self.preprocessed_texts and filepath not in self.skipped_filepaths]
        for filepath in tqdm(new_filepaths, desc="Processing new files", unit="file"):
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.pdf':
                text = get_text_from_pdf(filepath, self.log_filename)
            elif ext == '.txt':
                text = get_text_from_txt(filepath, self.log_filename)
            else:
                # print(f'Skipping file {filepath}')
                log_skipped_file(filepath, f'File type: {ext} not supported', self.log_filename)
            if text:
                tokens = preprocess_text(text)
                if not is_memory_usage_high():
                    self.preprocessed_texts[filepath] = tokens
                self.filepaths.append(filepath)
            else:
                self.skipped_filepaths.append(filepath)

        # Include previously processed filepaths
        self.filepaths.extend([filepath for filepath in self.preprocessed_texts if filepath not in self.filepaths])

    def __iter__(self) -> Generator[List[tuple], None, None]:
        """
        Iterates over the documents in the corpus, yielding their Bag-of-Words (BoW) representation.
        
        Yields:
            Generator[List[tuple], None, None]: A generator that yields the BoW representation of each document.
        """
        for filepath in self.filepaths:
            yield self.dictionary.doc2bow(self.preprocessed_texts[filepath])

    def __len__(self) -> int:
        """
        Returns the number of documents in the corpus.
        
        Returns:
            int: The number of documents.
        """
        return len(self.filepaths)

    def build_dictionary(self) -> None:
        """
        Builds the dictionary from the documents in the corpus.
        """
        for filepath in tqdm(self.filepaths, desc="Building dictionary", unit="file", leave=False):
            tokens = self.preprocessed_texts[filepath]
            if not tokens:
                logging.warning(f"No tokens found for file: {filepath}")
            self.dictionary.add_documents([tokens])
        logging.info(f"Dictionary size before filtering: {len(self.dictionary)}")
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)  # Filter based on user input
        logging.info(f"Dictionary size after filtering: {len(self.dictionary)}")

        # Save the cache to a file if a path is provided
        if self.cache_file:
            cache = {
                'preprocessed_texts': self.preprocessed_texts,
                'skipped_filepaths': self.skipped_filepaths
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
            logging.info(f"Saved cache to {self.cache_file}")

class TextCorpusWithProgress(TextCorpus):
    def __init__(self, directory: str, no_below: int = 1, no_above: float = 0.9, max_files: Optional[int] = None, cache_file: Optional[str] = None, log_filename: str = 'outputs/skipped_files.csv'):
        """
        Initializes the TextCorpusWithProgress with a directory and filtering parameters.
        
        Args:
            directory (str): The directory containing the text files.
            no_below (int): Minimum number of documents a token must appear in.
            no_above (float): Maximum proportion of documents a token can appear in.
            max_files (Optional[int]): Maximum number of files to process.
            cache_file (Optional[str]): Path to a file where the cache should be saved or loaded from.
        """
        super().__init__(directory, no_below, no_above, max_files, cache_file, log_filename)

    def __iter__(self) -> Generator[List[tuple], None, None]:
        """
        Iterates over the uncached documents in the corpus, yielding their BoW representation,
        and shows progress using a progress bar.
        
        Yields:
            Generator[List[tuple], None, None]: A generator that yields the BoW representation of each document.
        """
        uncached_filepaths = [filepath for filepath in self.filepaths if filepath not in self.preprocessed_texts]
        for filepath in tqdm(uncached_filepaths, desc="Processing files", unit="file", leave=False):
            yield self.dictionary.doc2bow(self.preprocessed_texts[filepath])

        for filepath in self.filepaths:
            if filepath in self.preprocessed_texts:
                yield self.dictionary.doc2bow(self.preprocessed_texts[filepath])
