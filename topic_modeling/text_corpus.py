import os
import logging
from tqdm import tqdm
from gensim import corpora
from data_preprocessing import get_text_from_pdf, get_text_from_txt, preprocess_text
import psutil
from typing import List, Dict, Optional, Generator

MEMORY_THRESHOLD = 80

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, directory: str, no_below: int = 1, no_above: float = 0.9, max_files: Optional[int] = None):
        """
        Initializes the TextCorpus with a directory and filtering parameters.
        
        Args:
            directory (str): The directory containing the text files.
            no_below (int): Minimum number of documents a token must appear in.
            no_above (float): Maximum proportion of documents a token can appear in.
            max_files (Optional[int]): Maximum number of files to process.
        """
        self.directory = directory
        self.dictionary = corpora.Dictionary()
        self.no_below = no_below
        self.no_above = no_above
        self.filepaths = []
        self.skipped_filepaths = []
        self.preprocessed_texts: Dict[str, List[str]] = {}

        all_filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.pdf', '.txt'))]
        if max_files is not None:
            all_filepaths = all_filepaths[:max_files]

        for filepath in all_filepaths:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.pdf':
                text = get_text_from_pdf(filepath)
            elif ext == '.txt':
                text = get_text_from_txt(filepath)
            if text:
                tokens = preprocess_text(text)
                if not is_memory_usage_high():
                    self.preprocessed_texts[filepath] = tokens
                self.filepaths.append(filepath)
            else:
                self.skipped_filepaths.append(filepath)

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

class TextCorpusWithProgress(TextCorpus):
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
