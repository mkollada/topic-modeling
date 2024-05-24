import os
from tqdm import tqdm
from gensim import corpora
from data_preprocessing import get_text_from_pdf, get_text_from_txt, preprocess_text
import psutil
from typing import List, Dict, Optional, Generator

MEMORY_THRESHOLD = 80

def is_memory_usage_high(threshold: int = MEMORY_THRESHOLD) -> bool:
    """Check if the memory usage is above the specified threshold."""
    memory_info = psutil.virtual_memory()
    return memory_info.percent > threshold

class TextCorpus:
    def __init__(self, directory: str, no_below: int = 1, no_above: float = 0.9, max_files: Optional[int] = None):
        self.directory = directory
        self.dictionary = corpora.Dictionary()
        self.no_below = no_below
        self.no_above = no_above
        self.filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.pdf', '.txt'))]
        if max_files is not None:
            self.filepaths = self.filepaths[:max_files]
        self.preprocessed_texts: Dict[str, List[str]] = {}

    def __iter__(self) -> Generator[List[tuple], None, None]:
        for filepath in self.filepaths:
            if filepath in self.preprocessed_texts:
                yield self.dictionary.doc2bow(self.preprocessed_texts[filepath])
            else:
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.pdf':
                    text = get_text_from_pdf(filepath)
                elif ext == '.txt':
                    text = get_text_from_txt(filepath)
                tokens = preprocess_text(text)
                if not is_memory_usage_high():
                    self.preprocessed_texts[filepath] = tokens
                yield self.dictionary.doc2bow(tokens)

    def build_dictionary(self) -> None:
        for filepath in tqdm(self.filepaths, desc="Building dictionary", unit="file", leave=False):
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.pdf':
                text = get_text_from_pdf(filepath)
            elif ext == '.txt':
                text = get_text_from_txt(filepath)
            tokens = preprocess_text(text)
            if not tokens:
                print(f"No tokens found for file: {filepath}")  # Debug print
            if not is_memory_usage_high():
                self.preprocessed_texts[filepath] = tokens
            self.dictionary.add_documents([tokens])
        print(f"Dictionary size before filtering: {len(self.dictionary)}")  # Debug print
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)  # Filter based on user input
        print(f"Dictionary size after filtering: {len(self.dictionary)}")  # Debug print

class TextCorpusWithProgress(TextCorpus):
    def __iter__(self) -> Generator[List[tuple], None, None]:
        uncached_filepaths = [filepath for filepath in self.filepaths if filepath not in self.preprocessed_texts]
        for filepath in tqdm(uncached_filepaths, desc="Processing files", unit="file", leave=False):
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.pdf':
                text = get_text_from_pdf(filepath)
            elif ext == '.txt':
                text = get_text_from_txt(filepath)
            tokens = preprocess_text(text)
            if not is_memory_usage_high():
                self.preprocessed_texts[filepath] = tokens
            yield self.dictionary.doc2bow(tokens)

        for filepath in self.filepaths:
            if filepath in self.preprocessed_texts:
                yield self.dictionary.doc2bow(self.preprocessed_texts[filepath])
