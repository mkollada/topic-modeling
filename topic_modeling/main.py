import os
import PyPDF2
import argparse
import pandas as pd
from tqdm import tqdm
from gensim import corpora, models
import numpy as np
from scipy.special import kl_div
from gensim.utils import simple_preprocess
import time
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import psutil

nltk.download('punkt')
nltk.download('stopwords')

# Memory usage threshold as a percentage of total available memory
MEMORY_THRESHOLD = 80

def get_text_from_pdf(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.pdf':
        raise ValueError(f'File must be a pdf. Got extension: {ext}')

    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        number_of_pages = reader.numPages
    
        doc_text = ''
    
        for page_number in range(number_of_pages):
            page = reader.getPage(page_number)
            try:
                page_text = page.extract_text()
            except:
                page_text = ""
            doc_text += '\n' + page_text if page_text else ''
    
    return doc_text.strip()

def get_text_from_txt(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.txt':
        raise ValueError(f'File must be a txt. Got extension: {ext}')

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]  # Filter out short tokens
    tokens = [word for word in tokens if word.isalpha()]  # Filter out punctuation and numbers
    return tokens

def is_memory_usage_high(threshold=MEMORY_THRESHOLD):
    """Check if the memory usage is above the specified threshold."""
    memory_info = psutil.virtual_memory()
    return memory_info.percent > threshold

class TextCorpus:
    def __init__(self, directory, no_below=1, no_above=0.9, max_files=None):
        self.directory = directory
        self.dictionary = corpora.Dictionary()
        self.no_below = no_below
        self.no_above = no_above
        self.filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.pdf', '.txt'))]
        if max_files is not None:
            self.filepaths = self.filepaths[:max_files]
        self.preprocessed_texts = {}

    def __iter__(self):
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

    def build_dictionary(self):
        print("Building dictionary from documents...")
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
    def __iter__(self):
        for filepath in tqdm(self.filepaths, desc="Processing files", unit="file", leave=False):
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

def get_topic_distribution(lda_model, corpus, num_topics, minimum_probability):
    topic_distributions = []
    print("Calculating topic distributions for documents...")
    for bow in tqdm(corpus, desc="Calculating topic distributions", unit="document", leave=False):
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=minimum_probability)  # Use user-defined minimum probability
        topic_probs = [0] * num_topics
        for topic_id, prob in topic_distribution:
            topic_probs[topic_id] = prob
        topic_distributions.append(topic_probs)
    return np.array(topic_distributions)

def calculate_kl_divergence(dist1, dist2):
    return np.sum(kl_div(dist1, dist2))

def save_topics_to_csv(lda_model, num_words, file_name):
    print(f"Saving topics to {file_name}...")
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    topics_words = [[word for word, _ in topic[1]] for topic in topics]
    df = pd.DataFrame(topics_words, columns=[f'Word {i+1}' for i in range(num_words)])
    df.index.name = 'Topic'
    df.to_csv(file_name)

def main(directory, reference_file, num_topics=5, passes=15, no_below=1, no_above=0.9, minimum_topic_probability=0, max_files=None, num_words=10):
    start_time = time.time()
    
    try:
        print("Initializing text corpus...")
        text_corpus = TextCorpusWithProgress(directory, no_below=no_below, no_above=no_above, max_files=max_files)

        print("Building dictionary...")
        text_corpus.build_dictionary()

        if len(text_corpus.dictionary) == 0:
            print("The dictionary is empty. Exiting...")
            return

        print("Loading reference text...")
        ext = os.path.splitext(reference_file)[1].lower()
        if ext == '.pdf':
            reference_text = get_text_from_pdf(reference_file)
        elif ext == '.txt':
            reference_text = get_text_from_txt(reference_file)
        reference_bow = text_corpus.dictionary.doc2bow(preprocess_text(reference_text))
        
        print("Training LDA model...")
        # Check the length of the corpus
        corpus_length = sum(1 for _ in text_corpus)
        if corpus_length == 0:
            print("The corpus is empty. Exiting...")
            return
        print(f"Corpus length: {corpus_length}")

        lda_model = models.LdaModel(corpus=text_corpus, num_topics=num_topics, id2word=text_corpus.dictionary, passes=passes, chunksize=max(1, corpus_length // 10))

        print("Getting topic distributions for documents in the corpus...")
        topic_distributions = get_topic_distribution(lda_model, text_corpus, num_topics, minimum_topic_probability)

        print("Getting topic distribution for the reference document...")
        reference_distribution = lda_model.get_document_topics(reference_bow, minimum_probability=minimum_topic_probability)
        reference_probs = [0] * num_topics
        for topic_id, prob in reference_distribution:
            reference_probs[topic_id] = prob
        reference_distribution = np.array(reference_probs)

        print("Calculating KL divergence for documents...")
        kl_divergences = []
        for dist in tqdm(topic_distributions, desc="Calculating KL divergences", unit="document", leave=False):
            kl_div = calculate_kl_divergence(dist, reference_distribution)
            kl_divergences.append(kl_div)
        
        print("Outputting topic distributions to CSV...")
        filenames = [os.path.splitext(os.path.basename(filepath))[0] for filepath in text_corpus.filepaths]
        df = pd.DataFrame(topic_distributions, index=filenames, columns=[f'Topic {i+1}' for i in range(num_topics)])
        df.to_csv('topic_distributions.csv')

        print("Outputting topics and words to CSV...")
        save_topics_to_csv(lda_model, num_words, 'topics_words.csv')

        print("Outputting KL divergences to CSV...")
        kl_df = pd.DataFrame(kl_divergences, index=filenames, columns=['KL Divergence'])
        kl_df.to_csv('kl_divergences.csv')
        for i, kl_div in enumerate(kl_divergences):
            print(f'Document {filenames[i]}: KL Divergence = {kl_div}')
        
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting gracefully...")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform LDA-based topic modeling and calculate KL divergence for PDFs and text files in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing PDF and text files.')
    parser.add_argument('reference_file', type=str, help='Path to the reference PDF or text file.')
    parser.add_argument('--num_topics', type=int, default=5, help='Number of topics for LDA. Default is 5.')
    parser.add_argument('--passes', type=int, default=15, help='Number of passes for LDA training. Default is 15.')
    parser.add_argument('--no_below', type=int, default=1, help='Filter out tokens that appear in fewer than no_below documents. Default is 1.')
    parser.add_argument('--no_above', type=float, default=0.9, help='Filter out tokens that appear in more than no_above proportion of documents. Default is 0.9.')
    parser.add_argument('--minimum_topic_probability', type=float, default=0, help='Minimum topic probability to include in the results. Default is 0.')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process. Default is None (process all files).')
    parser.add_argument('--num_words', type=int, default=10, help='Number of words per topic to save in the CSV. Default is 10.')
    
    args = parser.parse_args()

    print(f'Processing {len([os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(('.pdf', '.txt'))])} files in directory: {args.directory}')
    
    main(args.directory, args.reference_file, args.num_topics, args.passes, args.no_below, args.no_above, args.minimum_topic_probability, args.max_files, args.num_words)
