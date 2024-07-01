import argparse
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from text_corpus import TextCorpusWithProgress
from lda_model import train_lda_model, get_topic_distribution
from utils import calculate_kl_divergence, save_topics_to_csv, load_reference_text, get_topic_word_distributions, get_word_distribution, save_word_distribution_to_csv, save_topic_word_distributions_to_csv
import nltk
import logging
from typing import Optional
from data_preprocessing import initialize_csv_log

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

def main(directory: str, reference_file: str, num_topics: int = 5, passes: int = 15, no_below: int = 1, 
         no_above: float = 0.9, minimum_topic_probability: float = 0, max_files: Optional[int] = None, 
         num_words: int = 10, output_directory: str = 'outputs', cache_file: Optional[str] = 'text_corpus_cache.json',
         num_workers: int = 4) -> None:
    # Ensure necessary NLTK data is downloaded
    download_nltk_data()

    start_time = time.time()
    
    log_filename = os.path.join(output_directory, 'skipped_files.csv')
    initialize_csv_log(log_filename)
    
    try:
        logging.info("Initializing text corpus...")
        text_corpus = TextCorpusWithProgress(directory, no_below=no_below, no_above=no_above, max_files=max_files, cache_file=cache_file, log_filename=log_filename, num_workers=num_workers)

        logging.info("Building dictionary...")
        text_corpus.build_dictionary()

        if len(text_corpus.dictionary) == 0:
            logging.warning("The dictionary is empty. Exiting...")
            return

        logging.info("Loading reference text...")
        reference_text = load_reference_text(reference_file, log_filename)
        reference_word_distribution = get_word_distribution(reference_text, text_corpus.dictionary)
        logging.info("Finished loading reference text.")
        
        lda_start = time.time()
        logging.info("Training LDA model...")
        lda_model, corpus_length = train_lda_model(text_corpus, num_topics, passes, num_workers)
        logging.info(f'Finished training LDA Model. Time taken: {time.time() - lda_start:.2f} seconds')

        if corpus_length == 0:
            logging.warning("The corpus is empty. Exiting...")
            return

        logging.info("Calculating KL divergence for each topic in each document...")

        topic_word_distributions = get_topic_word_distributions(lda_model, num_topics, text_corpus.dictionary)
        kl_divergences = []
        for topic_id in range(num_topics):
            kl_div = calculate_kl_divergence(topic_word_distributions[topic_id], reference_word_distribution)
            kl_divergences.append(kl_div)
        
        logging.info("KL Divergences Calculated. Outputting KL divergences to CSV...")
        filenames = [os.path.splitext(os.path.basename(filepath))[0] for filepath in text_corpus.filepaths]
        kl_df = pd.DataFrame(kl_divergences, index=[f'Topic {i+1}' for i in range(num_topics)])
        kl_df.to_csv(os.path.join(output_directory, 'kl_divergences.csv'))

        logging.info("Outputting topic distributions to CSV...")
        topic_distributions = get_topic_distribution(lda_model, text_corpus, num_topics, minimum_topic_probability)
        df = pd.DataFrame(topic_distributions, index=filenames, columns=[f'Topic {i+1}' for i in range(num_topics)])
        df.to_csv(os.path.join(output_directory, 'topic_distributions.csv'))

        logging.info("Outputting topics and words to CSV...")
        save_topics_to_csv(lda_model, num_words, os.path.join(output_directory, 'topics_words.csv'))
        
        end_time = time.time()
        logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return

    except KeyboardInterrupt:
        logging.error("\nKeyboard Interrupt. Exiting gracefully...")
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
    parser.add_argument('--output_directory', type=str, default='outputs', help='Directory to save output CSV files. Default is "outputs".')
    parser.add_argument('--cache_file', type=str, default=None, help='Path to the cache file to save/load processed text data. Default is None.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads to use for both corpus creation and LDA training. Default is 4.')
    
    args = parser.parse_args()

    # Create output directory if it does not exist
    os.makedirs(args.output_directory, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(f'{args.output_directory}/lda_processing.log', mode='w')])
    logging.getLogger('gensim').setLevel(logging.ERROR)
    logging.getLogger('lda_model').setLevel(logging.ERROR)

    logging.info(f'Processing {len([os.path.join(args.directory, filename) for filename in os.listdir(args.directory) if filename.endswith(('.pdf', '.txt'))])} files in directory: {args.directory}')
    
    main(args.directory, args.reference_file, args.num_topics, args.passes, args.no_below, args.no_above, args.minimum_topic_probability, args.max_files, args.num_words, args.output_directory, args.cache_file, args.num_workers)
