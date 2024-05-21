import argparse
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from text_corpus import TextCorpusWithProgress
from lda_model import train_lda_model, get_topic_distribution
from utils import calculate_kl_divergence, save_topics_to_csv, load_reference_text

def main(directory, reference_file, num_topics=5, passes=15, no_below=1, no_above=0.9, minimum_topic_probability=0, max_files=None, num_words=10):
    start_time = time.time()

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Initializing text corpus...")
        text_corpus = TextCorpusWithProgress(directory, no_below=no_below, no_above=no_above, max_files=max_files)

        print("Building dictionary from documents...")
        text_corpus.build_dictionary()

        if len(text_corpus.dictionary) == 0:
            print("The dictionary is empty. Exiting...")
            return

        print("Loading reference text...")
        reference_bow = text_corpus.dictionary.doc2bow(load_reference_text(reference_file))
        
        print("Training LDA model...")
        lda_start = time.time()
        lda_model, corpus_length = train_lda_model(text_corpus, num_topics, passes)
        print(f'Finished LDA Model Training. Time taken: {time.time() - lda_start:.2f} seconds')
        if corpus_length == 0:
            print("The corpus is empty. Exiting...")
            return

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
        df.to_csv(os.path.join( output_dir, 'topic_distributions.csv'))

        print("Outputting topics and words to CSV...")
        save_topics_to_csv(lda_model, num_words, os.path.join(output_dir,'topics_words.csv'))

        print("Outputting KL divergences to CSV...")
        kl_df = pd.DataFrame(kl_divergences, index=filenames, columns=['KL Divergence'])
        kl_df.to_csv(os.path.join(output_dir, 'kl_divergences.csv'))
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
