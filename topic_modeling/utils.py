import os
import numpy as np
from scipy.special import kl_div
import pandas as pd
from collections import Counter
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from data_preprocessing import get_text_from_pdf, get_text_from_txt, preprocess_text

def calculate_kl_divergence(dist1: np.ndarray, dist2: np.ndarray) -> float:
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    dist1 = np.array(dist1) + epsilon
    dist2 = np.array(dist2) + epsilon
    return np.sum(kl_div(dist1, dist2))

def save_topics_to_csv(lda_model: LdaModel, num_words: int, file_name: str) -> None:
    print(f"Saving topics to {file_name}...")
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    topics_words = [[word for word, _ in topic[1]] for topic in topics]
    df = pd.DataFrame(topics_words, columns=[f'Word {i+1}' for i in range(num_words)])
    df.index.name = 'Topic'
    df.to_csv(file_name)

def load_reference_text(reference_file: str) -> str:
    ext = os.path.splitext(reference_file)[1].lower()
    if ext == '.pdf':
        reference_text = get_text_from_pdf(reference_file)
    elif ext == '.txt':
        reference_text = get_text_from_txt(reference_file)
    return reference_text 

def get_topic_word_distributions(lda_model: LdaModel, num_topics: int, dictionary: Dictionary) -> list:
    topic_word_distributions = []
    for topic_id in range(num_topics):
        word_distribution = np.zeros(len(dictionary))
        topic_terms = lda_model.get_topic_terms(topic_id, topn=len(dictionary))
        for word_id, prob in topic_terms:
            word_distribution[word_id] = prob
        topic_word_distributions.append(word_distribution)
    return topic_word_distributions

def get_word_distribution(text: str, dictionary: Dictionary) -> np.ndarray:
    tokens = preprocess_text(text)

    token_counts = Counter(tokens)

    word_distribution = np.zeros(len(dictionary))
    for token, count in token_counts.items():
        if token in dictionary.token2id:
            word_id = dictionary.token2id[token]
            word_distribution[word_id] = count

    if np.sum(word_distribution) == 0:
        print("Warning: Word distribution sums to zero.")  # 
    else:
        word_distribution = word_distribution / np.sum(word_distribution)  # Normalize the distribution

    return word_distribution


def save_word_distribution_to_csv(word_distribution: np.ndarray, dictionary: Dictionary, file_name: str) -> None:
    print(f"Saving word distribution to {file_name}...")
    words = [dictionary[i] for i in range(len(dictionary))]
    df = pd.DataFrame(word_distribution, index=words, columns=['Probability'])
    df.index.name = 'Word'
    df.to_csv(file_name)

def save_topic_word_distributions_to_csv(topic_word_distributions: list, dictionary: Dictionary, file_name: str) -> None:
    print(f"Saving topic word distributions to {file_name}...")
    words = [dictionary[i] for i in range(len(dictionary))]
    df = pd.DataFrame(topic_word_distributions, columns=words)
    df.index.name = 'Topic'
    df.to_csv(file_name)
