import logging
from gensim import models
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Optional
from text_corpus import TextCorpus

def train_lda_model(corpus: TextCorpus, num_topics: int, passes: int, workers: int) -> Tuple[Optional[models.LdaMulticore], int]:
    """
    Trains an LDA model on the given corpus using multiple cores.
    
    Args:
        corpus (TextCorpus): The text corpus.
        num_topics (int): The number of topics.
        passes (int): The number of passes for LDA training.
        workers (int): The number of worker threads to use for training.
    
    Returns:
        Tuple[Optional[models.LdaMulticore], int]: A tuple containing the trained LDA model and the length of the corpus.
                                                   Returns (None, 0) if the corpus is empty.
    """
    # Check the length of the corpus
    corpus_length = sum(1 for _ in corpus)
    if corpus_length == 0:
        logging.warning("The corpus is empty. Exiting training.")
        return None, 0
    logging.info(f"Training LDA model with {num_topics} topics, {passes} passes, and {workers} workers...")
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=corpus.dictionary, passes=passes, workers=workers, chunksize=max(1, corpus_length // 10))
    logging.info("LDA model training completed.")
    return lda_model, corpus_length

def get_topic_distribution(lda_model: models.LdaModel, corpus: TextCorpus, num_topics: int, minimum_probability: float) -> np.ndarray:
    """
    Calculates the topic distributions for documents in the corpus.
    
    Args:
        lda_model (models.LdaModel): The trained LDA model.
        corpus (TextCorpus): The text corpus.
        num_topics (int): The number of topics.
        minimum_probability (float): The minimum topic probability to include in the results.
    
    Returns:
        np.ndarray: An array of topic distributions for each document in the corpus.
    """
    topic_distributions = []
    logging.info("Calculating topic distributions for documents...")
    for bow in tqdm(corpus, desc="Calculating topic distributions", unit="document", leave=False):
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=minimum_probability)
        topic_probs = [0] * num_topics
        for topic_id, prob in topic_distribution:
            topic_probs[topic_id] = prob
        topic_distributions.append(topic_probs)
    logging.info("Topic distributions calculation completed.")
    return np.array(topic_distributions)
