from gensim import models
from tqdm import tqdm
import numpy as np

def train_lda_model(corpus, num_topics, passes):
    # Check the length of the corpus
    corpus_length = sum(1 for _ in corpus)
    if corpus_length == 0:
        return None, 0
    print(f"Corpus length: {corpus_length}")
    lda_model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=corpus.dictionary, passes=passes, chunksize=max(1, corpus_length // 10))
    return lda_model, corpus_length

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
