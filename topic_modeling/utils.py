import os
import numpy as np
from scipy.special import kl_div
import pandas as pd
from data_preprocessing import get_text_from_pdf, get_text_from_txt, preprocess_text

def calculate_kl_divergence(dist1, dist2):
    return np.sum(kl_div(dist1, dist2))

def save_topics_to_csv(lda_model, num_words, file_name):
    print(f"Saving topics to {file_name}...")
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    topics_words = [[word for word, _ in topic[1]] for topic in topics]
    df = pd.DataFrame(topics_words, columns=[f'Word {i+1}' for i in range(num_words)])
    df.index.name = 'Topic'
    df.to_csv(file_name)

def load_reference_text(reference_file):
    ext = os.path.splitext(reference_file)[1].lower()
    if ext == '.pdf':
        reference_text = get_text_from_pdf(reference_file)
    elif ext == '.txt':
        reference_text = get_text_from_txt(reference_file)
    return preprocess_text(reference_text)
