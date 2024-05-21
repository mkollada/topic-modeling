import os
import PyPDF2
from nltk.tokenize import word_tokenize

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
    
# Load stop words from the local file
def load_stopwords(file_path='topic_modeling/stopwords.txt'):
    with open(file_path, 'r') as file:
        stop_words = set(word.strip() for word in file.readlines())
    return stop_words

stop_words = load_stopwords()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]  # Filter out short tokens
    tokens = [word for word in tokens if word.isalpha()]  # Filter out punctuation and numbers
    return tokens
