import os
import PyPDF2
from typing import List

def get_text_from_pdf(file_path: str) -> str:
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

def get_text_from_txt(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.txt':
        raise ValueError(f'File must be a txt. Got extension: {ext}')

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def preprocess_text(text: str) -> List[str]:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]  # Filter out short tokens
    tokens = [word for word in tokens if word.isalpha()]  # Filter out punctuation and numbers
    return tokens
