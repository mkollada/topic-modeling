import os
import PyPDF2
import csv
from typing import List
from datetime import datetime

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


# Create a specific logger for file processing
log_filename = 'outputs/skipped_files.csv'

def initialize_csv_log():
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'read_failure_reason', 'timestamp'])

def log_skipped_file(filename: str, reason: str):
    print(f'Skipping file {filename}')
    with open(log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, reason, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

initialize_csv_log()

def get_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text.
    """
    def read_pdf():
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

    try:
        text = read_pdf()
    except Exception as e:
        log_skipped_file(file_path, str(e))
        return ''

    return text

def get_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a TXT file.
    
    Args:
        file_path (str): The path to the TXT file.
    
    Returns:
        str: The extracted text.
    """
    def read_txt():
        _, ext = os.path.splitext(file_path)
        if ext.lower() != '.txt':
            raise ValueError(f'File must be a txt. Got extension: {ext}')

        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    try:
        text = read_txt()
    except Exception as e:
        log_skipped_file(file_path, str(e))
        return ''

    return text

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses the text by tokenizing, removing stopwords, short tokens, and non-alphabetic tokens.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        List[str]: The list of preprocessed tokens.
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]  # Filter out short tokens
    tokens = [word for word in tokens if word.isalpha()]  # Filter out punctuation and numbers
    return tokens
