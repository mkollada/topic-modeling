# Topic Modeling and KL Divergence Calculation

This project performs LDA-based topic modeling on PDF and text files in a directory and calculates the KL divergence between the word distributions of topics in each document and a reference document. The results are saved in CSV files.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Arguments](#arguments)
- [Output](#output)

## Installation

1. **Extract the zipped file:**

   Download the zipped file and extract its contents to your desired directory.

   unzip topic_modeling.zip -d topic_modeling
   cd topic_modeling

2. **Install dependencies:**

   Install the required Python packages using pip.

   ``` 
   pip install -r requirements.txt
   ```


## Usage

1. **Prepare your data:**

   Ensure you have a directory with PDF and/or text files that you want to analyze. Also, have a reference PDF or text file for comparison.

2. **Run the script:**

   Use the following command to run the script. Replace the paths and parameters as needed.

   python topic_modeling/main.py path_to_directory_with_pdfs path_to_reference_file.pdf

   Example:
   
   ```
   python topic_modeling/main.py data/document_directory/ data/reference.pdf --num_topics 10 --passes 20 --no_below 5 --no_above 0.5 --minimum_topic_probability 0.01 --max_files 50 --num_words 15 --output_directory outputs --num_workers 7
   ```

## Project Structure

```
topic_modeling/
│
├── data_preprocessing.py     # Functions for reading and preprocessing text from PDF and text files.
├── lda_model.py              # Functions for training the LDA model and getting topic distributions.
├── main.py                   # Main script to run the analysis.
├── text_corpus.py            # Class definitions for handling text corpora.
└── utils.py                  # Utility functions for KL divergence calculation and saving results.
requirements.txt          # Python dependencies.

```

## Command Line Arguments
Required:
- `directory` (str): Path to the directory containing PDF and text files.
- `reference_file` (str): Path to the reference PDF or text file.

Optional:
- `--num_topics` (int): Number of topics for LDA. Default is 5.
- `--passes` (int): Number of passes for LDA training. Default is 15.
- `--no_below` (int): Filter out tokens that appear in fewer than no_below documents. Default is 1.
- `--no_above` (float): Filter out tokens that appear in more than no_above proportion of documents. Default is 0.9.
- `--minimum_topic_probability` (float): Minimum topic probability to include in the results. Default is 0.
- `--max_files` (int): Maximum number of files to process. Default is None (process all files).
- `--num_words` (int): Number of words per topic to save in the CSV. Default is 10.
- `--output_directory` (str): Directory to save output CSV files. Default is "outputs".
- `--cache_file` (str): Path to the cache file to save/load processed text data. Default is None.
- `--num_workers` (int): Number of CPUs available for parallel processing
- `--dict_size` (int): Number of words to limit the dictionary size to

## Output

The script generates the following CSV files in the specified output directory:

- `topic_distributions.csv`: Topic distributions for each document.
- `topics_words.csv`: Top words for each topic.
- `kl_divergences.csv`: KL divergence values for each document.
- `skipped_files.csv`: A csv of the document files that were skipped during processing
- `lda_processing.log`: A log output by the code documenting the process of the pipeline

## Note on `--cache_file` argument:

This argument can save significant time when running the code on the same set of documents again, or when augmenting a set of documents that you've already processed.

Setting this argument either:

- Creates a dictionary cache file if one with that name does not exist
- Reads in the dictionary cache if a file with that name already exisis

Set a new cache file name or don't use this argument if you're using a new dataset

## Note on `--dict_size` argument:

When limiting the dictionary size using this argument its important to consider how the dictionary is made smaller to get desired characteristics.

The dictionary will use the top <dict_size> words in terms of frequency AFTER filtering with the `no_below` and `no_above` parameters. Thus those two parameters should be considered when using the `--dict_size` argument.

