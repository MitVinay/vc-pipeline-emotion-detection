import numpy as np
import pandas as pd
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

# Transform the data
nltk.download('wordnet')
nltk.download('stopwords')


# confiduring logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#create a file handler
file_handler = logging.FileHandler('error.log')
file_handler.setLevel('ERROR')


# creating a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding the consoler to the handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def read_data(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        logger.debug('Text file successfully read.')
        return data
    except FileNotFoundError:
        logger.error(f"Error: The file at path '{path}' was not found.")
    except pd.errors.EmptyDataError:
        logger.error("Error: The file is empty.")
    except pd.errors.ParserError:
        logger.error("Error: The file could not be parsed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    logger.debug('lemmatisation performed succesfully')
    logger.debug('Text lower cased successfully')
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    logger.debug('Stop words remove successfully')
    df.content=df.content.apply(lambda content : removing_numbers(content))
    logger.debug('Numbers are removed successfully')
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    logger.debug('Punctuations removed succesfully')
    df.content=df.content.apply(lambda content : removing_urls(content))
    logger.debug('Urls are removed successfully')
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def main():
    train_processed_data = normalize_text(read_data("./data/raw/train.csv"))
    test_processed_data = normalize_text(read_data("./data/raw/test.csv"))


    # Defining the path
    data_path = os.path.join("data", "interim")

    # making directory
    os.makedirs(data_path)

    # write the file
    train_processed_data.to_csv(os.path.join(data_path, "train_interim.csv"))
    test_processed_data.to_csv(os.path.join(data_path, "test_interim.csv"))


if __name__ == "__main__":
    main()
