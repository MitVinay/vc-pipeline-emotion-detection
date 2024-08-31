import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# confiduring logger
logger = logging.getLogger('data_injgestion')
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


# function to load the paramters with type hinting
def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.debug('text sized recieved')
            return test_size
    except FileNotFoundError as e:
        logger.error(f"Error: The file '{params_path}' was not found.")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error: Failed to parse YAML file. Details: {e}")
        raise e
    except KeyError as e:
        logger.error(f"Error: Missing key in the YAML file. Details: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred. Details: {e}")
        raise e
    
# Function to read the data
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        raise
    except FileNotFoundError:
        print(f"Error: The file at '{url}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred. Details: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Dropping the 'tweet_id' column
        if 'tweet_id' in df.columns:
            df = df.drop(columns=['tweet_id'])
        else:
            print("Warning: 'tweet_id' column not found in the DataFrame.")

        # Filtering the data to only include 'happiness' and 'sadness'
        if 'sentiment' in df.columns:
            final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        else:
            raise KeyError("Error: 'sentiment' column not found in the DataFrame.")

        # Replacing 'happiness' with 1 and 'sadness' with 0
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        return final_df

    except KeyError as e:
        print(f"Key Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred. Details: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # Making the directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)

        # Writing the train and test data to CSV files
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        print(f"Files saved successfully in {data_path}")

    except PermissionError:
        print(f"Error: Permission denied while creating directory or writing files in '{data_path}'.")
        raise
    except OSError as e:
        print(f"Error: OS error occurred while creating directory or writing files. Details: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred. Details: {e}")
        raise

def main():
    test_size = load_params('params.yaml')
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = process_data(df)
    # Spliting the data
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    # Defining the path
    data_path = os.path.join("data", "raw")
    save_data(train_data, test_data, data_path)

if __name__ == "__main__":
    main()