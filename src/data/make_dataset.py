import numpy as np
import pandas as pd
import pathlib
import logging
import yaml
from sklearn.model_selection import train_test_split

def load_params(path: str, logger):
    try:
        with open("params.yaml", 'r') as f:
            logger.debug("Attempting to load parameters from params.yaml")
            params = yaml.safe_load(f)
            logger.debug("Parameters loaded successfully from params.yaml")
            return params['make_dataset']
    except yaml.YAMLError as e:
        logger.error(f"YAMLError: An error occurred while parsing params.yaml. Error: {e}")
        raise yaml.YAMLError
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The file params.yaml was not found. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading parameters: {e}")
        raise Exception

def load_data(path: str, logger) -> pd.DataFrame:
    try:
        with open(path + "/creditcard.csv") as f:
            logger.debug(f"Attempting to load dataset from {path}/creditcard.csv")
            data = pd.read_csv(f)
            logger.info("Dataset loaded successfully from creditcard.csv")
            return data
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The file creditcard.csv was not found in the specified path. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the dataset: {e}")
        raise Exception

def split_data(data: pd.DataFrame, params, logger):
    logger.debug("Attempting to split the dataset into training and testing sets")
    train, test = train_test_split(data, test_size=params['test_split'], random_state=params['seed'])
    logger.debug("Dataset split successfully into training and testing sets")
    return train, test

def save_data(train: pd.DataFrame, test: pd.DataFrame, path: str, logger) -> None:
    try:
        logger.debug(f"Attempting to save training and testing datasets to {path}")
        train.to_csv(path + "/train.csv", index=False)
        test.to_csv(path + "/test.csv", index=False)
        logger.debug("Training and testing datasets saved successfully")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The specified path for saving datasets was not found. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the datasets: {e}")
        raise Exception

def main() -> int:
    # Creating Logger
    logger = logging.getLogger("Data Ingestion Logger")
    logger.setLevel("DEBUG")

    # Creating Handler
    handler = logging.StreamHandler() # Command line handler
    handler.setLevel("DEBUG")

    # Creating Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Main Paths
    logger.info("Starting the creation of main paths")
    logger.debug("Resolving the current directory path")
    curr_dir = pathlib.Path(__file__).resolve() # resolve -> make the path absolute
    home_dir = curr_dir.parent.parent.parent
    logger.debug("Current directory path resolved successfully")

    # Params path
    logger.debug("Creating the path for params.yaml")
    params_path = home_dir.as_posix() + "/params.yaml"
    logger.debug("Path for params.yaml created successfully")

    # Data Paths
    logger.debug("Creating paths for data directories")
    data_path = home_dir.as_posix() + "/data" # Return the string representation of the path with forward (/) slashes.
    fetch_data_path = data_path + "/raw"
    store_data_path = data_path + "/interim"
    logger.debug("Paths for data directories created successfully")
    logger.info("All paths created successfully")

    # Loading Parameters
    logger.info("Starting to load parameters from params.yaml")
    params = load_params(path=params_path, logger=logger)
    logger.info("Parameters loaded successfully from params.yaml")  

    # Loading the Data
    logger.info("Starting to load the dataset from creditcard.csv")
    data = load_data(path=fetch_data_path, logger=logger)
    logger.info("Dataset loaded successfully from creditcard.csv")

    # Splitting the Data
    logger.info("Starting to split the dataset into training and testing sets")
    train, test = split_data(data, params, logger)
    logger.info("Dataset split successfully into training and testing sets")

    # Saving the Data
    logger.info("Starting to save the training and testing datasets")
    save_data(train, test, store_data_path, logger)
    logger.info("Training and testing datasets saved successfully")

    return 0

if __name__ == "__main__":
    main()