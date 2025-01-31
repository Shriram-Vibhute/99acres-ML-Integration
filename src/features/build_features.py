import numpy as np
import pandas as pd
import logging
import pathlib

def load_data(path: str, logger) -> pd.DataFrame:
    try:
        logger.debug(f"Attempting to load the training and testing datasets from the specified path: {path}")
        train = pd.read_csv(path + "/train.csv")
        test = pd.read_csv(path + "/test.csv")
        logger.info("Training and testing datasets loaded successfully from the specified path")
        return train, test
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The file was not found in the specified path: {path}. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the datasets from the specified path: {path}. Error: {e}")
        raise Exception

def build_feature(data: pd.DataFrame, logger):
    try:
        logger.debug("Attempting to build new features from the dataset")
        data_for_features = data.loc[:, [f"V{i}" for i in range(1, 28)]]
        new_feature = np.sum(data_for_features, axis=1) / 28
        data['V_avg'] = new_feature
        logger.info("New feature 'V_avg' built successfully and added to the dataset")
        return data
    except KeyError as e:
        logger.error(f"KeyError: One or more keys were not found in the dataset. Error: {e}")
        raise KeyError
    except Exception as e:
        logger.error(f"An unexpected error occurred while building features from the dataset. Error: {e}")
        raise Exception

def save_data(train: pd.DataFrame, test: pd.DataFrame, path: str, logger) -> None:
    try:
        logger.debug(f"Attempting to save the training and testing datasets to the specified path: {path}")
        train.to_csv(path + "/train.csv", index=False)
        test.to_csv(path + "/test.csv", index=False)
        logger.info("Training and testing datasets saved successfully to the specified path")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: The specified path for saving datasets was not found: {path}. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the datasets to the specified path: {path}. Error: {e}")
        raise Exception

def main() -> None:
    # Creating Logger
    logger = logging.getLogger("Build Feature Logger")
    logger.setLevel("DEBUG")

    # Creating Handler
    handler = logging.StreamHandler() # Command line handler
    handler.setLevel("DEBUG")

    # Creating Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Main Paths
    logger.info("Starting the process of creating main paths for data ingestion")
    logger.debug("Resolving the current directory path to an absolute path")
    curr_dir = pathlib.Path(__file__).resolve() # resolve -> make the path absolute
    home_dir = curr_dir.parent.parent.parent
    logger.debug("Current directory path resolved successfully to an absolute path")

    # Data Paths
    logger.debug("Creating paths for data directories based on the home directory")
    data_path = home_dir.as_posix() + "/data" 
    fetch_data_path = data_path + "/interim"
    store_data_path = data_path + "/processed"
    logger.debug("Paths for data directories created successfully based on the home directory")
    logger.info("All paths for data directories created successfully")

    # Loading Data
    logger.info("Starting the process of loading the training and testing datasets")
    train, test = load_data(fetch_data_path, logger)
    logger.info("Training and testing datasets loaded successfully")

    # Building Features
    logger.info("Starting the process of building new features for the training and testing datasets")
    train_features = build_feature(train, logger)
    test_features = build_feature(test, logger)
    logger.info("New features built successfully for both training and testing datasets")

    # Saving Data
    logger.info("Starting the process of saving the training and testing datasets with new features")
    save_data(train_features, test_features, store_data_path, logger)
    logger.info("Training and testing datasets with new features saved successfully")

if __name__ == "__main__":
    main()