import pathlib
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_params(path: str, logger) -> dict:
    try:
        with open("params.yaml", 'r') as f:
            logger.debug("Loading parameters from params.yaml")
            params = yaml.safe_load(f)
            logger.debug("Parameters loaded from params.yaml")
            return params['train_model']
    except yaml.YAMLError as e:
        logger.error(f"Error parsing params.yaml: {e}")
        raise yaml.YAMLError
    except FileNotFoundError as e:
        logger.error(f"params.yaml not found: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise Exception

def load_data(path: str, logger) -> tuple:
    try:
        logger.debug(f"Loading datasets from path: {path}")
        train = pd.read_csv(path + "/train.csv")
        logger.info("Datasets loaded from path")

        try:
            X = train.drop(columns=['Class'])
            y = train['Class']
            return X, y
        except KeyError as e:
            logger.error(f"Missing 'Class' column: {e}")
            raise KeyError
    
    except FileNotFoundError as e:
        logger.error(f"File not found at path: {path}. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"Unexpected error loading datasets: {e}")
        raise Exception

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict, logger):
    try:
        logger.debug("Initializing RandomForestClassifier")
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['seed']
        )

        logger.debug("Fitting model to training data")
        model.fit(X_train, y_train)
        logger.debug("Model training complete")
        
        return model
    except KeyError as e:
        logger.error(f"Missing parameter: {e}")
        raise KeyError
    except ValueError as e:
        logger.error(f"Error during model training: {e}")
        raise ValueError
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise Exception

def save_model(model, path: str, logger) -> None:
    logger.debug("Saving model to file")
    joblib.dump(model, path + "/model.joblib")
    logger.debug("Model saved to file")

def main():
    logger = logging.getLogger("Model Training Logger")
    logger.setLevel("DEBUG")

    handler = logging.StreamHandler()
    handler.setLevel("DEBUG")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Starting model training process")
    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parent.parent.parent

    params_path = home_dir.as_posix() + "/params.yaml"
    data_path = home_dir.as_posix() + "/data"
    training_data_path = data_path + "/processed"
    model_path = home_dir.as_posix() + "/models"

    logger.info("Loading parameters")
    params = load_params(path=params_path, logger=logger)
    logger.info("Parameters loaded")

    logger.info("Loading dataset")
    X, y = load_data(path=training_data_path, logger=logger)
    logger.info("Dataset loaded")

    logger.info("Training model")
    model = train_model(X, y, params, logger)
    logger.info("Model trained")

    logger.info("Saving model")
    save_model(model, model_path, logger)
    logger.info("Model saved")

if __name__ == '__main__':
    main()
