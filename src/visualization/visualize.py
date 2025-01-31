import pathlib
import pandas as pd
import joblib
import logging
import numpy as np
from sklearn import metrics
from dvclive import Live

def load_data(path: str, logger) -> tuple:
    try:
        logger.debug(f"Attempting to load datasets from path: {path}")
        train = pd.read_csv(path + "/train.csv")
        test = pd.read_csv(path + "/test.csv")
        logger.info("Datasets successfully loaded from the specified path")

        try:
            X_train = train.drop(columns=['Class'])
            y_train = train['Class']
            X_test = test.drop(columns=['Class'])
            y_test = test['Class']
            return X_train, X_test, y_train, y_test
        except KeyError as e:
            logger.error(f"Failed to find 'Class' column in the dataset: {e}")
            raise KeyError
    
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found at path: {path}. Error: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading datasets: {e}")
        raise Exception

def load_model(path: str, logger):
    try:
        logger.debug("Attempting to load the model")
        with open(path, 'r') as f:
            model = joblib.load(path)
            logger.debug("Model loaded successfully")
            return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise Exception
    
def evaluate(model, X, y, live, split, logger):
    # Predictions
    logger.info("Starting model predictions")
    prediction_by_class = model.predict_proba(X)
    predictions = prediction_by_class[:, 1] # Fetching the probability of fraud(Yes) credit cards
    logger.info("Model predictions completed successfully")

    # Evaluation Metrics
    logger.info("Calculating evaluation metrics")
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)
    logger.info("Evaluation metrics calculated successfully")

    # DVC Logging
    logger.info("Logging evaluation metrics to DVC")
    if not live.summary:
        live.summary = {
            "avg_prec": {},
            "roc_auc": {}
        }
    live.summary['avg_prec'][split] = avg_prec
    live.summary['roc_auc'][split] = roc_auc
    logger.info("Evaluation metrics logged to DVC successfully")

    # DVC logging - ROC curve
    logger.info("Logging ROC curve to DVC")
    live.log_sklearn_plot(
        "roc", y, predictions, name = f"roc/{split}"
    )

    # DVC Logging - Precision Recall Curve
    logger.info("Logging Precision-Recall curve to DVC")
    live.log_sklearn_plot(
        "precision_recall", y, predictions, name=f"prc/{split}", drop_intermediate = True,
    )

    # DVC Logging - Confusion Matrix
    logger.info("Logging confusion matrix to DVC")
    live.log_sklearn_plot(
        "confusion_matrix", y, prediction_by_class.argmax(-1), name=f"cm/{split}"
    )

def main():
    logger = logging.getLogger("Visualization Logger")
    logger.setLevel("DEBUG")

    handler = logging.StreamHandler()
    handler.setLevel("DEBUG")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Define the paths
    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir.as_posix() + "/data"
    data_path = data_path + "/processed"
    model_path = home_dir.as_posix() + "/models/model.joblib"

    logger.info("Initiating data loading process")
    X_train, X_test, y_train, y_test = load_data(data_path, logger)
    logger.info("Data loading process completed successfully")

    logger.info("Initiating model loading process")
    model = load_model(model_path, logger)
    logger.info("Model loading process completed successfully")

    output_path = home_dir.as_posix() + '/dvclive'

    with Live(output_path, dvcyaml = False) as live:
        # Train
        evaluate(
            model = model,
            X = X_train,
            y = y_train,
            live = live,
            split = "train",
            logger = logger
        )

        # Test
        evaluate(
            model = model,
            X = X_test,
            y = y_test,
            live = live,
            split = "test",
            logger = logger
        )
    logger.info("Evaluation process completed successfully")

if __name__ == "__main__":
    main()