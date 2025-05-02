import numpy as np
import pandas as pd
import joblib
import pathlib
from sklearn import metrics
from dvclive import Live

class CreditCardFraudVisualizer:
    """
    A class for visualizing and evaluating credit card fraud detection models.
    """
    
    def __init__(self, data_path=None, model_path=None, output_path=None):
        """
        Initialize the visualizer with paths to data, model, and output directory.
        
        Args:
            data_path (str, optional): Path to the processed data directory
            model_path (str, optional): Path to the saved model file
            output_path (str, optional): Path to the output directory for visualizations
        """
        # Set default paths if not provided
        if data_path is None or model_path is None or output_path is None:
            curr_dir = pathlib.Path(__file__).resolve()
            home_dir = curr_dir.parent.parent.parent
            
            if data_path is None:
                self.data_path = home_dir.as_posix() + "/data/processed"
            else:
                self.data_path = data_path
                
            if model_path is None:
                self.model_path = home_dir.as_posix() + "/models/model.joblib"
            else:
                self.model_path = model_path
                
            if output_path is None:
                self.output_path = home_dir.as_posix() + '/dvclive'
            else:
                self.output_path = output_path
        else:
            self.data_path = data_path
            self.model_path = model_path
            self.output_path = output_path
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> tuple:
        """
        Load and process the training and testing data.
        
        Returns:
            tuple: Processed training and testing data (X_train, X_test, y_train, y_test)
            
        Raises:
            KeyError: If required column 'Class' is not found in data
            FileNotFoundError: If train or test data files are not found
            Exception: For other errors during data loading
        """
        try:
            # Open and read the train and test files
            train_path = self.data_path + "/train.csv"
            test_path = self.data_path + "/test.csv"
            
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            
            # Process the dataframes
            self.X_train = train.drop(columns=['Class'])
            self.y_train = train['Class']
            self.X_test = test.drop(columns=['Class'])
            self.y_test = test['Class']
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        except KeyError as e:
            raise KeyError(f"Required column 'Class' not found in data: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Train or test data files not found at: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading train/test data: {e}")
    
    def load_model(self):
        """
        Load the trained model from the specified path.
        
        Returns:
            object: The loaded model
            
        Raises:
            FileNotFoundError: If model file is not found
            Exception: For other errors during model loading
        """
        try:
            self.model = joblib.load(self.model_path)
            return self.model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def evaluate(self, X, y, live, split):
        """
        Evaluate the model performance and log metrics.
        
        Args:
            X (DataFrame): Feature data
            y (Series): Target data
            live (Live): DVC Live instance for logging
            split (str): Data split name ('train' or 'test')
        """
        # Prediction
        prediction_by_class = self.model.predict_proba(X)
        predictions = prediction_by_class[:, 1]  # Probability of fraud (class 1)

        # Evaluation Metrics
        avg_prec = metrics.average_precision_score(y, predictions)
        roc_auc = metrics.roc_auc_score(y, predictions)

        # DVC Logging
        if not live.summary:
            live.summary = {
                "avg_prec": {},
                "roc_auc": {}
            }
        live.summary['avg_prec'][split] = avg_prec
        live.summary['roc_auc'][split] = roc_auc

        # DVC logging - ROC curve
        live.log_sklearn_plot(
            "roc", y, predictions, name=f"roc/{split}"
        )

        # DVC Logging - Precision Recall Curve
        live.log_sklearn_plot(
            "precision_recall", y, predictions, name=f"prc/{split}", drop_intermediate=True,
        )

        # DVC Logging - Confusion Matrix
        live.log_sklearn_plot(
            "confusion_matrix", y, prediction_by_class.argmax(-1), name=f"cm/{split}"
        )
        
        return {
            'avg_precision': avg_prec,
            'roc_auc': roc_auc
        }

    def run(self):
        """
        Run the complete visualization and evaluation pipeline.
        """
        # Load data and model if not already loaded
        if self.X_train is None or self.X_test is None:
            self.load_data()
        
        if self.model is None:
            self.load_model()

        # Evaluate model on train and test data
        with Live(self.output_path, dvcyaml=False) as live:
            # Train evaluation
            train_metrics = self.evaluate(
                X=self.X_train,
                y=self.y_train,
                live=live,
                split="train",
            )

            # Test evaluation
            test_metrics = self.evaluate(
                X=self.X_test,
                y=self.y_test,
                live=live,
                split="test",
            )
            
        return {
            'train': train_metrics,
            'test': test_metrics
        }

def main():
    """
    Main function to run the visualization pipeline.
    """
    visualizer = CreditCardFraudVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()