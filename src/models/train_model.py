import pathlib
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Tuple, Dict, Optional, Any

class ModelTrainer:
    def __init__(self):
        # Set up paths
        self.curr_dir = pathlib.Path(__file__).resolve()
        self.home_dir = self.curr_dir.parent.parent.parent
        self.params_path = self.home_dir.as_posix() + "/params.yaml"
        self.data_path = self.home_dir.as_posix() + "/data"
        self.training_data_path = self.data_path + "/processed"
        self.model_path = self.home_dir.as_posix() + "/models"
        
        # Initialize variables
        self.params = None
        self.X = None
        self.y = None
        self.model = None

    def load_params(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load model parameters from YAML file"""
        if path is None:
            path = self.params_path
            
        try:
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
                self.params = params['train_model']
                return self.params
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found at path: {path}")
        except Exception as e:
            raise Exception(f"Unexpected error loading parameters: {e}")

    def load_data(self, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        if path is None:
            path = self.training_data_path
            
        try:
            train = pd.read_csv(path + "/train.csv")

            try:
                self.X = train.drop(columns=['Class'])
                self.y = train['Class']
                return self.X, self.y
            except KeyError as e:
                raise KeyError(f"Required column 'Class' not found in data: {e}")
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Train data file not found at: {path}")
        except Exception as e:
            raise Exception(f"Error loading training data: {e}")

    def train_model(self, X_train: Optional[pd.DataFrame] = None, 
                   y_train: Optional[pd.Series] = None, 
                   params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
        """Train the machine learning model"""
        if X_train is None:
            X_train = self.X
        if y_train is None:
            y_train = self.y
        if params is None:
            params = self.params
            
        try:
            self.model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=params['seed']
            )

            self.model.fit(X_train, y_train)
            
            return self.model
        except KeyError as e:
            raise KeyError(f"Required parameter not found in configuration: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid value during model training: {e}")
        except Exception as e:
            raise Exception(f"Error during model training: {e}")

    def save_model(self, model: Optional[RandomForestClassifier] = None, 
                  path: Optional[str] = None) -> None:
        """Save trained model to disk"""
        if model is None:
            model = self.model
        if path is None:
            path = self.model_path
            
        try:
            # Create directory if it doesn't exist
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model, path + "/model.joblib")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory not found for saving model: {path}")
        except Exception as e:
            raise Exception(f"Error saving model: {e}")

    def process(self) -> None:
        """Execute the full model training pipeline"""
        try:
            self.load_params()
            self.load_data()
            self.train_model()
            self.save_model()
        except Exception as e:
            raise Exception(f"Error in model training pipeline: {e}")

def main():
    try:
        model_trainer = ModelTrainer()
        model_trainer.process()
    except Exception as e:
        raise

if __name__ == '__main__':
    main()
