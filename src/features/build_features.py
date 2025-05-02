import numpy as np
import pandas as pd
import pathlib
import yaml
from typing import Tuple, Optional

class FeatureBuilder:
    def __init__(self):
        # Main Paths
        self.curr_dir = pathlib.Path(__file__).resolve()  # resolve -> make the path absolute
        self.home_dir = self.curr_dir.parent.parent.parent
        
        # Params Path
        self.params_path = self.home_dir.as_posix() + "/params.yaml"
        
        # Data Paths
        self.data_path = self.home_dir.as_posix() + "/data" 
        self.fetch_data_path = self.data_path + "/interim"
        self.store_data_path = self.data_path + "/processed"

        # Variables
        self.params = None
        self.train = None
        self.test = None
        self.train_features = None
        self.test_features = None
        
        # Load parameters
        self.load_params()

    def load_params(self, path: Optional[str] = None) -> dict:
        """Load parameters from YAML file"""
        if path is None:
            path = self.params_path
            
        try:
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
                self.params = params.get('build_features', {})
                return self.params
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found at path: {path}")
        except Exception as e:
            raise Exception(f"Unexpected error loading parameters: {e}")

    def load_data(self, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data from CSV files"""
        if path is None:
            path = self.fetch_data_path

        try:
            self.train = pd.read_csv(path + "/train.csv")
            self.test = pd.read_csv(path + "/test.csv")
            return self.train, self.test
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Train or test data files not found at: {path}")
        except Exception as e:
            raise Exception(f"Error loading train/test data: {e}")

    def build_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build new features from existing data"""
        try:
            # Create a copy to avoid modifying the original dataframe
            data_copy = data.copy()
            
            # Calculate average of V1-V28 features
            v_columns = [f"V{i}" for i in range(1, 29)]  # Include V28 as well
            data_for_features = data_copy.loc[:, v_columns]
            new_feature = np.sum(data_for_features, axis=1) / len(v_columns)
            data_copy['V_avg'] = new_feature
            
            return data_copy
        except KeyError as e:
            raise KeyError(f"Required columns not found in data: {e}")
        except Exception as e:
            raise Exception(f"Error building features: {e}")

    def save_data(self, train: Optional[pd.DataFrame] = None, test: Optional[pd.DataFrame] = None, 
                  path: Optional[str] = None) -> None:
        """Save processed train and test data to CSV files"""
        if train is None:
            train = self.train_features
        if test is None:
            test = self.test_features
        if path is None:
            path = self.store_data_path

        try:
            # Create directory if it doesn't exist
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)           
            
            train.to_csv(path + "/train.csv", index=False)
            test.to_csv(path + "/test.csv", index=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory not found for saving data: {path}")
        except Exception as e:
            raise Exception(f"Error saving processed train/test data: {e}")
    
    def process(self) -> None:
        """Execute the full feature building pipeline"""
        try:
            # Loading Data
            train, test = self.load_data()

            # Building Features
            self.train_features = self.build_feature(train)
            self.test_features = self.build_feature(test)

            # Saving Data
            self.save_data()
        except Exception as e:
            raise Exception(f"Error in feature building pipeline: {e}")

def main():
    try:
        feature_builder = FeatureBuilder()
        feature_builder.process()
    except Exception as e:
        raise Exception(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()