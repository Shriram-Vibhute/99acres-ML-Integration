# Importing required libraries
import numpy as np
import pandas as pd
import pathlib
import yaml
from sklearn.model_selection import train_test_split

class DatasetMaker:
    def __init__(self):
        # Main Paths
        curr_dir = pathlib.Path(__file__).resolve()
        self.home_dir = curr_dir.parent.parent.parent
        
        # Params Path
        self.params_path = self.home_dir.as_posix() + "/params.yaml"
        
        # Data Path
        self.data_path = self.home_dir.as_posix() + "/data"
        self.fetch_data_path = self.data_path + "/raw"
        self.store_data_path = self.data_path + "/interim"
        
        # Parameters
        self.params = None
        self.data = None
        self.train = None
        self.test = None

    def load_params(self, path=None):
        """Load parameters from YAML file"""
        if path is None:
            path = self.params_path
            
        try:
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
                self.params = params['make_dataset']
                return self.params
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found at path: {path}")
        except Exception as e:
            raise Exception(f"Unexpected error loading parameters: {e}")

    def load_data(self, path=None):
        """Load credit card data from CSV file"""
        if path is None:
            path = self.fetch_data_path
            
        try:
            with open(path + "/creditcard.csv", 'r') as f:
                self.data = pd.read_csv(f)
                return self.data
        except FileNotFoundError as e:
            raise FileExistsError(f"Credit card data file not found at: {path}/creditcard.csv")
        except Exception as e:
            raise Exception(f"Error loading credit card data: {e}")

    def split_data(self, data=None, params=None):
        """Split data into training and test sets"""
        if data is None:
            data = self.data
        if params is None:
            params = self.params
            
        try:
            self.train, self.test = train_test_split(
                data, 
                test_size=params['test_split'], 
                random_state=params['seed']
            )
        except KeyError as e:
            missing_key = str(e)
            raise KeyError(f"Required parameter not found in configuration: {missing_key}")
        return self.train, self.test

    def save_data(self, train=None, test=None, path=None):
        """Save training and test data to CSV files"""
        if train is None:
            train = self.train
        if test is None:
            test = self.test
        if path is None:
            path = self.store_data_path
        
        # Create directory if not exists
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            
        try:
            train.to_csv(path + "/train.csv", index=False)
            test.to_csv(path + "/test.csv", index=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Directory not found for saving data: {path}")
        except Exception as e:
            raise Exception(f"Error saving train/test data: {e}")

    def process(self):
        """Execute the full data processing pipeline"""
        try:
            # Loading Params
            self.load_params()
            
            # Loading Data
            self.load_data()
            
            # Splitting Data
            self.split_data()
            
            # Saving Data
            self.save_data()
        
        except Exception as e:
            raise Exception(f"Error in data processing pipeline: {e}")

def main():
    try:
        dataset_maker = DatasetMaker()
        dataset_maker.process()
    except Exception as e:
        raise Exception(f"Unexpected error happened: {e}")

if __name__ == "__main__":
    main()# This is considered a Python best practice and is commonly used in scripts that can be both imported as modules and run as standalone programs.
# Make the file both importable as a module and executable as a script
