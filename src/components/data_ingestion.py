import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:
            # Read dataset
            df = pd.read_csv('notebook/StudentsPerformance.csv')
            logging.info("Dataset loaded into dataframe")

            # Create directories for saving data
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Perform train-test split
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            logging.error("An error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

