import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessors_path: str = os.path.join('artifacts', 'preprocessors.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer(self):
        """
        Creates and returns a preprocessor object for data transformation.
        """
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            # Define pipelines for numerical and categorical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('std_scaler', StandardScaler(with_mean=False))  # for sparse matrices
                ]
            )

            logging.info("Pipelines for numerical and categorical features created")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error("An error occurred while creating the data transformer")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Transforms the training and testing datasets.
        """
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            preprocessing_obj = self.get_data_transformer()

            target_column_name = "math score"
            numerical_columns = ['reading score', 'writing score']

            # Separate input features and target variable for train and test data
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Input features and target variable separated")

            # Fit-transform the train data and transform the test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Combine transformed features and target variable
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save the preprocessor object
            save_object(
                file_path=self.config.preprocessors_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor object saved")

            return train_arr, test_arr, self.config.preprocessors_path
        except Exception as e:
            logging.error("An error occurred during data transformation")
            raise CustomException(e, sys)
