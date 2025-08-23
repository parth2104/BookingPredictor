import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.Exception.exception import CustomException
from src.logger.logger import logging
from config.path_config import *
from utils.common_functions import MainUtils
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

class DataTransformation:
    def __init__(self, train_file_path, test_file_path, PREPROCESSED_DIR, CONFIG_PATH, encoder_path):
        self.train_df = train_file_path
        self.test_df = test_file_path
        self.preprocess_dir = PREPROCESSED_DIR
        self.utils = MainUtils()
        self.config = self.utils.read_yaml(CONFIG_PATH)
        self.encoder_path = encoder_path

        if not os.path.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
            logging.info(f"Created preprocessing directory: {self.preprocess_dir}")

    def data_cleaning(self, data):
        try:
            data.drop(columns=["Booking_ID"], inplace=True, errors="ignore")
            data.drop_duplicates(inplace=True)
            data = data.reset_index(drop=True)
            logging.info("Cleaned data (dropped Booking_ID and duplicates, reset index).")
            return data
        except Exception as e:
            logging.error(f"Data cleaning failed: {e}")
            raise CustomException(str(e), sys)

    def apply_encoding(self, train_df, test_df, cat_cols, save_path):
        try:
            logging.info(f"Categorical columns for encoding: {cat_cols}")
            if "booking_status" in cat_cols:
                raise ValueError("Target column 'booking_status' should not be in categorical columns")
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
            train_df[cat_cols] = ordinal_encoder.fit_transform(train_df[cat_cols].astype(str))
            test_df[cat_cols] = ordinal_encoder.transform(test_df[cat_cols].astype(str))

            joblib.dump(ordinal_encoder, save_path)
            logging.info(f"Ordinal encoder saved to {save_path}")
            return train_df, test_df
        except Exception as e:
            logging.error(f"Encoding failed: {e}")
            raise CustomException(str(e), sys)

    def balance_data(self, data):
        try:
            if "booking_status" not in data.columns:
                raise ValueError("Target column 'booking_status' not found in data")
            x = data.drop(columns=["booking_status"])
            y = data["booking_status"]

            smote = SMOTE(random_state=43)
            x_resampled, y_resampled = smote.fit_resample(x, y)

            balanced_data = pd.DataFrame(x_resampled, columns=x.columns)
            balanced_data["booking_status"] = y_resampled
            logging.info("Balanced data using SMOTE.")
            return balanced_data
        except Exception as e:
            logging.error(f"Data balancing failed: {e}")
            raise CustomException(str(e), sys)

    def feature_selection(self, data):
        try:
            if "booking_status" not in data.columns:
                raise ValueError("Target column 'booking_status' not found in data")
            x = data.drop(columns=["booking_status"])
            y = data["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)

            important_features = pd.DataFrame({
                "feature": x.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            top_n_features = self.config["data_preprocessing"]["number_of_feature"]
            selected_features = important_features["feature"].head(top_n_features).tolist()

            if "booking_status" in selected_features:
                selected_features.remove("booking_status")
                logging.warning("Removed 'booking_status' from selected features.")
            
            # Reorder selected_features to match num_columns + cat_columns
            cat_cols = self.config["data_preprocessing"]["categorical"]
            num_cols = self.config["data_preprocessing"]["numerical"]
            ordered_features = [col for col in num_cols + cat_cols if col in selected_features]
            logging.info(f"Selected top {top_n_features} features (ordered): {ordered_features}")

            self.utils.save_yaml({"selected_features": ordered_features}, feature_selected_path)
            logging.info(f"Selected features saved to {feature_selected_path}")

            return ordered_features
        except Exception as e:
            logging.error(f"Feature selection failed: {e}")
            raise CustomException(str(e), sys)

    def preprocessing(self, num_columns, cat_columns):
        try:
            if "booking_status" in num_columns or "booking_status" in cat_columns:
                raise ValueError("Target column 'booking_status' should not be in num_columns or cat_columns")
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("numerical", num_pipeline, num_columns),
                ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int), cat_columns)
            ])
            logging.info(f"Preprocessor created with numeric: {num_columns}, categorical: {cat_columns}")
            return preprocessor
        except Exception as e:
            logging.error(f"Preprocessing pipeline creation failed: {e}")
            raise CustomException(str(e), sys)

    def main(self):
        try:
            train_df = self.utils.load_data(self.train_df)
            test_df = self.utils.load_data(self.test_df)
            logging.info("Loaded train and test data.")

            train_df = self.data_cleaning(train_df)
            test_df = self.data_cleaning(test_df)

            cat_cols = self.config["data_preprocessing"]["categorical"]
            num_cols = self.config["data_preprocessing"]["numerical"]
            logging.info(f"Config categorical columns: {cat_cols}")
            logging.info(f"Config numerical columns: {num_cols}")

            train_df, test_df = self.apply_encoding(train_df, test_df, cat_cols, self.encoder_path)

            train_resampled = self.balance_data(train_df)

            selected_features = self.feature_selection(train_resampled)
            logging.info(f"Features selected for preprocessing: {selected_features}")

            x_train = train_resampled[selected_features]
            y_train = train_resampled["booking_status"]
            x_test = test_df[selected_features]
            y_test = test_df["booking_status"]

            cat_columns = [col for col in cat_cols if col in selected_features]
            num_columns = [col for col in selected_features if col not in cat_columns]
            logging.info(f"Numeric columns for preprocessor: {num_columns}")
            logging.info(f"Categorical columns for preprocessor: {cat_columns}")

            preprocessor_obj = self.preprocessing(num_columns=num_columns, cat_columns=cat_columns)

            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)

            train_arr = np.c_[x_train_transformed, y_train]
            test_arr = np.c_[x_test_transformed, y_test]

            self.utils.save_preprocessor(preprocessor_obj, preprocessor_obj_path)
            self.utils.save_npz(preprocessed_train_path, arr=train_arr)
            self.utils.save_npz(preprocessed_test_path, arr=test_arr)

            self.utils.save_yaml({
                "feature_columns": num_columns + cat_columns,
                "target_column": "booking_status"
            }, os.path.join(PREPROCESSED_DIR, "columns.yaml"))
            logging.info(f"Saved column names to {os.path.join(PREPROCESSED_DIR, 'columns.yaml')}")

            logging.info("Data preprocessing completed successfully.")
            return preprocessor_obj, train_arr, test_arr

        except Exception as e:
            logging.error(f"Data transformation failed: {e}")
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    data_transformer = DataTransformation(
        train_file_path=train_file_path,
        test_file_path=test_file_path,
        PREPROCESSED_DIR=PREPROCESSED_DIR,
        CONFIG_PATH=CONFIG_PATH,
        encoder_path=path_ordinal_encoder
    )
    preprocessor_obj, train_arr, test_arr = data_transformer.main()