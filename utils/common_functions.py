import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as cm
from src.Exception.exception import CustomException
from src.logger.logger import logging
from config.path_config import MODEL_PATH


class MainUtils:

    def read_yaml(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")
            logging.info(f"File exists: {file_path}")
            with open(file_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
                logging.info("Successfully read YAML file.")
                return config
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

    def load_data(self, path):
        try:
            logging.info(f"Loading data from: {path}")
            data = pd.read_csv(path)
            logging.info(f"Data sample:\n{data.head()}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)

    def save_preprocessor(self, obj, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(obj, file_path)
            logging.info(f"Preprocessor saved successfully at {file_path}")
            return file_path
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def load_obj(self, filename):
        try:
            return joblib.load(filename)
        except Exception as e:
            raise CustomException(str(e), sys)

    def save_npz(self, file_path, **arrays):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.savez_compressed(file_path, **arrays)
            return file_path
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def load_npz(self, filename):
        try:
            data = np.load(filename, allow_pickle=True)
            if len(data.files) == 1:
                return data[data.files[0]]
            raise ValueError("NPZ file does not contain exactly one array.")
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

    @staticmethod
    def get_base_model(model_name):
        try:
            if model_name == "XGBClassifier":
                from xgboost import XGBClassifier
                return XGBClassifier()

            all_models = dict(all_estimators())
            if model_name not in all_models:
                raise ValueError(f"{model_name} is not a valid sklearn model.")
            return all_models[model_name]()  # instantiate
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

    def get_params(self, model, x_train, y_train):
        try:
            model_name = model.__class__.__name__
            model_config = self.read_yaml(MODEL_PATH)
            model_params = model_config.get("training", {}).get(model_name)
            if not model_params:
                raise ValueError(f"No parameters found for model: {model_name}")

            grid = GridSearchCV(model, model_params, cv=2, verbose=3, n_jobs=-1)
            grid.fit(x_train, y_train)

            logging.info(f"Best params for {model_name}: {grid.best_params_}")
            return grid.best_params_
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

    def get_model_score(self, y_test, pred, average):
        logging.info("Calculating model evaluation metrics...")
        try:
            accuracy = np.round(accuracy_score(y_test, pred), 4)
            precision = np.round(precision_score(y_test, pred, average=average, zero_division=0), 4)
            recall = np.round(recall_score(y_test, pred, average=average, zero_division=0), 4)
            f1 = np.round(f1_score(y_test, pred, average=average, zero_division=0), 4)
            conf_matrix = cm(y_test, pred).tolist()

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix
            }
            logging.info(f"Model Metrics: {metrics}")
            return metrics
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def get_tuned_model(self, model_name, x_train, y_train, x_test, y_test):
        try:
            model = self.get_base_model(model_name)
            params = self.get_params(model, x_train, y_train)
            model.set_params(**params)
            logging.info(f"{model_name} - Parameters set: {params}")

            cv_score = cross_val_score(model, x_train, y_train, cv=5)
            logging.info(f"{model_name} - Cross-validation scores: {cv_score}")
            logging.info(f"{model_name} - Mean CV score: {np.mean(cv_score)}")

            model.fit(x_train, y_train)
            pred = model.predict(x_test if isinstance(x_test, pd.DataFrame) else pd.DataFrame([x_test]))


            metrics = self.get_model_score(y_test, pred, average="weighted")
            accuracy = metrics["accuracy"]

            return (model_name, model, metrics, accuracy)
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            return None

    def get_best_model_with_name_and_score(self, model_list):
        try:
            if not model_list:
                raise CustomException("Model list is empty", sys)

            best_model_tuple = max(model_list, key=lambda x: x[3])  # score at index 3
            best_model_name = best_model_tuple[0]
            best_model_object = best_model_tuple[1]
            best_model_metrics = best_model_tuple[2]
            best_model_score = best_model_tuple[3]

            logging.info(f"Best model selected: {best_model_name} with score: {best_model_score}")
            return best_model_metrics, best_model_object, best_model_name, best_model_score
        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

    def update_model_score(self, best_model_score):
        logging.info("Updating base model score in YAML...")
        try:
            model_config = self.read_yaml(MODEL_PATH)
            model_config["base_model_score"] = str(best_model_score)
            with open(MODEL_PATH, "w") as fp:
                yaml.safe_dump(model_config, fp, sort_keys=False)
            logging.info(f"Updated base model score to {best_model_score}")
        except Exception as e:
            raise CustomException(str(e), sys)
