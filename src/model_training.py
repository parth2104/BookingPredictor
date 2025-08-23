import os
import sys
import pandas as pd
import numpy as np
from src.Exception.exception import CustomException
from src.logger.logger import logging
from src.cost_model import CostModel
from utils.common_functions import MainUtils
from config.path_config import *
import mlflow
from dotenv import load_dotenv
import dagshub
import warnings

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, preprocessed_train_path, preprocessed_test_path, MODEL_PATH, preprocessor_obj_path, path_ordinal_encoder, feature_selected_path, CONFIG_PATH):
        self.train_data = preprocessed_train_path
        self.test_data = preprocessed_test_path
        self.preprocessor_obj = preprocessor_obj_path
        self.ordinal_encoder = path_ordinal_encoder
        self.model_path = MODEL_PATH
        self.feature_selected_path = feature_selected_path
        self.config_path = CONFIG_PATH

        self.utils = MainUtils()
        self.read_config_yaml = self.utils.read_yaml(self.config_path)
        logging.info("Loaded YAML config successfully")

    def get_trained_model(self, train_data, test_data):
        try:
            model_config = self.utils.read_yaml(MODEL_PATH)
            logging.info(f"Loaded model_config: {model_config}")

            model_list = list(model_config.get("training", {}).keys())
            logging.info(f"Training models: {model_list}")

            logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

            x_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]
            x_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            trained_models = [
                self.utils.get_tuned_model(model_name, x_train, y_train, x_test, y_test)
                for model_name in model_list
            ]

            logging.info("All models trained successfully.")
            return trained_models

        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise CustomException(str(e), sys)

    def model_trainer_initiate(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            train_arr = pd.DataFrame(self.utils.load_npz(self.train_data))
            test_arr = pd.DataFrame(self.utils.load_npz(self.test_data))

            logging.info(f"Loaded train_arr shape: {train_arr.shape}")
            logging.info(f"Loaded test_arr shape: {test_arr.shape}")

            trained_models = self.get_trained_model(train_arr, test_arr)

            best_model_metrics, best_model_object, best_model_name, best_model_score = \
                self.utils.get_best_model_with_name_and_score(trained_models)

            best_model_params = best_model_object.get_params()
            model_config = self.utils.read_yaml(MODEL_PATH)
            base_model_score = float(model_config.get("base_model_score", 0))

            selected_data = self.utils.read_yaml(self.feature_selected_path)
            selected_features = selected_data.get("selected_features", [])
            if "booking_status" in selected_features:
                selected_features.remove("booking_status")
                logging.warning("Removed 'booking_status' from selected_features in ModelTrainer.")
                self.utils.save_yaml({"selected_features": selected_features}, self.feature_selected_path)
                logging.info(f"Updated selected features saved to {self.feature_selected_path}")
            logging.info(f"Using selected_features: {selected_features}")

            columns_yaml = self.utils.read_yaml(os.path.join(PREPROCESSED_DIR, "columns.yaml"))
            preprocessor_features = columns_yaml.get("feature_columns", [])
            if "booking_status" in preprocessor_features:
                raise ValueError("Target column 'booking_status' found in preprocessor feature_columns")
            logging.info(f"Preprocessor feature_columns: {preprocessor_features}")

            with mlflow.start_run(run_name="Best Model Logging"):
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("best_model_score", best_model_score)

                for param_name, param_value in best_model_params.items():
                    mlflow.log_param(param_name, param_value)

                if isinstance(best_model_metrics, dict):
                    for metric_name, metric_value in best_model_metrics.items():
                        if isinstance(metric_value, list):
                            mlflow.log_metric(metric_name, float(np.mean(metric_value)))
                        else:
                            mlflow.log_metric(metric_name, float(metric_value))

            if best_model_score >= base_model_score:
                self.utils.update_model_score(best_model_score)
                logging.info("Updated best_model_score in config")

                ordinal_encoder = self.utils.load_obj(self.ordinal_encoder)
                preprocessor_obj = self.utils.load_obj(self.preprocessor_obj)

                cost_model = CostModel(ordinal_encoder, preprocessor_obj, best_model_object, selected_features)
                logging.info(f"Created cost_model object with feature_order: {selected_features}")

                model_file_path = self.utils.save_preprocessor(cost_model, self.model_path)

                logging.info(f"Model saved at {model_file_path}")
                return {
                    "model_path": model_file_path,
                    "model_name": best_model_name,
                    "score": best_model_score,
                    "metrics": best_model_metrics
                }

            else:
                logging.warning("No better model found than base model score.")
                return {
                    "model_path": None,
                    "model_name": None,
                    "score": None,
                    "metrics": None,
                    "message": "No model surpassed the base model score."
                }

        except Exception as e:
            logging.error(f"Model trainer initiation failed: {e}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    load_dotenv()
    dagshub.init(repo_owner='parth2104', repo_name='BookingPredictor', mlflow=True)

    trainer = ModelTrainer(
        preprocessed_train_path=preprocessed_train_path,
        preprocessed_test_path=preprocessed_test_path,
        MODEL_PATH=model_out_put_path,
        preprocessor_obj_path=preprocessor_obj_path,
        path_ordinal_encoder=path_ordinal_encoder,
        feature_selected_path=feature_selected_path,
        CONFIG_PATH=CONFIG_PATH
    )
    trainer.model_trainer_initiate()