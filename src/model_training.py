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
import mlflow.sklearn
from dotenv import load_dotenv
import dagshub


class ModelTrainer:

    def __init__(self,preprocessed_train_path,prepprocessor_test_path,MODEL_PATH,preprocessor_obj_path,label_encoder,experiment_name):


        self.test_data=prepprocessor_test_path
        self.train_data=preprocessed_train_path
        self.preprocessor_obj=preprocessor_obj_path
        self.label_encoder=label_encoder
        self.model_path=MODEL_PATH
        self.experiment_name=experiment_name
        
        self.utils=MainUtils()

        self.read_config_yaml=self.utils.read_yaml(CONFIG_PATH)
        logging.info("read yaml file ")

    def get_trained_model(self,train_data,test_data):

        try:

            model_config=self.utils.read_yaml(MODEL_PATH)
            logging.info(f"read model_config : {model_config}")

            model_list=list(model_config.get("training" , {}).keys())
            logging.info(f" get modle list: {model_list}")

            
            logging.info(f"train data shape :{train_data.shape} and test data shape {test_data.shape}")

            x_train=train_data.iloc[:,:-1]
            logging.info(f"x_train {x_train.head(1)}")
            y_train=train_data.iloc[:,-1]
            logging.info(f"y_train {y_train.head(1)}")
            x_test=test_data.iloc[:,:-1]
            logging.info(f"x_train {x_test.head(1)}")
            y_test=test_data.iloc[:,-1]
            logging.info(f"x_train {y_test.head(1)}")


            trained_model=[
                self.utils.get_tuned_model(model_name,x_train,y_train,x_test,y_test)
                for model_name in model_list
            ]

            logging.info("Model training completed.")
            logging.info("Got Trained Model List")
            logging.info("Exited the get_trained_models method of Model_Trainer class")
            return trained_model

        except Exception as e:
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
        
    def model_trainer_initiate(self):

        try:
            os.makedirs(MODEL,exist_ok=True)

            train_arr=pd.DataFrame(self.utils.load_npz(self.train_data))
            logging.info(f"train_arr : {train_arr.head(1)}")
            test_arr=pd.DataFrame(self.utils.load_npz(self.test_data))
            logging.info(f"test_arra : {test_arr.head(1)}")

            logging.info("------------------------------------------------------------------------------------------")

            trained_model=self.get_trained_model(train_arr,test_arr)
            logging.info(f"trained_model : {trained_model}")

            logging.info("--------------------------------------------------------------------------------------------------------------------------------------------")
            best_model_metrics, best_model_object, best_model_name, best_model_score = self.utils.get_best_model_with_name_and_score(trained_model)

            model_config=self.utils.read_yaml(MODEL_PATH)
            base_model_score=float(model_config["base_model_score"])
            #--------------------------------------MLFLOW_START---------------------------------

            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
                
            with mlflow.start_run():
                mlflow.log_param("best_model_name",best_model_name)
                mlflow.log_param("best_model_score",best_model_score)

                if isinstance(best_model_metrics, dict):
                     
                    for metric_name, metric_value in best_model_metrics.items():
                        if isinstance(metric_value, list):
                            mlflow.log_metric(metric_name, float(np.mean(metric_value)))
                        else:
                            mlflow.log_metric(metric_name, float(metric_value))

                
                mlflow.log_metric("best_model_score",best_model_score)


            if best_model_score >= base_model_score:

                self.utils.update_model_score(best_model_score)
                logging.info("update best_model_score")

                label_encoder=self.utils.load_obj(self.label_encoder)
                logging.info("sucsessfully loaded label_encoder")
                preprocessor_obj=self.utils.load_obj(self.preprocessor_obj)
                logging.info("successfully loaded preprocessor_obj")

                cost_model=CostModel(label_encoder,preprocessor_obj,best_model_object)
                logging.info(f"Created cost_model: {cost_model} and their type {type(cost_model)}")

                model_file_path=self.utils.save_preprocessor(cost_model,self.model_path)

                mlflow.sklearn.log_model(best_model_object,artifact_path="Best_model")
                mlflow.log_artifact(model_file_path)
                mlflow.log_param("model_file_path",model_file_path)

                logging.info(f"model save sucessfully {model_file_path}")
                return {
                    "model_path": model_file_path,
                    "model_name": best_model_name,
                    "score": best_model_score,
                    "metrics": best_model_metrics
                }

            else:
                logging.warning("No better model found than existing base model score.")
                mlflow.log_param("model_file_path",None)
                mlflow.log_param("message","No model surpassed thae base model score ")
                return {
                    "model_path": None,
                    "model_name": None,
                    "score": None,
                    "metrics": None,
                    "message": "No model surpassed the base model score."
                }
                #---------------------------mlflow_end------------------------------
        except Exception as e:
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
    
if __name__=="__main__":

    load_dotenv()
    os.getenv("ML_TRACKING_URL")
    experiment_name="Booking_prediction"
    dagshub.init(repo_owner='parth2104', repo_name='BookingPredictor', mlflow=True)


    
    Trainer=ModelTrainer(
        preprocessed_train_path= preprocessed_train_path,
        prepprocessor_test_path= preprocessed_test_path ,
        MODEL_PATH = model_out_put_path,
        preprocessor_obj_path= preprocessor_obj_path,
        label_encoder= label_encoder,
        experiment_name=experiment_name
    )
    Trainer.model_trainer_initiate()
    