from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataTransformation
from src.model_training import ModelTrainer
from src.logger.logger import logging
from utils.common_functions import MainUtils
from config.path_config import *

if __name__=="__main__":

    #Data ingestion 

    utils = MainUtils()

    data_ingestion=DataIngestion(utils.read_yaml(CONFIG_PATH))
    data_ingestion.run()
    logging.info("data ingestion is done.")

    #Data Preprocessing 

    data_transformation=DataTransformation(
        train_file_path,test_file_path,
        PREPROCESSED_DIR,CONFIG_PATH,
        label_encoder
    )
    data_transformation.main()

    #Model Training 

    Trainer=ModelTrainer(
        preprocessed_train_path= preprocessed_train_path,
        prepprocessor_test_path= preprocessed_test_path ,
        MODEL_PATH = model_out_put_path,
        preprocessor_obj_path= preprocessor_obj_path,
        label_encoder= label_encoder
    )
    Trainer.model_trainer_initiate()

   

