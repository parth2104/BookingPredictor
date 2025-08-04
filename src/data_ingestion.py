import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from config.path_config import *
from src.logger.logger import logging
from src.Exception.exception import CustomException
from utils.common_functions import MainUtils
 


class DataIngestion:

    def __init__(self,config):

        self.config=config["data_ingestion"]
        self.bucket_name=self.config["bucket_name"]
        self.file_name=self.config["bucket_file_name"]
        self.train_test_ratio=self.config["train_ratio"]
        self.utils=MainUtils()

        os.makedirs(RAW_DATA,exist_ok=True)

        logging.info(f"data ingestion is started,bucket name {self.bucket_name} , filename is {self.file_name}")

    def download_csv_file_from_gcp(self):

        try:
             client=storage.Client()
             bucket=client.bucket(self.bucket_name)
             blob=bucket.blob(self.file_name)

             blob.download_to_filename(raw_file_path)
             logging.info(f"raw data downloaded successfully {raw_file_path} ")

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
    
    def split_data(self):

        try:
            logging.info("starting spliting data")

            data=pd.read_csv(raw_file_path)
            train_data,test_data=train_test_split(data,test_size=1 - self.train_test_ratio,random_state=43)

            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

            train_data.to_csv(train_file_path, index=False)
            test_data.to_csv(test_file_path, index=False)


            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e,sys)
        
    def run(self):

        try:
            self.download_csv_file_from_gcp()
            self.split_data()
        except CustomException as ce:
            logging.error(f"customexception : {str(ce)}")
        finally:
            logging.info("Data ingestion is successfully complited")

if __name__=="__main__":

    utils = MainUtils()

    data_ingestion=DataIngestion(utils.read_yaml(CONFIG_PATH))
    data_ingestion.run()
    logging.info("data ingestion is done.")

 
 
    
