import os
import sys
from src.Exception.exception import CustomException
from src.logger.logger import logging
from config.path_config import *
from google.cloud import storage
from utils.common_functions import MainUtils


class GcpUploder:
    def __init__(self,config):
        self.config=config["data_ingestion"]
        self.bucket_name=self.config["bucket_name"]
        self.folder_name=self.config["folder_name"]
        

    def upload_to_gcp(self, model_out_put_path):
        try:
            if not os.path.exists(model_out_put_path):
                raise FileNotFoundError(f"File path does not exist: {model_out_put_path}")

            logging.info(f"Uploading {model_out_put_path} to gs://{self.bucket_name}/{self.folder_name}")

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob_name = os.path.join(self.folder_name, os.path.basename(model_out_put_path)).replace("\\", "/")
            blob = bucket.blob(blob_name)

            blob.upload_from_filename(model_out_put_path, timeout=600)

            gcp_uri = f"gs://{self.bucket_name}/{blob_name}"
            logging.info(f"GCP_URI uploaded successfully: {gcp_uri}")

            
            logging.info(f"Listing contents of bucket '{self.bucket_name}':")
            blobs = bucket.list_blobs(prefix=self.folder_name)
            for b in blobs:
                logging.info(f" - {b.name}")

            return gcp_uri

        except Exception as e:
          raise CustomException(str(e), sys)

if __name__=="__main__":
   
   utils=MainUtils()
   config=utils.read_yaml(CONFIG_PATH)

   gcp_uploder=GcpUploder(config)

   local_model_path=model_out_put_path
   gcp_uri=gcp_uploder.upload_to_gcp(local_model_path)
 


          
