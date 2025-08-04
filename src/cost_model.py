import sys
import pandas as pd
from src.Exception.exception import CustomException
from src.logger.logger import logging
import warnings

warnings.filterwarnings("ignore")

class CostModel:
    def __init__(self, label_encoder, preprocessor, model):
        self.preprocessor_obj = preprocessor
        self.label_encoder = label_encoder
        self.model = model

    def predication(self, X):
        try:
            logging.info(f"Received input data: {X.head(1)}")

            x_transformed = X.drop_duplicates()
            logging.info("Dropped duplicate rows.")

            x_transformed = self.label_encoder.transform(x_transformed)
            logging.info("Applied label encoder.")

            x_transformed = self.preprocessor_obj.transform(x_transformed)
            logging.info("Applied preprocessor.")

            logging.info("All transformations complete. Making predictions...")
            return self.model.predict(x_transformed)

        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)

     