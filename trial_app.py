from src.Exception.exception import CustomException
from src.logger.logger import logging
import os
import sys 
try:
    logging.info("Hello")
    1 / 0  # Deliberate ZeroDivisionError
    
except Exception as e:
     logging.info(CustomException(e,sys))
     raise CustomException(e,sys)