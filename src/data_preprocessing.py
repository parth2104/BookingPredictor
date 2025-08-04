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
from sklearn.preprocessing import LabelEncoder,FunctionTransformer,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

class DataTransformation( ):

    def __init__(self,train_file_path,test_file_path,PREPROCESSED_DIR,CONFIG_PATH,label_encoder):

        self.train_df=train_file_path
        self.test_df=test_file_path
        self.preprocess_dir=PREPROCESSED_DIR
        self.utils=MainUtils()
        self.config=self.utils.read_yaml(CONFIG_PATH)
        self.label_encoder=label_encoder
        

        if not os.path.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

    def data_cleaning(self,data):
        try:
            data.drop(columns=["Booking_ID"],inplace=True)
            logging.info(f"drop data column Booking_ID {data.head(1)}")

            data.drop_duplicates(inplace=True)
            logging.info("drop duplicated values from data ")
            return data
            
        except Exception as e:
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)

    def apply_label_encoding(self,train_df, test_df, cat_col, save_path):
        try:

            logging.info("Applying Label Encoding to categorical columns")
            label_encoders = {}
            

            for col in cat_col:
                encoder = LabelEncoder()
                train_df[col] = encoder.fit_transform(train_df[col].astype(str))
                test_df[col] = encoder.transform(test_df[col].astype(str))
                label_encoders[col] = encoder

        
            joblib.dump(label_encoders, save_path)
            logging.info(f"Label encoders saved to {save_path}")

            return train_df, test_df
        except Exception as e:
            logging.error(CustomException(str(e),sys))
        



    def data_imbalanced(self,data):
        try:
            x=data.drop(columns=["booking_status"])
            y=data["booking_status"]
            logging.info(f"separating x and y: x shape = {x.shape}, y shape = {y.shape}")

            smote = SMOTE(random_state=43)
            x_resampled, y_resampled = smote.fit_resample(x, y)

            balanced_data = pd.DataFrame(x_resampled, columns=x.columns if hasattr(x, 'columns') else [f"feature_{i}" for i in range(x.shape[1])])
            balanced_data["booking_status"] = y_resampled

            return balanced_data

        except Exception as e:
            logging.error(CustomException(str(e), sys))
            raise CustomException(str(e), sys)
        

    def feature_selection(self,data):
         try:
            x=data.drop(columns=["booking_status"])
            y=data["booking_status"]

            model=RandomForestClassifier(random_state=42)
            model=model.fit(x,y)

            important_feature=model.feature_importances_
            important_features=pd.DataFrame({
                "feature":x.columns,
                "importance":important_feature
            })
            logging.info(f"feture importance {important_features}")

            important_features = important_features.sort_values(by="importance", ascending=False)

            top_n_features = self.config["data_preprocessing"]["number_of_feature"]
            selected_features = important_features["feature"].head(top_n_features).tolist()

            logging.info(f"Top {top_n_features} features: {selected_features}")

            return selected_features
         except Exception as e :
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)

    def preprrocessing(self,num_columns):
        try:
            


            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("FunctionTransformer",FunctionTransformer(np.log1p,validate=True)),
                       ("StandardScaler",StandardScaler())]
            )

            preprocessor=ColumnTransformer(
                transformers=[
                    ("numerical",num_pipeline,num_columns)
                ]
            )
            return preprocessor


        except Exception as e:
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
    
    def main(self):

        try:

            train_df=self.utils.load_data(train_file_path)
            test_df=self.utils.load_data(test_file_path)
            logging.info("load data succsessfully ")
            logging.info("--------------------------------------------------------")

            train_df=self.data_cleaning(train_df)
            test_df=self.data_cleaning(test_df)
            logging.info("data cleaning successfully.")
            logging.info("-------------------------------------------------")

            cat_col=self.config["data_preprocessing"]["categorical"]
            logging.info("Applying Label Encoding")
            
            save_path= label_encoder
            logging.info(f"Label Encoding .pkl will be saved at : {save_path}")

            train_df,test_df=self.apply_label_encoding(train_df,test_df,cat_col,save_path)

            logging.info("lable encoder complited")
            logging.info("-----------------------------------------------------------")

            train_resampled= self.data_imbalanced(train_df)
            logging.info("data imbalanced complited")
            logging.info("---------------------------------------------")

            selected_features = self.feature_selection(train_resampled)

            train_selected = train_resampled[selected_features + ["booking_status"]]
            test_selected = test_df[selected_features + ["booking_status"]]


            logging.info(f"tetst selected {test_selected.shape}")
            logging.info(f"{test_selected}")
            selected_col=list(train_selected.columns)
            selected_col = [col for col in selected_col if col != "booking_status"]


            
            logging.info(f"seleccted columns:{selected_col}")
            

            train_df=train_selected
            logging.info(f" after feature selection taindf:{train_df.head(1)}")
            test_df=test_selected
            logging.info(f"after feature selecctio test_df:{test_df.head(1)}")

            logging.info("feature selecction is complited")
            logging.info("-----------------------------------------------------------------")
            
            x_train=train_df.drop(columns=["booking_status"])
            y_train=train_df["booking_status"]
            x_test=test_df.drop(columns=["booking_status"])
            y_test=test_df["booking_status"]
            logging.info(f"x_train{x_train.head(1)}")
            logging.info(f"x_test{x_test.head(1)}")
            logging.info(f"y_train{y_train.head(1)}")
            logging.info(f"y_test{y_test.head(1)}")


            preprocessor_obj=self.preprrocessing(selected_col)
            logging.info(f"preprocessing: {preprocessor_obj}")
            x_train_transformed=preprocessor_obj.fit_transform(x_train)
            logging.info(f"x_train_transformed: {x_train_transformed}")

            x_test_transformed=preprocessor_obj.transform(x_test)
            logging.info(f"x_test_transformed:{x_test_transformed}")

            y_train_transformed=np.array(y_train)
            y_test_transformed=np.array(y_test)

            train_arr= np.c_[(x_train_transformed,y_train_transformed)]
            test_arr=np.c_[(x_test_transformed,y_test_transformed)]

            self.utils.save_preprocessor(preprocessor_obj,preprocessor_obj_path)
            self.utils.save_npz(preprocessed_train_path,arr=train_arr)
            self.utils.save_npz(preprocessed_test_path,arr=test_arr)
            return preprocessor_obj,train_arr,test_arr

        except Exception as e:
            logging.error(CustomException(str(e),sys))
            raise CustomException(str(e),sys)
            
if __name__=="__main__":
    data_transformation=DataTransformation(
        train_file_path,test_file_path,
        PREPROCESSED_DIR,CONFIG_PATH,
        label_encoder
    )
    data_transformation.main()


        
 