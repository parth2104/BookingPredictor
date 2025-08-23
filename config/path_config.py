import os

######################################_Data_Ingestion_########################################

RAW_DATA="artifacts//raw"
raw_file_path=os.path.join(RAW_DATA,"raw.csv")
test_file_path=os.path.join(RAW_DATA,"test.csv")
train_file_path=os.path.join(RAW_DATA,"train.csv")

###########################_read_yaml_file_############################################################

CONFIG_PATH="config//config.yaml"
MODEL_PATH="config/model.yaml"


#############################_Data_preprocessing_#############################################

PREPROCESSED_DIR="artifacts//preprocessed"
path_ordinal_encoder=os.path.join(PREPROCESSED_DIR,"encoder.pkl")
preprocessor_obj_path=os.path.join(PREPROCESSED_DIR,"preprocessor.pkl")
preprocessed_train_path=os.path.join(PREPROCESSED_DIR,"preprocessed_train.npz")
preprocessed_test_path=os.path.join(PREPROCESSED_DIR,"preprocessed_test.npz")
feature_selected_path=os.path.join(PREPROCESSED_DIR,"feature_names.yaml")

############################# Mdel_Traiinig ##################################################

MODEL="artifacts//Model"
model_out_put_path=os.path.join(MODEL,"model.pkl")

 