import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from src.Exception.exception import CustomException
from src.logger.logger import logging
from src.cost_model import CostModel
from config.path_config import model_out_put_path, CONFIG_PATH, PREPROCESSED_DIR
from utils.common_functions import MainUtils

app = Flask(__name__)

utils = MainUtils()
config = utils.read_yaml(CONFIG_PATH)
logging.info(f"Loaded config: {config}")

try:
    logging.info("Loading artifacts...")
    cost_model = joblib.load(model_out_put_path)
    selected_features = cost_model.feature_order
    if "booking_status" in selected_features:
        raise ValueError("Target column 'booking_status' found in selected_features")
    logging.info(f"Loaded selected_features from CostModel: {selected_features}")

    # Load feature_columns from columns.yaml
    columns_yaml = utils.read_yaml(os.path.join(PREPROCESSED_DIR, "columns.yaml"))
    feature_columns = columns_yaml.get("feature_columns", [])
    if "booking_status" in feature_columns:
        raise ValueError("Target column 'booking_status' found in feature_columns")
    logging.info(f"Loaded feature_columns from columns.yaml: {feature_columns}")
except Exception as e:
    logging.error(f"Failed to load artifacts: {e}")
    raise CustomException(str(e), sys)

# Derive categorical_cols from feature_columns and config
categorical_cols = [
    col for col in config["data_preprocessing"]["categorical"]
    if col in feature_columns
]
logging.info(f"Categorical columns (filtered by feature_columns): {categorical_cols}")

categorical_mappings = {}
try:
    for i, col in enumerate(config["data_preprocessing"]["categorical"]):
        if col not in feature_columns:
            logging.warning(f"Categorical column '{col}' not in feature_columns")
            continue
        categories = cost_model.encoder.categories_[i]
        categorical_mappings[col] = {str(idx): cat for idx, cat in enumerate(categories)}
    logging.info(f"Categorical mappings: {categorical_mappings}")
except Exception as e:
    logging.error(f"Failed to create categorical mappings: {e}")
    raise CustomException(str(e), sys)

def safe_float(val, col_name):
    try:
        f = float(val)
        if not np.isfinite(f):
            raise ValueError
        return f
    except:
        raise ValueError(f"Invalid numeric input for '{col_name}': {val}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            input_data = {}
            for col in feature_columns:  # Use feature_columns order
                if col == "booking_status":
                    raise ValueError("Target column 'booking_status' should not be included in input data")
                form_value = request.form.get(col)
                if form_value is None:
                    raise ValueError(f"Missing form input for '{col}'")
                if col in categorical_cols:
                    value = categorical_mappings.get(col, {}).get(form_value)
                    if value is None:
                        raise ValueError(f"Invalid category for '{col}': {form_value}")
                    input_data[col] = value
                else:
                    input_data[col] = safe_float(form_value, col)

            # Create DataFrame with exact column order from columns.yaml
            features = pd.DataFrame([input_data], columns=feature_columns)
            logging.info(f"Input features: {features.to_dict(orient='records')}")

            # Validate column order
            input_columns = list(features.columns)
            if input_columns != feature_columns:
                raise ValueError(
                    f"Input columns {input_columns} do not match expected {feature_columns}"
                )

            prediction = cost_model.predict(features)[0]
            logging.info(f"Prediction: {prediction}")

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        selected_features=feature_columns,  # Pass feature_columns to form
        categorical_cols=categorical_cols,
        categorical_mappings=categorical_mappings
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)