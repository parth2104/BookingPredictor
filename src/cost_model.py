import sys
import pandas as pd
from src.Exception.exception import CustomException
from src.logger.logger import logging

class CostModel:
    def __init__(self, encoder, preprocessor, model, feature_order):
        """
        encoder: Fitted OrdinalEncoder
        preprocessor: Fitted preprocessor (ColumnTransformer)
        model: Trained ML model
        feature_order: List of features in correct order (exclude target)
        """
        if "booking_status" in feature_order:
            raise ValueError("Target column 'booking_status' should not be included in feature_order")
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.model = model
        self.feature_order = feature_order
        logging.info(f"CostModel initialized with feature_order: {feature_order}")

    def predict(self, X: pd.DataFrame):
        try:
            input_columns = list(X.columns)
            logging.info(f"Input columns for prediction: {input_columns}")
            logging.info(f"Expected feature_order: {self.feature_order}")
            logging.info(f"Input data: {X.head(1).to_dict()}")

            if "booking_status" in input_columns:
                raise ValueError("Target column 'booking_status' found in input data")
            missing_features = [col for col in self.feature_order if col not in input_columns]
            extra_features = [col for col in input_columns if col not in self.feature_order]
            if missing_features or extra_features:
                raise ValueError(
                    f"Feature mismatch: Missing features: {missing_features}, "
                    f"Extra features: {extra_features}"
                )

            cat_cols = [col for col in X.columns if X[col].dtype == 'object']
            if cat_cols:
                X[cat_cols] = self.encoder.transform(X[cat_cols].astype(str))

            X = X[self.feature_order]

            X_transformed = self.preprocessor.transform(X)

            prediction = self.model.predict(X_transformed)
            logging.info(f"Prediction result: {prediction}")
            return prediction

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise CustomException(str(e), sys)