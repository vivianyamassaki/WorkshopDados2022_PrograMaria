"""
Script que contém o pipeline da feature engineering. Ele chama scripts secundários que fazem as transformações necessárias
dos dados e na ordem em que constam no pipeline
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from feature_engineering.missing_imputer import MissingValuesImputer
from feature_engineering.numerical_scaler import NumericalFeaturesScaler
from feature_engineering.ohe_transformer import OneHotEncode


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        self.pipeline = Pipeline(
            [
                ("ohe", OneHotEncode()),
                ("missing_imputer", MissingValuesImputer()),
                ("numerical_scaler", NumericalFeaturesScaler(numerical_features=self.numerical_features)),
            ]
        )

    def transform(self, data):
        return self.pipeline.transform(data)
