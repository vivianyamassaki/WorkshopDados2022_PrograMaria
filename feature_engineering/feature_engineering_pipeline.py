import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from feature_engineering.missing_imputer import MissingValuesImputer
from feature_engineering.numerical_scaler import NumericalFeaturesScaler
from feature_engineering.ohe_transformer import OneHotEncode

logger = logging.getLogger(__name__)


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_features):
        self.numerical_features = numerical_features

    def get_pipeline(self):
        logger.info('Fazendo a feature engineering...')
        return Pipeline(
            [
                ("ohe", OneHotEncode()),
                ("missing_imputer", MissingValuesImputer()),
                ("numerical_scaler", NumericalFeaturesScaler(numerical_features=self.numerical_features)),
            ]
        )
