from src.utils import load_pickle
import settings as st

from feature_engineering.feature_engineering_pipeline import FeatureEngineering

class Predict():

    def __init__(self):
        self.model_path = st.model_path
        self.model = load_pickle(self.model_path)

    def predict(self, data):
        processed_data = FeatureEngineering(st.numerical_features).transform(data)
        result= self.model.predict_proba(processed_data)[0]
        status = 'morreu'
        if result[0] > 0.5:
            status = 'sobreviveu'
        return status, result[0]
