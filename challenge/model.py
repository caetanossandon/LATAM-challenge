import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Union, List
import os

class DelayModel:
    MODEL_PATH = 'data/models/trained_model.pkl'  # Path to save the trained model

    def __init__(self):
        self._model = None
        self.top_features = [
            "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM", "MES_12", 
            "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
        ]

        if os.path.exists(self.MODEL_PATH):
            self.load_model()

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        self._generate_features(data)
        features = self._one_hot_encode_features(data)

        if target_column:
            return features, data[[target_column]]
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)
        value_counts = y_train['delay'].value_counts()
        n_y0 = value_counts.get(0, 0)
        n_y1 = value_counts.get(1, 0)
        scale = n_y0/n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(x_train, y_train)
        self.save_model()

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("The model hasn't been trained yet. Call the 'fit' method before 'predict'.")
        predictions = self._model.predict(features)
        return predictions.tolist()

    def _generate_features(self, data: pd.DataFrame) -> None:
        data['period_day'] = self._get_period_day(data['Fecha-I'])
        data['high_season'] = self._is_high_season(data['Fecha-I'])
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        data['delay'] = np.where(data['min_diff'] > 15, 1, 0)

    def _one_hot_encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded_data = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)
        
        # Ensure all columns in self.top_features are present
        for col in self.top_features:
            if col not in encoded_data.columns:
                encoded_data[col] = 0

        return encoded_data[self.top_features]


    def save_model(self):
        """Save the trained model to a file."""
        if self._model is not None:
            joblib.dump(self._model, self.MODEL_PATH)
        else:
            raise ValueError("Model is not trained yet!")

    def load_model(self):
        """Load the trained model from a file."""
        self._model = joblib.load(self.MODEL_PATH)

    @staticmethod
    def _get_period_day(dates):
        dates = pd.to_datetime(dates, errors='coerce')
        hours = dates.dt.hour
        conditions = [
            (hours >= 5) & (hours < 12),
            (hours >= 12) & (hours < 19)
        ]
        choices = ['mañana', 'tarde']
        return pd.Series(np.select(conditions, choices, default='noche'))

    @staticmethod
    def _is_high_season(dates):
        dates = pd.to_datetime(dates, errors='coerce')
        years = dates.dt.year
        date_ranges = [
            ('12-15', '12-31'),
            ('01-01', '03-03'),
            ('07-15', '07-31'),
            ('09-11', '09-30')
        ]
        masks = []
        for start, end in date_ranges:
            range_start = pd.to_datetime(years.astype(str) + '-' + start, errors='coerce')
            range_end = pd.to_datetime(years.astype(str) + '-' + end, errors='coerce')
            masks.append((dates >= range_start) & (dates <= range_end))
        final_mask = np.logical_or.reduce(masks)
        return final_mask.astype(int)
