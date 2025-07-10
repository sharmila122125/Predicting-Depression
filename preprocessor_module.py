# preprocessor_module.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DepressionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_transformer = None
        self.feature_names = None
        self.columns_to_drop = ['id', 'Name', 'City']

    def fit(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        self.num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.column_transformer = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])

        self.column_transformer.fit(X)

        num_features = self.num_cols
        cat_features = list(self.column_transformer.named_transformers_['cat']['encoder'].get_feature_names_out(self.cat_cols))
        self.feature_names = num_features + cat_features

        return self

    def transform(self, X):
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        return pd.DataFrame(self.column_transformer.transform(X), columns=self.feature_names)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
