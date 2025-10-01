# common/libraries/python/ml/preprocessing.py
"""
Preprocessing utilities: cleaning, scaling, encoding.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    return df.drop(columns, axis=1).join(encoded_df)
