from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_processed_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    X = data.drop(columns=['addicted_label'])
    y = data['addicted_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler