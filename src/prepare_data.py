import pandas as pd
import numpy as nd

def prepare_data(raw_data: pd.DataFrame) -> nd.ndarray:
    raw_data = raw_data.drop(columns=['transaction_id', 'user_id'])
    numeric_data = raw_data.select_dtypes(include=['float64','int64'])

    target = 'addicted_label'
    features = numeric_data.drop(columns=[target]).columns.tolist()

    data = raw_data[features + [target]]
    data.to_csv("../data/prepared/prepared_data.csv", index=False)

    return data