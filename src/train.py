import pandas as pd
import joblib
import os
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from prepare_data import prepare_data
from preprocessing import get_processed_data
from evaluate import evaluate

raw_data = pd.read_csv("../data/raw/data.csv")

def build_model():
    return RandomForestClassifier(random_state=42)

def train_model():
    data = prepare_data(raw_data)
    X_train, X_test, y_train, y_test, scaler = get_processed_data(data)

    best_score = -1.0
    best_model = None
    best_pca = None
    best_metrics = None
    best_n = None

    for n in range(1, X_train.shape[1] + 1):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        model = build_model()
        model.fit(X_train_pca, y_train)

        y_pred = model.predict(X_test_pca)

        m = evaluate(y_test, y_pred)
        print(f"PCA components: {n}, metrics: {m}")

        if m["accuracy"] > best_score:
            best_score = m["accuracy"]
            best_model = model
            best_pca = pca
            best_metrics = m
            best_n = n

    return best_model, best_pca, scaler, best_metrics, best_n


def save_model():
    best_model, best_pca, scaler, best_metrics, best_n = train_model()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = f"../artifacts/model_{timestamp}"

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(best_pca, f"{model_dir}/pca.pkl")
    joblib.dump(best_metrics, f"{model_dir}/metrics.pkl")

    print(f"Saved model to: {model_dir}")
    print(f"Best PCA components: {best_n}")
    print(f"Best metrics: {best_metrics}")

    return model_dir, best_metrics, best_n

if __name__ == "__main__":
    save_model()