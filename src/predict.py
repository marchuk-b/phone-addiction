import joblib
import pandas as pd

FEATURE_NAMES = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

MODEL_DIR = "./artifacts/model_2026-04-15_19-57-28"


def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    pca = joblib.load(f"{MODEL_DIR}/pca.pkl")
    return model, scaler, pca


def parse_user_input(user_input: str) -> pd.DataFrame:
    values = user_input.strip().split()

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(f"You must enter exactly {len(FEATURE_NAMES)} values.")

    try:
        values = [float(v) for v in values]
    except ValueError as e:
        raise ValueError("All input values must be numeric.") from e

    return pd.DataFrame([values], columns=FEATURE_NAMES)


def predict():
    model, scaler, pca = load_artifacts()

    print("Enter feature values in this order:")
    print(", ".join(FEATURE_NAMES))

    user_input = input("Input values separated by spaces: ")

    try:
        sample = parse_user_input(user_input)

        sample_scaled = scaler.transform(sample)
        sample_pca = pca.transform(sample_scaled)
        prediction = model.predict(sample_pca)[0]

        if prediction == 1:
            print("Prediction: Addicted")
        else:
            print("Prediction: Not addicted")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    predict()