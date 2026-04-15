import argparse
import datetime
import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

PROCESSED_A = "data/processed/prepared_A.csv"
PROCESSED_B = "data/processed/prepared_B.csv"
MODELS_DIR = "models/versions"
RAW_PATH = "data/raw/motor_data14-2018.csv"


def get_split(df):
    # Временное разбиение: 80% обучение, 20% тест
    split_idx = int(len(df) * 0.8)
    X = df.drop(columns=["CLAIM_PAID"])
    y = df["CLAIM_PAID"]
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], split_idx


def apply_inflation_coef(old_df, new_df):
    # Вычисляем коэффициент инфляции по перекрывающемуся 2014 году
    old_2014 = old_df[old_df["INSR_YEAR"] == 2014]
    new_2014 = new_df[new_df["INSR_YEAR"] == 2014]

    if len(old_2014) == 0 or len(new_2014) == 0 or "PREMIUM" not in old_df.columns:
        print("Не удалось вычислить коэффициент инфляции, пропускаем масштабирование")
        return old_df

    coef = new_2014["PREMIUM"].median() / old_2014["PREMIUM"].median()
    print(f"Коэффициент инфляции: {coef:.4f}")

    for col in ["CLAIM_PAID", "INSURED_VALUE", "PREMIUM"]:
        if col in old_df.columns:
            old_df[col] = old_df[col] * coef

    return old_df


def preprocess_retrain_data(old_path, new_df_raw):
    # Загружаем старый датасет и применяем те же преобразования что и в preparator.py
    old_df = pd.read_csv(old_path)
    old_df["CLAIM_PAID"] = old_df["CLAIM_PAID"].fillna(0)
    old_df["INSR_BEGIN"] = pd.to_datetime(old_df["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
    old_df["INSR_END"] = pd.to_datetime(old_df["INSR_END"], format="%d-%b-%y", errors="coerce")
    old_df["INSR_DURATION"] = (old_df["INSR_END"] - old_df["INSR_BEGIN"]).dt.days
    old_df["INSR_YEAR"] = old_df["INSR_BEGIN"].dt.year
    old_df = old_df.drop(columns=["INSR_BEGIN", "INSR_END", "OBJECT_ID"], errors="ignore")

    new_raw = new_df_raw.copy()
    new_raw["CLAIM_PAID"] = new_raw["CLAIM_PAID"].fillna(0)
    new_raw["INSR_BEGIN"] = pd.to_datetime(new_raw["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
    new_raw["INSR_END"] = pd.to_datetime(new_raw["INSR_END"], format="%d-%b-%y", errors="coerce")
    new_raw["INSR_DURATION"] = (new_raw["INSR_END"] - new_raw["INSR_BEGIN"]).dt.days
    new_raw["INSR_YEAR"] = new_raw["INSR_BEGIN"].dt.year
    new_raw = new_raw.drop(columns=["INSR_BEGIN", "INSR_END", "OBJECT_ID"], errors="ignore")

    old_df = apply_inflation_coef(old_df, new_raw)
    return old_df


def train_models(retrain_path=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Обучаем на двух вариантах препроцессинга
    variants = {"A": PROCESSED_A, "B": PROCESSED_B}

    for variant, path in variants.items():
        print(f"\n--- Вариант {variant} ---")
        df = pd.read_csv(path)

        if retrain_path is not None:
            # Режим дообучения: объединяем старые данные с текущими
            print(f"Режим дообучения: добавляем данные из {retrain_path}")
            new_df_raw = pd.read_csv(RAW_PATH)
            old_extra = preprocess_retrain_data(retrain_path, new_df_raw)

            with open("models/encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
            with open(f"models/scaler_{variant}.pkl", "rb") as f:
                scaler = pickle.load(f)

            # Кодируем категориальные признаки старого датасета
            for col, le in encoders.items():
                if col in old_extra.columns:
                    old_extra[col] = old_extra[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

            # Приводим колонки к тому же набору что и основной датасет
            old_extra = old_extra.reindex(columns=df.columns, fill_value=0)

            feature_cols = [c for c in df.columns if c != "CLAIM_PAID"]
            old_extra[feature_cols] = scaler.transform(old_extra[feature_cols])

            df = pd.concat([old_extra, df], ignore_index=True)
            print(f"Итоговый размер датасета: {len(df)}")

        X_train, X_test, y_train, y_test, split_idx = get_split(df)

        models = {
            "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
            "knn": KNeighborsRegressor(n_neighbors=5),
            "sgd": SGDRegressor(max_iter=1000, random_state=42),
        }

        # Если есть сохранённая SGD-модель — дообучаем её через partial_fit
        if retrain_path is not None:
            existing = sorted([f for f in os.listdir(MODELS_DIR) if f"sgd_{variant}" in f and f.endswith(".pkl")])
            if existing:
                sgd_model = joblib.load(os.path.join(MODELS_DIR, existing[-1]))
                print(f"Дообучаем SGD модель: {existing[-1]}")
                sgd_model.partial_fit(X_train, y_train)
                models["sgd"] = sgd_model

        for name, model in models.items():
            # SGD после partial_fit уже обучена, остальные обучаем с нуля
            if not (name == "sgd" and hasattr(model, "coef_")):
                model.fit(X_train, y_train)

            save_path = os.path.join(MODELS_DIR, f"{name}_{variant}_{timestamp}.pkl")
            joblib.dump(model, save_path)
            print(f"Сохранена модель: {save_path}")

        split_info = {"split_idx": split_idx, "timestamp": timestamp, "variant": variant}
        with open(os.path.join(MODELS_DIR, f"split_info_{variant}_{timestamp}.json"), "w") as f:
            json.dump(split_info, f)

    print("\nОбучение завершено")
    return timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", type=str, default=None, help="Путь ко второму датасету для дообучения")
    args = parser.parse_args()
    train_models(retrain_path=args.retrain)
