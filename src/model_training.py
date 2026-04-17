import argparse
import datetime
import json
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger

LOGGER = None


def get_split(df):
    # Временное разбиение: 80% обучение, 20% тест
    split_ratio = CONFIG["model_training"]["train_split"]
    split_idx = int(len(df) * split_ratio)
    X = df.drop(columns=[CONFIG["data_preparation"]["target_column"]])
    y = df[CONFIG["data_preparation"]["target_column"]]
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], split_idx


def apply_inflation_coef(old_df, new_df):
    year = CONFIG["model_training"]["inflation_year"]
    old_year = old_df[old_df["INSR_YEAR"] == year]
    new_year = new_df[new_df["INSR_YEAR"] == year]

    if len(old_year) == 0 or len(new_year) == 0 or "PREMIUM" not in old_df.columns:
        LOGGER.warning("Не удалось вычислить коэффициент инфляции, пропускаем масштабирование")
        return old_df

    coef = new_year["PREMIUM"].median() / old_year["PREMIUM"].median()
    LOGGER.info(f"Коэффициент инфляции: {coef:.4f}")

    for col in CONFIG["model_training"]["inflation_columns"]:
        if col in old_df.columns:
            old_df[col] = old_df[col] * coef

    return old_df


def preprocess_retrain_data(old_path, new_df_raw):
    date_fmt = CONFIG["data_preparation"]["date_format"]
    drop_cols = CONFIG["data_preparation"]["drop_columns"]
    target = CONFIG["data_preparation"]["target_column"]

    old_df = pd.read_csv(old_path)
    old_df[target] = old_df[target].fillna(0)
    old_df["INSR_BEGIN"] = pd.to_datetime(old_df["INSR_BEGIN"], format=date_fmt, errors="coerce")
    old_df["INSR_END"] = pd.to_datetime(old_df["INSR_END"], format=date_fmt, errors="coerce")
    old_df["INSR_YEAR"] = old_df["INSR_BEGIN"].dt.year
    old_df["policy_duration_days"] = (old_df["INSR_END"] - old_df["INSR_BEGIN"]).dt.days.fillna(365)
    old_df["premium_per_seat"] = old_df["PREMIUM"] / (old_df["SEATS_NUM"] + 1e-6)
    old_df["insured_value_per_ton"] = old_df["INSURED_VALUE"] / (old_df["CCM_TON"] + 1e-6)
    old_df["claim_ratio"] = (old_df[target] / (old_df["PREMIUM"] + 1e-6)).clip(0, 10)
    old_df["is_claim"] = (old_df[target] > 0).astype(int)
    old_df["premium_log"] = np.log(old_df["PREMIUM"] + 1e-6)
    old_df = old_df.drop(columns=drop_cols, errors="ignore")

    new_raw = new_df_raw.copy()
    new_raw[target] = new_raw[target].fillna(0)
    new_raw["INSR_BEGIN"] = pd.to_datetime(new_raw["INSR_BEGIN"], format=date_fmt, errors="coerce")
    new_raw["INSR_END"] = pd.to_datetime(new_raw["INSR_END"], format=date_fmt, errors="coerce")
    new_raw["INSR_YEAR"] = new_raw["INSR_BEGIN"].dt.year
    new_raw["policy_duration_days"] = (new_raw["INSR_END"] - new_raw["INSR_BEGIN"]).dt.days.fillna(365)
    new_raw["premium_per_seat"] = new_raw["PREMIUM"] / (new_raw["SEATS_NUM"] + 1e-6)
    new_raw["insured_value_per_ton"] = new_raw["INSURED_VALUE"] / (new_raw["CCM_TON"] + 1e-6)
    new_raw["claim_ratio"] = (new_raw[target] / (new_raw["PREMIUM"] + 1e-6)).clip(0, 10)
    new_raw["is_claim"] = (new_raw[target] > 0).astype(int)
    new_raw["premium_log"] = np.log(new_raw["PREMIUM"] + 1e-6)
    new_raw = new_raw.drop(columns=drop_cols, errors="ignore")

    old_df = apply_inflation_coef(old_df, new_raw)
    return old_df


def train_models(retrain_path=None):
    global LOGGER
    LOGGER = setup_logger(
        "ModelTraining",
        log_file=CONFIG["model_training"]["log_file"],
        level=CONFIG["logging"]["level"],
    )

    models_dir = CONFIG["model_training"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    processed_a = CONFIG["data_preparation"]["processed_a"]
    processed_b = CONFIG["data_preparation"]["processed_b"]

    # Обучаем на двух вариантах препроцессинга
    variants = {"A": processed_a, "B": processed_b}

    for variant, path in variants.items():
        LOGGER.info(f"--- Вариант {variant} ---")
        df = pd.read_csv(path)

        if retrain_path is not None:
            # Режим дообучения: объединяем старые данные с текущими
            LOGGER.info(f"Режим дообучения: добавляем данные из {retrain_path}")
            new_df_raw = pd.read_csv(CONFIG["dataset"]["paths"][-1])
            old_extra = preprocess_retrain_data(retrain_path, new_df_raw)

            with open(CONFIG["model_training"]["encoders_path"], "rb") as f:
                encoders = pickle.load(f)
            with open(f"models/scaler_{variant}.pkl", "rb") as f:
                scaler = pickle.load(f)

            # Кодируем категориальные признаки старого датасета
            for col, le in encoders.items():
                if col in old_extra.columns:
                    old_extra[col] = old_extra[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

            old_extra = old_extra.reindex(columns=df.columns, fill_value=0)

            target = CONFIG["data_preparation"]["target_column"]
            feature_cols = [c for c in df.columns if c != target]
            old_extra[feature_cols] = scaler.transform(old_extra[feature_cols])

            df = pd.concat([old_extra, df], ignore_index=True)
            LOGGER.info(f"Итоговый размер датасета: {len(df)}")

        X_train, X_test, y_train, y_test, split_idx = get_split(df)

        dt_cfg = CONFIG["model_training"]["decision_tree"]
        knn_cfg = CONFIG["model_training"]["knn"]
        sgd_cfg = CONFIG["model_training"]["sgd"]

        models = {
            "decision_tree": DecisionTreeRegressor(
                max_depth=dt_cfg["max_depth"],
                random_state=dt_cfg["random_state"],
            ),
            "knn": KNeighborsRegressor(n_neighbors=knn_cfg["n_neighbors"]),
            "sgd": SGDRegressor(
                max_iter=sgd_cfg["max_iter"],
                random_state=sgd_cfg["random_state"],
            ),
        }

        # Если есть сохранённая SGD-модель — дообучаем её через partial_fit
        if retrain_path is not None:
            existing = sorted([f for f in os.listdir(models_dir) if f"sgd_{variant}" in f and f.endswith(".pkl")])
            if existing:
                sgd_model = joblib.load(os.path.join(models_dir, existing[-1]))
                LOGGER.info(f"Дообучаем SGD модель: {existing[-1]}")
                sgd_model.partial_fit(X_train, y_train)
                models["sgd"] = sgd_model

        for name, model in models.items():
            if not (name == "sgd" and hasattr(model, "coef_")):
                model.fit(X_train, y_train)

            save_path = os.path.join(models_dir, f"{name}_{variant}_{timestamp}.pkl")
            joblib.dump(model, save_path)
            LOGGER.info(f"Сохранена модель: {save_path}")

        split_info = {"split_idx": split_idx, "timestamp": timestamp, "variant": variant}
        with open(os.path.join(models_dir, f"split_info_{variant}_{timestamp}.json"), "w") as f:
            json.dump(split_info, f)

    LOGGER.info("Обучение завершено")
    return timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", type=str, default=None, help="Путь ко второму датасету для дообучения")
    args = parser.parse_args()
    train_models(retrain_path=args.retrain)
