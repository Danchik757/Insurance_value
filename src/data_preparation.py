import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.storage import DatabaseStorage

LOGGER = None


def load_data():
    # Пробуем прочитать очищенные данные из БД
    try:
        storage = DatabaseStorage(CONFIG["storage"]["cleaned_table"])
        batches = list(storage.read())
        if batches:
            df = pd.concat(batches, ignore_index=True)
            LOGGER.info(f"Загружены данные из БД: {len(df)} строк")
            return df
        else:
            LOGGER.warning("В БД нет очищенных данных, переходим к CSV")
    except Exception as e:
        LOGGER.warning(f"Не удалось прочитать из БД: {e}")

    # Если БД недоступна — читаем из CSV
    cleaned_path = CONFIG["storage"]["cleaned_csv"]
    raw_path = CONFIG["dataset"]["paths"][-1]

    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        LOGGER.info(f"Загружены очищенные данные из CSV: {cleaned_path}")
    else:
        df = pd.read_csv(raw_path)
        LOGGER.info(f"Загружены сырые данные из CSV: {raw_path}")
    return df


def prepare_data():
    global LOGGER
    LOGGER = setup_logger(
        "DataPreparation",
        log_file=CONFIG["data_preparation"]["log_file"],
        level=CONFIG["logging"]["level"],
    )

    target = CONFIG["data_preparation"]["target_column"]
    date_fmt = CONFIG["data_preparation"]["date_format"]
    drop_cols = CONFIG["data_preparation"]["drop_columns"]
    processed_a = CONFIG["data_preparation"]["processed_a"]
    processed_b = CONFIG["data_preparation"]["processed_b"]
    models_dir = "models"

    df = load_data()

    # Пропуски в целевой переменной — нет выплаты, значит 0
    df[target] = df[target].fillna(0)

    # Парсим даты и создаём признак длительности страховки
    df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format=date_fmt, errors="coerce")
    df["INSR_END"] = pd.to_datetime(df["INSR_END"], format=date_fmt, errors="coerce")
    df["INSR_DURATION"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
    df["INSR_YEAR"] = df["INSR_BEGIN"].dt.year

    # Сортируем по времени для временного разбиения
    df = df.sort_values("INSR_BEGIN").reset_index(drop=True)
    df = df.drop(columns=drop_cols, errors="ignore")

    # Заполняем пропуски: числовые — медианой, категориальные — модой
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Кодируем категориальные переменные
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    feature_cols = [c for c in df.columns if c != target]

    # Вариант A — StandardScaler (нормализация по стандартному отклонению)
    df_a = df.copy()
    scaler_a = StandardScaler()
    df_a[feature_cols] = scaler_a.fit_transform(df_a[feature_cols])
    with open(os.path.join(models_dir, "scaler_A.pkl"), "wb") as f:
        pickle.dump(scaler_a, f)

    # Вариант B — MinMaxScaler (масштабирование в диапазон [0, 1])
    df_b = df.copy()
    scaler_b = MinMaxScaler()
    df_b[feature_cols] = scaler_b.fit_transform(df_b[feature_cols])
    with open(os.path.join(models_dir, "scaler_B.pkl"), "wb") as f:
        pickle.dump(scaler_b, f)

    os.makedirs(os.path.dirname(processed_a), exist_ok=True)
    df_a.to_csv(processed_a, index=False)
    df_b.to_csv(processed_b, index=False)

    LOGGER.info(f"Вариант A (StandardScaler) сохранён: {processed_a}")
    LOGGER.info(f"Вариант B (MinMaxScaler) сохранён: {processed_b}")

    return df_a, df_b


if __name__ == "__main__":
    prepare_data()
