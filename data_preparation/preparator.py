import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

RAW_PATH = "data/raw/motor_data14-2018.csv"
CLEANED_PATH = "data/cleaned/cleaned_data.csv"
PROCESSED_A = "data/processed/prepared_A.csv"
PROCESSED_B = "data/processed/prepared_B.csv"
MODELS_DIR = "models"


def load_data():
    # Загружаем очищенные данные если они есть, иначе сырые
    if os.path.exists(CLEANED_PATH):
        df = pd.read_csv(CLEANED_PATH)
        print(f"Загружены очищенные данные: {CLEANED_PATH}")
    else:
        df = pd.read_csv(RAW_PATH)
        print(f"Загружены сырые данные: {RAW_PATH}")
    return df


def prepare_data():
    df = load_data()

    # Пропуски в целевой переменной — нет выплаты, значит 0
    df["CLAIM_PAID"] = df["CLAIM_PAID"].fillna(0)

    # Парсим даты и создаём признак длительности страховки
    df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
    df["INSR_END"] = pd.to_datetime(df["INSR_END"], format="%d-%b-%y", errors="coerce")
    df["INSR_DURATION"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
    df["INSR_YEAR"] = df["INSR_BEGIN"].dt.year

    # Сортируем по времени для временного разбиения
    df = df.sort_values("INSR_BEGIN").reset_index(drop=True)
    df = df.drop(columns=["INSR_BEGIN", "INSR_END", "OBJECT_ID"], errors="ignore")

    # Заполняем пропуски: числовые — медианой, категориальные — модой
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != "CLAIM_PAID":
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

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    feature_cols = [c for c in df.columns if c != "CLAIM_PAID"]

    # Вариант A — StandardScaler (нормализация по стандартному отклонению)
    df_a = df.copy()
    scaler_a = StandardScaler()
    df_a[feature_cols] = scaler_a.fit_transform(df_a[feature_cols])
    with open(os.path.join(MODELS_DIR, "scaler_A.pkl"), "wb") as f:
        pickle.dump(scaler_a, f)

    # Вариант B — MinMaxScaler (масштабирование в диапазон [0, 1])
    df_b = df.copy()
    scaler_b = MinMaxScaler()
    df_b[feature_cols] = scaler_b.fit_transform(df_b[feature_cols])
    with open(os.path.join(MODELS_DIR, "scaler_B.pkl"), "wb") as f:
        pickle.dump(scaler_b, f)

    os.makedirs("data/processed", exist_ok=True)
    df_a.to_csv(PROCESSED_A, index=False)
    df_b.to_csv(PROCESSED_B, index=False)

    print(f"Вариант A (StandardScaler) сохранён: {PROCESSED_A}")
    print(f"Вариант B (MinMaxScaler) сохранён: {PROCESSED_B}")

    return df_a, df_b


if __name__ == "__main__":
    prepare_data()
