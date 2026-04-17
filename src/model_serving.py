import datetime
import json
import os
import pickle
import sys
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger

LOGGER = None
REPORTS_DIR = "reports"


def select_best_model():
    # Читаем имя лучшей модели из результатов валидации
    best_info_path = os.path.join(REPORTS_DIR, "best_model.json")
    if not os.path.exists(best_info_path):
        LOGGER.error("Файл best_model.json не найден, сначала запустите model_validation.py")
        raise FileNotFoundError("Сначала запустите src/model_validation.py")

    with open(best_info_path) as f:
        best_info = json.load(f)

    models_dir = CONFIG["model_training"]["models_dir"]
    best_model_path = CONFIG["model_serving"]["best_model_path"]

    best_path = os.path.join(models_dir, best_info["best_model"])
    model = joblib.load(best_path)
    joblib.dump(model, best_model_path)
    LOGGER.info(f"Лучшая модель сохранена: {best_model_path}")
    return model


def _preprocess_input(df):
    # Применяем те же преобразования что и при обучении
    encoders_path = CONFIG["model_training"]["encoders_path"]
    if not os.path.exists(encoders_path) or not os.path.exists("models/scaler_A.pkl"):
        LOGGER.error("Не найдены encoders.pkl или scaler_A.pkl")
        raise FileNotFoundError("Сначала запустите src/data_preparation.py")

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    with open("models/scaler_A.pkl", "rb") as f:
        scaler = pickle.load(f)

    date_fmt = CONFIG["data_preparation"]["date_format"]
    drop_cols = CONFIG["data_preparation"]["drop_columns"]
    target = CONFIG["data_preparation"]["target_column"]

    df = df.copy()

    if target in df.columns:
        df = df.drop(columns=[target])

    if "INSR_BEGIN" in df.columns and "INSR_END" in df.columns:
        df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format=date_fmt, errors="coerce")
        df["INSR_END"] = pd.to_datetime(df["INSR_END"], format=date_fmt, errors="coerce")
        df["INSR_DURATION"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
        df["INSR_YEAR"] = df["INSR_BEGIN"].dt.year

    df = df.drop(columns=drop_cols, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    feature_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else df.columns.tolist()
    valid_cols = [c for c in feature_cols if c in df.columns]
    df[valid_cols] = scaler.transform(df[valid_cols])

    return df


def predict(input_path):
    best_model_path = CONFIG["model_serving"]["best_model_path"]
    predictions_path = CONFIG["model_serving"]["predictions_path"]
    performance_log = CONFIG["model_serving"]["performance_log"]

    if not os.path.exists(best_model_path):
        select_best_model()

    model = joblib.load(best_model_path)
    df_raw = pd.read_csv(input_path)
    df_processed = _preprocess_input(df_raw)

    # Замеряем время и память
    tracemalloc.start()
    start_time = time.time()

    predictions = model.predict(df_processed)

    elapsed = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    df_raw["predict"] = predictions
    df_raw.to_csv(predictions_path, index=False)

    # Логируем производительность
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_file": input_path,
        "rows_predicted": len(df_raw),
        "elapsed_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak / 1024 / 1024, 4),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    logs = []
    if os.path.exists(performance_log):
        with open(performance_log) as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(performance_log, "w") as f:
        json.dump(logs, f, indent=2)

    LOGGER.info(f"Предсказания сохранены: {predictions_path} ({elapsed:.3f}с, пик {peak/1024/1024:.2f} МБ)")
    return predictions_path


def get_summary_report():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {}

    # Собираем все доступные отчёты в один файл
    for key, file in [
        ("validation_results", "validation_results.json"),
        ("best_model", "best_model.json"),
        ("feature_importances", "feature_importances.json"),
    ]:
        fpath = os.path.join(REPORTS_DIR, file)
        if os.path.exists(fpath):
            with open(fpath) as f:
                report[key] = json.load(f)

    performance_log = CONFIG["model_serving"]["performance_log"]
    if os.path.exists(performance_log):
        with open(performance_log) as f:
            report["performance_log"] = json.load(f)

    summary_path = os.path.join(REPORTS_DIR, "summary_report.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    LOGGER.info(f"Итоговый отчёт сохранён: {summary_path}")
    return summary_path


def setup():
    global LOGGER
    LOGGER = setup_logger(
        "ModelServing",
        log_file=CONFIG["model_serving"]["log_file"],
        level=CONFIG["logging"]["level"],
    )


if __name__ == "__main__":
    setup()
    select_best_model()
