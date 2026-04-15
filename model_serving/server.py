import datetime
import json
import os
import pickle
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd

REPORTS_DIR = "reports"
MODELS_DIR = "models/versions"
BEST_MODEL_PATH = "models/best_model.pkl"
PERFORMANCE_LOG = "reports/performance_log.json"


def select_best_model():
    # Читаем имя лучшей модели из результатов валидации
    best_info_path = os.path.join(REPORTS_DIR, "best_model.json")
    if not os.path.exists(best_info_path):
        raise FileNotFoundError("Сначала запустите model_validation/validator.py")

    with open(best_info_path) as f:
        best_info = json.load(f)

    best_path = os.path.join(MODELS_DIR, best_info["best_model"])
    model = joblib.load(best_path)
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"Лучшая модель сохранена: {BEST_MODEL_PATH}")
    return model


def _preprocess_input(df):
    # Применяем те же преобразования что и при обучении
    if not os.path.exists("models/encoders.pkl") or not os.path.exists("models/scaler_A.pkl"):
        raise FileNotFoundError("Сначала запустите data_preparation/preparator.py")

    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/scaler_A.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = df.copy()

    if "CLAIM_PAID" in df.columns:
        df = df.drop(columns=["CLAIM_PAID"])

    if "INSR_BEGIN" in df.columns and "INSR_END" in df.columns:
        df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
        df["INSR_END"] = pd.to_datetime(df["INSR_END"], format="%d-%b-%y", errors="coerce")
        df["INSR_DURATION"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
        df["INSR_YEAR"] = df["INSR_BEGIN"].dt.year
        df = df.drop(columns=["INSR_BEGIN", "INSR_END"], errors="ignore")

    df = df.drop(columns=["OBJECT_ID"], errors="ignore")

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
    if not os.path.exists(BEST_MODEL_PATH):
        select_best_model()

    model = joblib.load(BEST_MODEL_PATH)
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

    output_path = "data/predictions.csv"
    df_raw.to_csv(output_path, index=False)

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
    if os.path.exists(PERFORMANCE_LOG):
        with open(PERFORMANCE_LOG) as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(PERFORMANCE_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"Предсказания сохранены: {output_path} ({elapsed:.3f}с, пик памяти {peak/1024/1024:.2f} МБ)")
    return output_path


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

    if os.path.exists(PERFORMANCE_LOG):
        with open(PERFORMANCE_LOG) as f:
            report["performance_log"] = json.load(f)

    summary_path = os.path.join(REPORTS_DIR, "summary_report.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Итоговый отчёт сохранён: {summary_path}")
    return summary_path


if __name__ == "__main__":
    select_best_model()
