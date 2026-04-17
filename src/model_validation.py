import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger

LOGGER = None
REPORTS_DIR = "reports"


def validate_models():
    global LOGGER
    LOGGER = setup_logger(
        "ModelValidation",
        log_file=CONFIG["model_validation"]["log_file"],
        level=CONFIG["logging"]["level"],
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    models_dir = CONFIG["model_training"]["models_dir"]
    processed_a = CONFIG["data_preparation"]["processed_a"]
    target = CONFIG["data_preparation"]["target_column"]
    train_split = CONFIG["model_training"]["train_split"]

    # Загружаем подготовленные данные и воссоздаём тестовую выборку
    df = pd.read_csv(processed_a)
    X = df.drop(columns=[target])
    y = df[target]

    split_idx = int(len(df) * train_split)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model_files = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))

    results = {}
    best_model_file = None
    best_mae = float("inf")

    for path in model_files:
        model = joblib.load(path)
        name = os.path.basename(path)

        if not hasattr(model, "predict"):
            continue

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            LOGGER.warning(f"Пропускаем {name}: {e}")
            continue

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        LOGGER.info(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model_file = name

    # Сохраняем метрики всех моделей
    with open(os.path.join(REPORTS_DIR, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Сохраняем имя лучшей модели
    best_info = {"best_model": best_model_file, "mae": round(best_mae, 4)}
    with open(os.path.join(REPORTS_DIR, "best_model.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    LOGGER.info(f"Лучшая модель: {best_model_file} (MAE={best_mae:.2f})")

    # Важность признаков по дереву решений
    dt_files = sorted([p for p in model_files if "decision_tree" in os.path.basename(p)])
    if dt_files:
        dt_model = joblib.load(dt_files[-1])
        if hasattr(dt_model, "feature_importances_"):
            importances = dict(zip(X.columns, dt_model.feature_importances_))
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            LOGGER.info("Важность признаков (Decision Tree):")
            for feat, imp in importances.items():
                LOGGER.info(f"  {feat}: {imp:.4f}")
            with open(os.path.join(REPORTS_DIR, "feature_importances.json"), "w") as f:
                json.dump(importances, f, indent=2)

    return results


if __name__ == "__main__":
    validate_models()
