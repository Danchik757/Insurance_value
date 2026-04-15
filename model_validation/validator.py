import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_A = "data/processed/prepared_A.csv"
MODELS_DIR = "models/versions"
REPORTS_DIR = "reports"


def validate_models():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Загружаем подготовленные данные и воссоздаём тестовую выборку
    df = pd.read_csv(PROCESSED_A)
    X = df.drop(columns=["CLAIM_PAID"])
    y = df["CLAIM_PAID"]

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))

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
            print(f"Пропускаем {name}: {e}")
            continue

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

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

    print(f"\nЛучшая модель: {best_model_file} (MAE={best_mae:.2f})")

    # Важность признаков по дереву решений
    dt_files = sorted([p for p in model_files if "decision_tree" in os.path.basename(p)])
    if dt_files:
        dt_model = joblib.load(dt_files[-1])
        if hasattr(dt_model, "feature_importances_"):
            importances = dict(zip(X.columns, dt_model.feature_importances_))
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            print("\nВажность признаков (Decision Tree):")
            for feat, imp in importances.items():
                print(f"  {feat}: {imp:.4f}")
            with open(os.path.join(REPORTS_DIR, "feature_importances.json"), "w") as f:
                json.dump(importances, f, indent=2)

    return results


if __name__ == "__main__":
    validate_models()
