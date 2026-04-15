# Insurance Value — MLOps Pipeline

Предсказание размера страховой выплаты (CLAIM_PAID) по данным Ethiopian Insurance Corporation.

## Структура проекта

```
Insurance_value/
├── data/
│   ├── raw/            # Исходные данные (батчи от Stage 1)
│   ├── cleaned/        # Очищенные данные (от Stage 2)
│   └── processed/      # Подготовленные данные (Stage 3)
├── models/
│   └── versions/       # Версии обученных моделей
├── reports/            # Метрики качества и отчёты
├── data_collection/    # Stage 1 — сбор данных
├── data_analysis/      # Stage 2 — анализ данных
├── data_preparation/   # Stage 3 — подготовка данных
├── model_training/     # Stage 4 — обучение моделей
├── model_validation/   # Stage 5 — валидация моделей
├── model_serving/      # Stage 6 — сервинг модели
├── requirements.txt
└── run.py
```

## Датасет

Файл: `motor_data14-2018.csv` (~508 000 строк, 16 признаков)  
Целевая переменная: `CLAIM_PAID` — сумма страховой выплаты (0 если выплат не было)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск обучения (вручную, по этапам)

```bash
# Поместить датасет
cp /путь/к/motor_data14-2018.csv data/raw/

# Stage 3 — подготовка данных
python data_preparation/preparator.py

# Stage 4 — обучение моделей
python model_training/trainer.py

# Stage 4 — дообучение на втором датасете (когда нужно)
python model_training/trainer.py --retrain /путь/к/motor_data11-14lats.csv

# Stage 5 — валидация
python model_validation/validator.py

# Stage 6 — выбор лучшей модели
python model_serving/server.py
```

## Запуск через run.py

```bash
# Предсказание (inference)
python run.py -mode "inference" -file "./data/raw/motor_data14-2018.csv"

# Полный цикл переобучения (update)
python run.py -mode "update"

# Отчёт о состоянии системы (summary)
python run.py -mode "summary"
```

## Модели

Обучаются три модели в двух вариантах препроцессинга (StandardScaler / MinMaxScaler):
- Decision Tree (max_depth=10)
- k-Nearest Neighbors (k=5)
- SGD Regressor (поддерживает дообучение через `partial_fit`)

Лучшая модель выбирается по минимальному MAE на тестовой выборке.
