# Insurance Value — MLOps Pipeline

Предсказание размера страховой выплаты (CLAIM_PAID) по данным Ethiopian Insurance Corporation.

## Структура проекта

```
Insurance_value/
├── config.yaml         # настройки пайплайна (пути, параметры моделей)
├── requirements.txt
├── run.py              # точка входа (inference / update / summary)
├── src/
│   ├── data_collection.py    # Stage 1 — сбор данных
│   ├── data_analysis.py      # Stage 2 — анализ данных
│   ├── data_preparation.py   # Stage 3 — подготовка данных
│   ├── model_training.py     # Stage 4 — обучение моделей
│   ├── model_validation.py   # Stage 5 — валидация моделей
│   ├── model_serving.py      # Stage 6 — сервинг модели
│   └── utils/
│       ├── config.py         # загрузка config.yaml
│       ├── logger.py         # настройка логгера
│       └── storage.py        # работа с SQLite БД
├── data/
│   ├── raw/            # исходные CSV файлы
│   └── processed/      # подготовленные данные для обучения
├── models/
│   └── versions/       # версии обученных моделей
└── reports/            # метрики, отчёты, логи производительности
```

## Датасет

Источник: [Vehicle Insurance Data — Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/vehicle-insurance-data)
Файлы:
- `motor_data11-14.csv` (~294 000 строк, 16 признаков)
- `motor_data14-2018.csv` (~508 000 строк, 16 признаков)
Целевая переменная: `CLAIM_PAID` — сумма страховой выплаты (0 если выплат не было)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск обучения (вручную, по этапам)

Все команды выполнять из корня репозитория `Insurance_value/`.

```bash
# Поместить датасеты
cp /путь/к/motor_data11-14.csv data/raw/
cp /путь/к/motor_data14-2018.csv data/raw/

# Stage 1 — сбор данных
python src/data_collection.py

# Stage 2 — анализ данных
python src/data_analysis.py

# Stage 3 — подготовка данных
python src/data_preparation.py

# Stage 4 — обучение моделей
python src/model_training.py

# Stage 4 — дообучение на втором датасете (когда нужно)
python src/model_training.py --retrain data/raw/motor_data11-14lats.csv

# Stage 5 — валидация
python src/model_validation.py

# Stage 6 — выбор лучшей модели
python src/model_serving.py
```

## Запуск через run.py

```bash
# Вывести список доступных опций
python run.py --help

# Предсказание (inference)
python run.py -mode "inference" -file "./data/raw/motor_data14-2018.csv"

# Полный цикл обучения/дообучения (update)
python run.py -mode "update"

# Отчёт о состоянии системы (summary)
python run.py -mode "summary"

# Отчёт расширенный отчет о состоянии системы (dashboard)
python run.py -mode "dashboard"
```

## Модели

Обучаются три модели в двух вариантах препроцессинга (StandardScaler / MinMaxScaler):
- Decision Tree (max_depth=10)
- k-Nearest Neighbors (k=5)
- SGD Regressor (поддерживает дообучение через `partial_fit`)

Лучшая модель выбирается по минимальному MAE на тестовой выборке.