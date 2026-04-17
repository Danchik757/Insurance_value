import argparse
from pathlib import Path
import os
import json
from math import ceil

from src.data_collection import collect_data
from src.data_analysis import analyse_data
from src.data_preparation import prepare_data
from src.model_training import train_models
from src.model_validation import validate_models
from src.model_serving import predict, get_summary_report

from src.utils.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.storage import DatabaseStorage

logger = setup_logger("Entry Point", log_file=CONFIG["logging"]["path"], level=CONFIG["logging"]["level"])

class View:
    def __init__(self):
        pass

    def log_path(self, path):
        print(f"Путь до файла: {path}")

    def log_boolean(self, value, message_if_true="Да", message_if_false="Нет"):
        print(message_if_true if value else message_if_false)

    def log_struct(self, data, offset=0):
        OFFSET_STR = "    "

        if type(data) is dict:
            if offset > 0 :
                print()
            if len(data) == 0 :
                print(OFFSET_STR * offset, "...")
            for i in data:
                print(OFFSET_STR * offset, i, ": ", sep="", end="")
                self.log_struct(data[i], offset+1)
        else:
            print(data)

class Model:
    def __init__(self, view):
        self.view = view

    def inference(self, path):
        predict(path)
        self.view.log_path(CONFIG["model_serving"]["predictions_path"])

    def update(self):
        train_models()
        validate_models()

        self.view.log_boolean(True, "Модель успешно дообучена", "Ошибка при дообучении модели")

    def summary(self):
        self.view.log_path(get_summary_report())

    def default(self):
        collect_data()
        analyse_data()
        prepare_data()
        train_models()
        validate_models()

        self.view.log_boolean(True, "Модель успешно обучена", "Ошибка при обучении модели")

    def dashboard(self):
        d = {}
        d["Data Collection"] = {}
        d["Data Collection"]["Число батчей сырых данных"] = DatabaseStorage(CONFIG["storage"]["raw_table"]).fetch_next_index_to_add()
        d["Data Analysis"] = {}
        d["Data Analysis"]["Число батчей очищенных данных"] = DatabaseStorage(CONFIG["storage"]["cleaned_table"]).fetch_next_index_to_add()
        d["Data Preparation"] = {}
        d["Data Preparation"]["Объем подготовленных данных"] = f"{ceil((os.path.getsize(CONFIG["data_preparation"]["processed_a"]) + os.path.getsize(CONFIG["data_preparation"]["processed_b"])) / 1024 / 1024)} MB"
        d["Model Training"] = {}
        d["Model Training"]["Число обученных моделей"] = len(list(Path('./models/versions').glob("*.pkl")))
        d["Model Validation"] = {}
        d["Model Validation"]["Лучшая модель"] = {}
        
        best_model = Path(CONFIG["logging"]["reports_dir"]) / "best_model.json"
        if best_model.is_file():
            with open(best_model, "r", encoding="utf-8") as file:
                data = json.load(file)
                d["Model Validation"]["Лучшая модель"]["Название"] = data["best_model"]
                d["Model Validation"]["Лучшая модель"]["MAE"] = data["mae"]
        d["Model Serving"] = {}

        self.view.log_struct(d)

class Controller:
    def __init__(self, model):
        self.model = model

    def run(self) :
        parser = argparse.ArgumentParser()
        parser.add_argument("-mode", type=str, default=None, help="Режим работы: \"inference\", \"update\" или \"summary\"")
        parser.add_argument("-file", type=Path, default=None, help="Путь к файлу для обработки")

        args = parser.parse_args()

        if args.mode:
            if args.mode == "inference":
                if not args.file :
                    logger.error("Путь к файлу с данными не указан")
                else :
                    self.model.inference(args.file)
            elif args.mode == "update":
                self.model.update()
            elif args.mode == "summary":
                self.model.summary()
            elif args.mode == "dashboard":
                self.model.dashboard()
        else:
            self.model.default()
                

if __name__ == "__main__":
    controller = Controller(Model(View()))
    controller.run()