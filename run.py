import argparse
from pathlib import Path

from src.data_collection import collect_data
# TODO: data_analysis
from src.data_preparation import prepare_data
from src.model_training import train_models
from src.model_validation import validate_models
from src.model_serving import predict, get_summary_report

from src.utils.config import CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("Entry Point", log_file=CONFIG["logging"]["path"], level=CONFIG["logging"]["level"])

class View:
    def __init__(self):
        pass

    def log_path(self, path):
        print(f"Путь до файла: {path}")

    def log_boolean(self, value, message_if_true="Да", message_if_false="Нет"):
        print(message_if_true if value else message_if_false)

class Model:
    def __init__(self, view):
        self.view = view

    def inference(self, path):
        predict(path)
        self.view.log_path(CONFIG["model_serving"]["predictions_path"])

    def update(self):
        train_models()

        self.view.log_boolean(True, "Модель успешно дообучена", "Ошибка при дообучении модели")

    def summary(self):
        # TODO: data_analysis
        validate_models()
        get_summary_report()

    def default(self):
        collect_data()
        # TODO: data_analysis
        prepare_data()
        train_models()
        validate_models()

        self.view.log_boolean(True, "Модель успешно обучена", "Ошибка при обучении модели")

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
        else:
            self.model.default()
                

if __name__ == "__main__":
    controller = Controller(Model(View()))
    controller.run()