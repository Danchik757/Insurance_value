import argparse
from pathlib import Path

from src.data_collection import collect_data
from src.data_preparation import prepare_data

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
        pass

    def update(self):
        pass

    def summary(self):
        pass

class Controller:
    def __init__(self, model):
        self.model = model

    def run() :
        parser = argparse.ArgumentParser()
        parser.add_argument("-mode", type=str, default=None, help="Режим работы: \"inference\", \"update\" или \"summary\"")
        parser.add_argument("-file", type=Path, default=None, help="Путь к файлу для обработки")

if __name__ == "__main__":
    controller = Controller(Model(View()))
    controller.run()