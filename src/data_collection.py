from abc import ABC, abstractmethod
import pandas as pd
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger
from src.utils.storage import DatabaseStorage

VERSION = "1.0.0"

LOGGER = None

class DataSource(ABC):
    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def get_data(self, indices):
        pass

class CSVDataSource(DataSource):
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self._data = pd.read_csv(self.csv_path)
    
    def get_size(self):
        return len(self._data)

    def get_data(self, i, n):
        if i >= self.get_size() :
            LOGGER.error("Попытка прочитать данные вне диапазона")
            return None
        
        if n <= 0 :
            LOGGER.error("Попытка прочитать данные не положительного размера")
            return None
        
        if i + n > self.get_size() :
            LOGGER.warning("Попытка прочитать данные вне диапазона, данные будут обрезаны")

        return self._data.iloc[i:min(i+n, self.get_size())]

class CompositeSource(DataSource):
    def __init__(self, sources):
        self.sources = sources
        self._total_size = sum(i.get_size() for i in self.sources)
        if len(self.sources) == 0 :
            LOGGER.warning("Попытка создать составной источник данных без источников данных")
    
    def get_size(self):
        return self._total_size

    def get_data(self, i, n):
        if len(self.sources) == 0 :
            LOGGER.error("Попытка прочитать из составного источника данных без источников данных")
            return None

        offset = 0
        idx = 0
        compound = None
        while i >= offset + self.sources[idx].get_size() :
            offset += self.sources[idx].get_size()
            idx += 1
            if idx >= len(self.sources) :
                LOGGER.error("Попытка прочитать вне диапазона")
                return None
        
        compound = self.sources[idx].get_data(i - offset, min(self.sources[idx].get_size() + offset - i, n))
        if compound is None :
            return None
        compound = compound.reset_index(drop=True)
        offset += self.sources[idx].get_size()
        idx += 1
        
        while i + n > offset :
            if idx >= len(self.sources) :
                LOGGER.warning("Попытка прочитать данные вне диапазона, данные будут обрезаны")
                break

            append = self.sources[idx].get_data(0, min(self.sources[idx].get_size(), i + n - offset))

            if append is None :
                return None
            
            compound = pd.concat([compound, append], ignore_index=True)
            offset += self.sources[idx].get_size()
            idx += 1

        compound.index += i

        return compound

class StreamEmulator:
    def __init__(self, data_source, batch_size, delay=0.1, offset=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.delay = delay
        self.offset = offset

    def stream(self):
        for i in range(self.offset, self.data_source.get_size(), self.batch_size):
            batch_data = self.data_source.get_data(i, min(self.batch_size, self.data_source.get_size() - i))

            if batch_data is None :
                break

            yield {
                "data": batch_data,
                "index_in_source": i,
                "size": len(batch_data)
            }

            time.sleep(self.delay)

def collect_data():
    global LOGGER

    LOGGER = setup_logger("DataCollection", log_file=CONFIG["logging"]["path"], level=CONFIG["logging"]["level"])

    sources = CompositeSource([CSVDataSource(i) for i in CONFIG["dataset"]["paths"]])
    sources_name = ";".join(CONFIG["dataset"]["paths"])

    storage = DatabaseStorage(CONFIG["storage"]["raw_table"])

    index = storage.fetch_next_index_to_add()
    streamer = StreamEmulator(data_source=sources, batch_size=CONFIG["stream"]["batch_size"], delay=CONFIG["stream"]["delay_seconds"], offset=storage.fetch_next_index_in_source_to_add(sources_name))

    LOGGER.info("Data Collection начато")
    try :
        for batch in streamer.stream():
            try:
                meta = {
                    "timestamp": time.time(),
                    "sources": sources_name,
                    "index_in_source": batch["index_in_source"],
                    "size": batch["size"],
                    "data_collection_version": VERSION
                }
                storage.save_batch(index, batch["data"], meta)
                LOGGER.info(f"Сохранение батча {index} резмера {batch["size"]}")
                LOGGER.info(f"Метапараметры для батча {index}: timestamp={meta["timestamp"]}; sources={meta["sources"]}; data_collection_version={meta["data_collection_version"]}")
                index += 1

            except Exception as e:
                LOGGER.error(f"Ошибка при обработки батча {index}: {e}", exc_info=True)
                break
    except KeyboardInterrupt :
        LOGGER.warning(f"Обработка потока была прервана")

    LOGGER.info("Data Collection закончено")

if __name__ == "__main__":
    collect_data()