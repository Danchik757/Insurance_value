import yaml
import logging
import sys
from abc import ABC, abstractmethod
import pandas as pd
import time
import sqlite3

VERSION = "1.0.0"

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

LOGGER = None

def setup_logger(name: str, log_file: str, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

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
            LOGGER.error("Attempting to read data outside of data source")
            return None
        
        if n <= 0 :
            LOGGER.error("Attempting to read data with non-positive size")
            return None
        
        if i + n > self.get_size() :
            LOGGER.warning("Trying to read data outside of data source, the data will be trimmed")

        return self._data.iloc[i:min(i+n, self.get_size())]

class CompositeSource(DataSource):
    def __init__(self, sources):
        self.sources = sources
        self._total_size = sum(i.get_size() for i in self.sources)
        if len(self.sources) == 0 :
            LOGGER.warning("Trying to create composite data source without any sources")
    
    def get_size(self):
        return self._total_size

    def get_data(self, i, n):
        if len(self.sources) == 0 :
            LOGGER.error("Attempting to read data from composite data source without any sources")
            return None

        offset = 0
        idx = 0
        compound = None
        while i >= offset + self.sources[idx].get_size() :
            offset += self.sources[idx].get_size()
            idx += 1
            if idx >= len(self.sources) :
                LOGGER.error("Attempting to read data outside of data source")
                return None
        
        compound = self.sources[idx].get_data(i - offset, min(self.sources[idx].get_size() + offset - i, n))
        if compound is None :
            return None
        compound = compound.reset_index(drop=True)
        offset += self.sources[idx].get_size()
        idx += 1
        
        while i + n > offset :
            if idx >= len(self.sources) :
                LOGGER.warning("Trying to read data outside of data source, the data will be trimmed")
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
    def __init__(self, data_source, batch_size, delay=0.1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.delay = delay

    def stream(self):
        for i in range(0, self.data_source.get_size(), self.batch_size):
            batch_data = self.data_source.get_data(i, min(self.batch_size, self.data_source.get_size() - i))

            if batch_data is None :
                break

            yield {
                "id": i // self.batch_size,
                "data": batch_data
            }

            time.sleep(self.delay)

class DatabaseStorage:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS raw_batches (id INT PRIMARY KEY, data_json TEXT)")
        self.conn.commit()

    def save_batch(self, batch_data):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO raw_batches (id, data_json) VALUES (?, ?)", (batch_data["id"], batch_data["data"].to_json()))
        self.conn.commit()

def collect_data():
    global LOGGER

    config = load_config()
    LOGGER = setup_logger("DataCollection", log_file=config["logging"]["path"], level=config["logging"]["level"])

    sources = CompositeSource([CSVDataSource(i) for i in config["dataset"]["paths"]])

    streamer = StreamEmulator(data_source=sources, batch_size=config["stream"]["batch_size"], delay=config["stream"]["delay_seconds"])

    storage = DatabaseStorage(config["storage"]["path"])

    metas = []

    LOGGER.info("Data Collection began")
    for batch in streamer.stream():
        try:
            storage.save_batch(batch)
            LOGGER.info(f"Saving batch {batch["id"]} of size {len(batch["data"])}")

            if config["meta"]["compute_per_batch"]:
                meta = {
                    "timestamp": time.time(),
                    "sources": config["dataset"]["paths"],
                    "version": VERSION
                }
                metas.append(meta)
                LOGGER.info(f"Meta parameters for batch {batch["id"]}: timestamp={meta["timestamp"]}; sources={meta["sources"]}; version={meta["version"]}")

        except Exception as e:
            LOGGER.error(f"Exception processing batch {batch["id"]}: {e}", exc_info=True)
            break

    if not config["meta"]["compute_per_batch"] :
        meta = {
            "timestamp": time.time(),
            "sources": config["dataset"]["paths"],
            "version": VERSION
        }
        metas.append(meta)
        LOGGER.info(f"Meta parameters for all batches: timestamp={meta["timestamp"]}; sources={meta["sources"]}; version={meta["version"]}")

    LOGGER.info("Data Collection ended")
    return config["storage"]["path"], metas

if __name__ == "__main__":
    collect_data()