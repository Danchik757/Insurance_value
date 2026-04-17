import sqlite3
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
from src.utils.config import CONFIG

_CONNECTION = sqlite3.connect(CONFIG["storage"]["path"])
cur = _CONNECTION.cursor()
cur.execute(f"CREATE TABLE IF NOT EXISTS {CONFIG["storage"]["metadata_table"]} (id INT PRIMARY KEY, timestamp REAL NOT NULL, sources TEXT NOT NULL, index_in_source INT NOT NULL, size INT NOT NULL, data_collection_version TEXT NOT NULL, data_analysis_version TEXT DEFAULT '');")
_CONNECTION.commit()

class DatabaseStorage:
    def __init__(self, table_name):
        self._table_name = table_name
        self._init_db()

    def _init_db(self):
        cur = _CONNECTION.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name} (id INT PRIMARY KEY, data_json TEXT NOT NULL);")
        _CONNECTION.commit()

    def save_batch(self, index, data, meta={}):
        cur = _CONNECTION.cursor()
        cur.execute(f"INSERT OR REPLACE INTO {self._table_name} (id, data_json) VALUES (?, ?);", (index, data.to_json()))

        if meta :
            cur.execute(f"UPDATE {CONFIG["storage"]["metadata_table"]} SET {", ".join(map(lambda x : f"{x} = ?", meta))} WHERE id = ?;", (*meta.values(), index))
            if cur.rowcount == 0:
                cur.execute(f"INSERT INTO {CONFIG["storage"]["metadata_table"]} (id, {", ".join(meta)}) VALUES (?{", ?" * len(meta)});", (index, *meta.values()))

        _CONNECTION.commit()

    def read_batch(self, index):
        cur = _CONNECTION.cursor()
        cur.execute(f"SELECT data_json FROM {self._table_name} WHERE id = {index}")
        data_json = cur.fetchone()
        if data_json :
            data = json.loads(data_json[0])
            return pd.DataFrame(data)
        else :
            return None

    def read(self) :
        i = 0
        while True :
            res = self.read_batch(i)
            if res is None :
                break
            yield res
            i += 1

    def fetch_next_index_to_add(self, meta={}) :
        cur = _CONNECTION.cursor()
        if meta :
            ...
        else :
            cur.execute(f"SELECT MAX(id) FROM {self._table_name}")
        data = cur.fetchone()
        if data and data[0] :
            return data[0] + 1

        return 0

    def fetch_next_index_in_source_to_add(self, sources) :
        cur = _CONNECTION.cursor()
        cur.execute(f"SELECT MAX(index_in_source), size FROM {CONFIG["storage"]["metadata_table"]} WHERE sources = '{sources}';")
        data = cur.fetchone()
        if data and data[0] and data[1] :
            return data[0] + data[1]

        return 0