import sqlite3
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
from src.utils.config import CONFIG
from src.utils.logger import setup_logger

VERSION = "1.0.1"

LOGGER = setup_logger("Storage", log_file=CONFIG["logging"]["path"], level=CONFIG["logging"]["level"])

determined_version = ""
if os.path.exists(CONFIG["storage"]["path"]):
    if os.path.exists(CONFIG["storage"]["db_metadata"]):
        with open(CONFIG["storage"]["db_metadata"], "r") as f :
            determined_version = f.readline().strip()
    else:
        determined_version = "1.0.0"

_flag = True

if determined_version != "" and [i[0] - i[1] for i in zip(map(int, determined_version.split('.')), map(int, VERSION.split('.')))] > [0, 0, 0] :
    LOGGER.warning("Более новые версии базы данных могут не поддерживаться")
    _flag = False

_CONNECTION = sqlite3.connect(CONFIG["storage"]["path"])

if determined_version == "" :
    cur = _CONNECTION.cursor()
    cur.execute(f"CREATE TABLE IF NOT EXISTS {CONFIG["storage"]["metadata_table"]} (id INT PRIMARY KEY, timestamp REAL NOT NULL, sources TEXT NOT NULL, index_in_source INT NOT NULL, size INT NOT NULL, data_collection_version TEXT NOT NULL, data_analysis_version TEXT DEFAULT '', used_for_learning INTEGER CHECK (used_for_learning IN (0, 1)) DEFAULT 0);")
    _CONNECTION.commit()
elif determined_version == "1.0.0" :
    cur = _CONNECTION.cursor()
    cur.execute(f"ALTER TABLE {CONFIG["storage"]["metadata_table"]} ADD COLUMN used_for_learning INTEGER CHECK (used_for_learning IN (0, 1)) DEFAULT 0;")
    _CONNECTION.commit()

if _flag :
    with open(CONFIG["storage"]["db_metadata"], "w") as f :
        f.write(VERSION)

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

    def update_meta(self, index, meta):
        cur = _CONNECTION.cursor()

        cur.execute(f"UPDATE {CONFIG["storage"]["metadata_table"]} SET {", ".join(map(lambda x : f"{x} = ?", meta))} WHERE id = ?;", (*meta.values(), index))
        if cur.rowcount == 0:
            cur.execute(f"INSERT INTO {CONFIG["storage"]["metadata_table"]} (id, {", ".join(meta)}) VALUES (?{", ?" * len(meta)});", (index, *meta.values()))

        _CONNECTION.commit()

    def read_batch(self, index):
        cur = _CONNECTION.cursor()
        cur.execute(f"SELECT data_json FROM {self._table_name} WHERE id = {index};")
        data_json = cur.fetchone()
        if data_json :
            data = json.loads(data_json[0])
            return pd.DataFrame(data)
        else :
            return None

    def read(self, beg=0) :
        i = beg
        while True :
            res = self.read_batch(i)
            if res is None :
                break
            yield res
            i += 1

    def fetch_next_index_to_add(self, meta={}) :
        cur = _CONNECTION.cursor()
        if meta:
            cur.execute(f"SELECT MIN(id) FROM {self._table_name} WHERE id IN (SELECT id FROM {CONFIG["storage"]["metadata_table"]} WHERE {" OR ".join(map(lambda x : f"{x} != ?", meta))});", tuple(meta.values()))
            data = cur.fetchone()

            if not data or not data[0]:
                cur.execute(f"SELECT MAX(id) + 1 FROM {self._table_name} WHERE id IN (SELECT id FROM {CONFIG["storage"]["metadata_table"]} WHERE {" AND ".join(map(lambda x : f"{x} = ?", meta))});", tuple(meta.values()))
                data = cur.fetchone()
        else:
            cur.execute(f"SELECT MAX(id) + 1 FROM {self._table_name};")
            data = cur.fetchone()
        if data and data[0] :
            return data[0]

        return 0

    def fetch_next_index_in_source_to_add(self, sources, meta={}) :
        cur = _CONNECTION.cursor()
        if meta:
            cur.execute(f"SELECT MAX(index_in_source) + size FROM {CONFIG["storage"]["metadata_table"]} WHERE sources = '{sources}' AND id IN (SELECT id FROM {CONFIG["storage"]["metadata_table"]} WHERE {" AND ".join(map(lambda x : f"{x} = ?", meta))});", tuple(meta.values()))
        else:
            cur.execute(f"SELECT MAX(index_in_source) + size FROM {CONFIG["storage"]["metadata_table"]} WHERE sources = '{sources}';")
        data = cur.fetchone()
        if data and data[0] :
            return data[0]

        return 0
    
    def read_all_not_used_for_learning(self) :
        i = self.fetch_next_index_to_add(meta={"used_for_learning":1})
        while True :
            res = self.read_batch(i)
            if res is None :
                break
            self.update_meta(i, meta={"used_for_learning":1})
            yield res
            i += 1
    
def clear_database() :
    cur = _CONNECTION.cursor()
    cur.execute(f"DELETE FROM {CONFIG["storage"]["metadata_table"]};")
    cur.execute(f"DELETE FROM {CONFIG["storage"]["raw_table"]};")
    cur.execute(f"DELETE FROM {CONFIG["storage"]["cleaned_table"]};")
    _CONNECTION.commit()