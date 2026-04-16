import sqlite3
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
from src.utils.config import CONFIG

class DatabaseStorage:
    def __init__(self, table_name):
        self.conn = sqlite3.connect(CONFIG["storage"]["path"])
        self._table_name = table_name
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name} (id INT PRIMARY KEY, data_json TEXT)")
        self.conn.commit()

    def save_batch(self, batch_data):
        cur = self.conn.cursor()
        cur.execute(f"INSERT OR REPLACE INTO {self._table_name} (id, data_json) VALUES (?, ?)", (batch_data["id"], batch_data["data"].to_json()))
        self.conn.commit()

    def read_batch(self, index):
        cur = self.conn.cursor()
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