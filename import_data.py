import sqlite3
import os

import pandas as pd

DB_PATH = "data.db"

def get_filenames() -> list[str]:
    return list(filter(lambda file: file.endswith(".csv"), os.listdir(".")))

def get_tablename(filename: str) -> str:
    basename = os.path.splitext(filename)[0]
    tablename = basename.replace("ECG_samples_AD_", "")

    return tablename

def import_from_files() -> None:
    csv_files = get_filenames()
    connection = sqlite3.connect(DB_PATH)

    for filename in csv_files:
        tablename = get_tablename(filename)
        df = pd.read_csv(
            filename, 
            header=None, 
            usecols=[2, 4, 6],
            names=["timestamp", "hundredsOfSecond", "value"],
            dtype={"hundredsOfSecond": "int64", "value": "int64"}
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors="coerce")
        
        df.to_sql(
            tablename,
            connection,
            if_exists="replace",
            index=True,
            index_label="id",
            dtype={
                "timestamp": "TEXT",
                "hunderdOfSeconds": "INTEGER",
                "value": "INTEGER"
            }
        )

    connection.close()