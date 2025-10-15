from dataclasses import asdict, dataclass
from datetime import datetime
import sqlite3
import traceback
from typing import Literal

import pandas

from import_data import DB_PATH

@dataclass(frozen=True)
class Row:
    timestamp: float
    hundredsOfSeconds: int
    value: int

SELECT_TEMPLATE = "SELECT {columns} FROM {tablename}"
WHERE_TEMPLATE = "WHERE {condition}"
ORDER_BY_TEMPLATE = "ORDER BY {columnname} {argument}"
LIMIT_TEMPLATE = "LIMIT {value}"

def read_from_db(
        *, 
        tablename: str,
        columns: Literal["timestamp", "hundredsOfSecond", "value", "*"] = "*",
        condition: str | None = None,
        columnname: Literal["timestamp", "hundredsOfSecond", "value"] | None = None,
        argument: Literal["ASC", "DESC"] = "DESC",
        value: int | None = None
    ) -> pandas.DataFrame | None:
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()

        sql = SELECT_TEMPLATE.format(tablename=tablename, columns=columns)

        if condition is not None:
            sql += "\n" + WHERE_TEMPLATE.format(condition=condition)
        if columnname is not None:
            sql += "\n" + ORDER_BY_TEMPLATE.format(columnname=columnname, argument=argument)
        if value is not None:
            sql += "\n" + LIMIT_TEMPLATE.format(value=str(value))

        sql += ";"
        cursor.execute(sql)
        rows = cursor.fetchall()

        result = []
        first_timestamp = 0
        
        for i, row in enumerate(rows):
            _, timestamp, hundredsOfSecond, rowValue = row
            _hundredsOfSecond = int(hundredsOfSecond)
            _rowValue = int(rowValue)
            try:
                _timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            except ValueError:
                _timestamp = datetime.strptime(timestamp + ".000", "%Y-%m-%d %H:%M:%S.%f").timestamp()

            if i == 0:
                first_timestamp = _timestamp

            _timestamp -= first_timestamp
            
            result.append(
                Row(
                    timestamp=_timestamp,
                    hundredsOfSeconds=_hundredsOfSecond,
                    value=_rowValue
                )
            )
            
        return pandas.DataFrame([asdict(row) for row in result])
    except Exception as exc:
        traceback.print_exception(exc)
        return None
    finally:
        connection.close()