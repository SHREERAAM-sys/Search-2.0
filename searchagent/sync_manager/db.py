import sqlite3
from typing import Any, List, Union

from searchagent.sync_manager.file_change_detector import FileChangeEvent

DB_NAME = "file_changes"


def execute_query(query: str, params=(), query_read=False) -> Union[List[Any], None]:
    with sqlite3.connect(f"{DB_NAME}.db") as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        return cur.fetchall() if query_read else None


def create_table_if_not_exists():
    query = f"""
        CREATE TABLE IF NOT EXISTS {DB_NAME} (
            filepath TEXT PRIMARY KEY,
            change_type TEXT,
            timestamp TEXT,
            dest_path TEXT
        )
    """
    execute_query(query)


def insert_data(event_list: List[FileChangeEvent]):
    create_table_if_not_exists()

    query = f"""
        INSERT OR REPLACE INTO {DB_NAME} (filepath, change_type, timestamp, dest_path)
        VALUES (?, ?, ?, ?)
    """

    for e in event_list:
        execute_query(
            query, (e.filepath, e.change_type.value, e.timestamp, e.dest_path)
        )


def read_db():
    query = f"SELECT * FROM {DB_NAME}"
    result = execute_query(query, query_read=True)
    for row in result:
        print(row)
