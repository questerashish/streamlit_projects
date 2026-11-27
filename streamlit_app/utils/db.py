"""Database utilities for the desktop Streamlit trading suite."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

DB_FILE = Path(__file__).resolve().parents[1] / "app.db"
SCHEMA_VERSION = 1

TABLE_DEFINITIONS: dict[str, str] = {
    "schema_version": (
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    ),
    "trades": (
        "CREATE TABLE IF NOT EXISTS trades ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "date TEXT,"
        "symbol TEXT,"
        "segment TEXT,"
        "direction TEXT,"
        "qty REAL,"
        "entry REAL,"
        "exit REAL,"
        "fees REAL,"
        "sl REAL,"
        "tp REAL,"
        "timeframe TEXT,"
        "strategy TEXT,"
        "tags TEXT,"
        "notes TEXT,"
        "screenshot_path TEXT"
        ")"
    ),
    "daily_bars": (
        "CREATE TABLE IF NOT EXISTS daily_bars ("
        "symbol TEXT,"
        "date TEXT,"
        "o REAL,"
        "h REAL,"
        "l REAL,"
        "c REAL,"
        "v REAL,"
        "source_file TEXT,"
        "PRIMARY KEY(symbol, date)"
        ")"
    ),
    "returns": (
        "CREATE TABLE IF NOT EXISTS returns ("
        "date TEXT,"
        "symbol TEXT,"
        "ret REAL,"
        "PRIMARY KEY(date, symbol)"
        ")"
    ),
    "levels": (
        "CREATE TABLE IF NOT EXISTS levels ("
        "symbol TEXT,"
        "level_type TEXT,"
        "price REAL,"
        "date TEXT,"
        "confidence REAL,"
        "notes TEXT"
        ")"
    ),
    "strategies": (
        "CREATE TABLE IF NOT EXISTS strategies ("
        "name TEXT PRIMARY KEY,"
        "type TEXT,"
        "rules_json TEXT,"
        "capital_req REAL,"
        "tags TEXT,"
        "file_path TEXT"
        ")"
    ),
    "baskets": (
        "CREATE TABLE IF NOT EXISTS baskets ("
        "name TEXT PRIMARY KEY,"
        "components_json TEXT,"
        "capital_split_json TEXT,"
        "rebal_rule TEXT,"
        "notes TEXT"
        ")"
    ),
    "risk_settings": (
        "CREATE TABLE IF NOT EXISTS risk_settings ("
        "profile_name TEXT PRIMARY KEY,"
        "max_dd REAL,"
        "risk_per_trade REAL,"
        "kelly_cap REAL,"
        "heat_limits_json TEXT"
        ")"
    ),
}


def init_db(db_file: Optional[Path] = None) -> None:
    """Initialise the SQLite database and ensure schema version."""
    db_path = db_file or DB_FILE
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for ddl in TABLE_DEFINITIONS.values():
            cursor.execute(ddl)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
        if cursor.fetchone() is None:
            cursor.execute(TABLE_DEFINITIONS["schema_version"])
            cursor.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()


@contextmanager
def get_connection(db_file: Optional[Path] = None):
    """Yield a SQLite connection with row factory set to dictionary-like output."""
    db_path = db_file or DB_FILE
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def execute(query: str, params: Iterable | None = None) -> None:
    """Execute a write query."""
    with get_connection() as conn:
        conn.execute(query, params or [])
        conn.commit()


def fetch_all(query: str, params: Iterable | None = None) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(query, params or [])
        return cur.fetchall()


def upsert(table: str, data: dict) -> None:
    """Simple UPSERT helper using INSERT OR REPLACE."""
    columns = ",".join(data.keys())
    placeholders = ":" + ",:".join(data.keys())
    sql = f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})"
    execute(sql, data)
