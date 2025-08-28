import json
import sqlite3
from datetime import datetime
from typing import List, Tuple


def _connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                top1_label TEXT NOT NULL,
                top1_confidence REAL NOT NULL,
                predictions_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        # Migration: add model_name column if missing
        cols = conn.execute("PRAGMA table_info(predictions)").fetchall()
        col_names = {c[1] for c in cols}
        if "model_name" not in col_names:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_name TEXT DEFAULT 'unknown'")
        conn.commit()


def insert_prediction(
    db_path: str,
    filename: str,
    top1_label: str,
    top1_confidence: float,
    predictions: List[Tuple[str, float]],
    model_name: str,
) -> None:
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    predictions_json = json.dumps([{"label": l, "prob": p} for (l, p) in predictions])
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO predictions (filename, top1_label, top1_confidence, predictions_json, created_at, model_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, top1_label, float(top1_confidence), predictions_json, created_at, model_name),
        )
        conn.commit()


def get_label_counts(db_path: str) -> List[Tuple[str, int]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT top1_label, COUNT(*) AS cnt
            FROM predictions
            GROUP BY top1_label
            ORDER BY cnt DESC, top1_label ASC
            LIMIT 50
            """
        ).fetchall()
        return [(row["top1_label"], int(row["cnt"])) for row in rows]


def get_recent_predictions(db_path: str, limit: int = 20):
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT filename, top1_label, top1_confidence, predictions_json, created_at, model_name
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows] 