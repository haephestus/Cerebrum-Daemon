import json
import sqlite3
from pathlib import Path

from cerebrum_core.file_util_inator import CEREBRUM_PATHS

DB_PATH = CEREBRUM_PATHS.get_card_bd() / "learning_center_cards.sqlite"

CREATE_CARDS_TABLE = """
    CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    bubble_id TEXT,
    create at TEXT,
    updated_at TEXT
);
"""


def conn():
    conn = sqlite3.connect(str(DB_PATH))
