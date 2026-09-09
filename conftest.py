import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
import database


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point config.DB_PATH at a fresh temp file for the duration of a test."""
    db_path = tmp_path / "test_memory.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_path))
    database.init_db()
    yield str(db_path)
