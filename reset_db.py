"""
Fully resets the memory database — deletes all learned corrections and
metrics history, leaving an empty, freshly-initialized schema.

Usage:
    python reset_db.py            # asks for confirmation
    python reset_db.py --yes      # skips confirmation (for scripts/CI)
"""

from __future__ import annotations

import argparse
import os
import sys

import config
import database


def reset() -> None:
    """Delete the SQLite database file entirely, then recreate an empty schema."""
    if os.path.exists(config.DB_PATH):
        os.remove(config.DB_PATH)
    database.init_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully reset the memory database.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    if not args.yes:
        answer = input(f"This will permanently delete {config.DB_PATH}. Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted — nothing was deleted.")
            sys.exit(0)

    reset()
    print(f"Reset complete. {config.DB_PATH} is now empty.")
    print("Run `python populate_memory.py` if you want to reseed it.")
