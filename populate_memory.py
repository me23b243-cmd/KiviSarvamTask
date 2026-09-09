import json

import database
import memory_store
from memory import MemoryElement


MEMORY_JSON = "memory_export.json"


def load_memory_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    memory = {}

    for representative, item in data.items():
        element = MemoryElement(
            og_word=item["og_word"],
            pref_word=item["pref_word"],
            context=item.get("context", []),
            occurrence=int(item.get("occurrence", 0)),
            representative=item.get(
                "representative",
                representative
            ),
        )

        memory[representative] = element

    return memory


def main():
    print("Initializing database...")
    database.init_db()

    print(f"Loading memory from {MEMORY_JSON}...")
    memory = load_memory_from_json(MEMORY_JSON)

    print(f"Loaded {len(memory)} memory elements.")

    print("Writing memory to SQLite...")
    memory_store.persist_memory_elements(memory)

    print("Done.")
    print(f"Database now contains approximately {database.count_rows()} memory elements.")


if __name__ == "__main__":
    main()