from pathlib import Path


def load_knowledge(file_path: str) -> str:

    path = Path(file_path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()