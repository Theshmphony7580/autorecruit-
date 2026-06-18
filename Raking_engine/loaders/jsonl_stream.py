import json
from typing import Iterator

def stream_candidates(path: str) -> Iterator[dict]:
    """
    Stream candidates from JSONL file. Yields candidate dicts one by one.
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
            except json.JSONDecodeError:
                continue
