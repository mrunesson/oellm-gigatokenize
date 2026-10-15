"""Discovery and streaming of zstd-compressed JSON Lines input files."""

import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import orjson
import zstandard

DEFAULT_PATTERN = "*.jsonl.zst"


class InputFormatError(ValueError):
    """Raised when an input line cannot be interpreted as a document."""


@dataclass(frozen=True, slots=True)
class InputFile:
    """An input file together with its on-disk (compressed) size in bytes."""

    path: Path
    size: int


def discover_files(directory: Path, pattern: str = DEFAULT_PATTERN) -> list[InputFile]:
    """Find all files under ``directory`` (recursively) matching ``pattern``.

    The result is sorted by path so that runs are deterministic.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {directory}")
    paths = sorted(p for p in directory.rglob(pattern) if p.is_file())
    return [InputFile(path=p, size=p.stat().st_size) for p in paths]


def iter_documents(path: Path, text_key: str = "text") -> Iterator[str]:
    """Stream the ``text_key`` field of every JSON object in a zstd-compressed JSONL file.

    Decompression is streaming, so memory use does not depend on the file size.
    Blank lines are ignored.
    """
    decompressor = zstandard.ZstdDecompressor()
    with path.open("rb") as raw, decompressor.stream_reader(raw) as stream:
        lines = io.BufferedReader(stream, buffer_size=1 << 20)
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise InputFormatError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or text_key not in record:
                raise InputFormatError(f"{path}:{line_no}: missing key {text_key!r}")
            text = record[text_key]
            if not isinstance(text, str):
                raise InputFormatError(f"{path}:{line_no}: {text_key!r} is not a string")
            yield text


def iter_batches(documents: Iterable[str], max_batch_bytes: int) -> Iterator[list[str]]:
    """Group documents into batches of roughly ``max_batch_bytes`` characters each.

    A batch is emitted as soon as it reaches the budget, so a batch can exceed the
    budget by at most one document.
    """
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")
    batch: list[str] = []
    batch_size = 0
    for doc in documents:
        batch.append(doc)
        batch_size += len(doc)
        if batch_size >= max_batch_bytes:
            yield batch
            batch, batch_size = [], 0
    if batch:
        yield batch
