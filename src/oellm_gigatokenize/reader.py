"""Discovery and streaming of zstd-compressed JSON Lines input files."""

import io
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import orjson
import zstandard

from oellm_gigatokenize.partitioning import round_robin

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


@contextmanager
def _open_decompressed(path: Path) -> Generator[io.BufferedReader]:
    decompressor = zstandard.ZstdDecompressor()
    with path.open("rb") as raw, decompressor.stream_reader(raw, read_across_frames=True) as stream:
        yield io.BufferedReader(stream, buffer_size=1 << 20)


def iter_rows(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield ``(line_number, line)`` for every non-blank line of a zstd-compressed file.

    Each non-blank line is one document. Line numbers start at 1 and count blank
    lines too, so they match what an editor shows.
    """
    with _open_decompressed(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield line_number, line


def parse_document(line: bytes, text_key: str, location: str) -> str:
    """The ``text_key`` field of one JSON Lines row."""
    try:
        record = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        raise InputFormatError(f"{location}: invalid JSON: {exc}") from exc
    if not isinstance(record, dict) or text_key not in record:
        raise InputFormatError(f"{location}: missing key {text_key!r}")
    text = record[text_key]
    if not isinstance(text, str):
        raise InputFormatError(f"{location}: {text_key!r} is not a string")
    return text


def iter_documents(path: Path, text_key: str = "text", partition: int = 0, num_partitions: int = 1) -> Iterator[str]:
    """Stream the documents of one round-robin partition of a zstd-compressed JSONL file.

    Document ``i`` (the ``i``-th non-blank row, from 0) belongs to partition
    ``i % num_partitions``. The whole file is decompressed, but only the rows of
    this partition are parsed. Memory use does not depend on the file size.
    """
    for line_number, line in round_robin(iter_rows(path), partition, num_partitions):
        yield parse_document(line, text_key, f"{path}:{line_number}")


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
