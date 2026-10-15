"""Results of tokenizing partitions and files, and the JSON summary of a run."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from oellm_gigatokenize import __version__
from oellm_gigatokenize.indexed_dataset import bin_path, idx_path
from oellm_gigatokenize.reader import InputFile


def write_json(path: Path, data: object) -> None:
    """Write ``data`` as indented JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def load_if_matching(path: Path, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """The ``result`` stored in a manifest at ``path`` if it was produced from ``inputs``.

    Manifests are ``{"inputs": ..., "result": ...}``. Anything unreadable or
    mismatching counts as absent, so the work is redone.
    """
    try:
        stored = json.loads(path.read_text())
        if stored["inputs"] != json.loads(json.dumps(inputs)):
            return None
        return stored["result"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


@dataclass(slots=True)
class PartitionStats:
    documents: int = 0
    tokens: int = 0
    empty_documents: int = 0
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PartitionResult:
    number: int
    reused: bool
    stats: PartitionStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "reused": self.reused,
            **asdict(self.stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            number=data["number"],
            reused=data["reused"],
            stats=PartitionStats(
                documents=data["documents"],
                tokens=data["tokens"],
                empty_documents=data["empty_documents"],
                seconds=data["seconds"],
            ),
        )


@dataclass(frozen=True, slots=True)
class FileResult:
    """The outcome of tokenizing one input file into one output dataset."""

    input: InputFile
    output_prefix: Path
    reused: bool
    merge_seconds: float
    partitions: list[PartitionResult]

    @property
    def documents(self) -> int:
        return sum(p.stats.documents for p in self.partitions)

    @property
    def tokens(self) -> int:
        return sum(p.stats.tokens for p in self.partitions)

    @property
    def empty_documents(self) -> int:
        return sum(p.stats.empty_documents for p in self.partitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input.path.resolve()),
            "input_bytes": self.input.size,
            "output_bin": str(bin_path(self.output_prefix).resolve()),
            "output_idx": str(idx_path(self.output_prefix).resolve()),
            "reused": self.reused,
            "documents": self.documents,
            "tokens": self.tokens,
            "empty_documents": self.empty_documents,
            "merge_seconds": self.merge_seconds,
            "partitions": [p.to_dict() for p in self.partitions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, reused: bool) -> Self:
        output_bin = Path(data["output_bin"])
        return cls(
            input=InputFile(path=Path(data["input"]), size=data["input_bytes"]),
            output_prefix=output_bin.with_suffix(""),
            reused=reused,
            merge_seconds=data["merge_seconds"],
            partitions=[PartitionResult.from_dict(p) for p in data["partitions"]],
        )


@dataclass(slots=True)
class RunSummary:
    """Everything about a run: the config used, timing, totals and per-file results."""

    config: dict[str, Any]
    tokenizer: str
    vocab_size: int
    dtype: str
    started_at: datetime
    finished_at: datetime
    total_seconds: float
    files: list[FileResult] = field(default_factory=list)

    @property
    def documents(self) -> int:
        return sum(f.documents for f in self.files)

    @property
    def tokens(self) -> int:
        return sum(f.tokens for f in self.files)

    @property
    def empty_documents(self) -> int:
        return sum(f.empty_documents for f in self.files)

    @property
    def reused_files(self) -> int:
        return sum(f.reused for f in self.files)

    @property
    def num_partitions(self) -> int:
        return sum(len(f.partitions) for f in self.files)

    @property
    def reused_partitions(self) -> int:
        return sum(p.reused for f in self.files for p in f.partitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": {"name": "oellm-gigatokenize", "version": __version__},
            "config": {**self.config, "tokenizer": self.tokenizer, "vocab_size": self.vocab_size},
            "dtype": self.dtype,
            "timing": {
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat(),
                "total_seconds": self.total_seconds,
                "tokens_per_second": self.tokens / self.total_seconds if self.total_seconds > 0 else None,
            },
            "totals": {
                "input_files": len(self.files),
                "input_bytes": sum(f.input.size for f in self.files),
                "reused_files": self.reused_files,
                "partitions": self.num_partitions,
                "reused_partitions": self.reused_partitions,
                "documents": self.documents,
                "tokens": self.tokens,
                "empty_documents": self.empty_documents,
            },
            "files": [f.to_dict() for f in self.files],
        }

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())
