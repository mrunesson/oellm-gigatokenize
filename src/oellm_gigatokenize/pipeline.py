"""The end-to-end tokenization pipeline: partition, tokenize each partition, merge."""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oellm_gigatokenize import __version__
from oellm_gigatokenize.indexed_dataset import (
    DatasetWriter,
    bin_path,
    dataset_exists,
    idx_path,
    merge_datasets,
    optimal_dtype,
    remove_dataset,
)
from oellm_gigatokenize.partitioning import split_into_partitions
from oellm_gigatokenize.reader import DEFAULT_PATTERN, InputFile, discover_files, iter_batches, iter_documents
from oellm_gigatokenize.tokenization import Encoder

logger = logging.getLogger(__name__)

DEFAULT_BATCH_BYTES = 256 * 1024 * 1024


class NoInputFilesError(FileNotFoundError):
    """Raised when the input directory contains no matching files."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    input_dir: Path
    output_prefix: Path
    num_partitions: int
    text_key: str = "text"
    pattern: str = DEFAULT_PATTERN
    batch_bytes: int = DEFAULT_BATCH_BYTES
    eod_id: int | None = None
    work_dir: Path | None = None
    keep_partitions: bool = False

    @property
    def partition_dir(self) -> Path:
        if self.work_dir is not None:
            return self.work_dir
        return self.output_prefix.with_name(self.output_prefix.name + ".partitions")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, with paths made absolute."""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value.resolve())
        values["work_dir"] = str(self.partition_dir.resolve())
        return values


def manifest_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".json")


def summary_path(output_prefix: Path) -> Path:
    """Where the run summary is written: ``<output_prefix>.json``."""
    return output_prefix.with_name(output_prefix.name + ".json")


@dataclass(slots=True)
class PartitionStats:
    documents: int = 0
    tokens: int = 0
    empty_documents: int = 0
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Everything that determines a partition's contents.

    Stored next to each finished partition, together with the partition's stats, so
    that a re-run can reuse partitions that are still valid (and still report their
    stats) and redo those whose inputs or settings changed.
    """

    files: list[tuple[str, int]]
    tokenizer: str
    text_key: str
    eod_id: int | None
    dtype: str

    def save(self, prefix: Path, stats: PartitionStats) -> None:
        content = {"inputs": asdict(self), "stats": asdict(stats)}
        manifest_path(prefix).write_text(json.dumps(content, indent=2))

    def load_matching_stats(self, prefix: Path) -> PartitionStats | None:
        """The stored stats if a finished partition built from these inputs exists at ``prefix``."""
        path = manifest_path(prefix)
        if not (path.is_file() and dataset_exists(prefix)):
            return None
        try:
            stored = json.loads(path.read_text())
            if stored["inputs"] != json.loads(json.dumps(asdict(self))):
                return None
            return PartitionStats(**stored["stats"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


@dataclass(slots=True)
class PartitionResult:
    number: int
    files: list[InputFile]
    reused: bool
    stats: PartitionStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "reused": self.reused,
            "input_files": [str(f.path.resolve()) for f in self.files],
            "input_bytes": sum(f.size for f in self.files),
            **asdict(self.stats),
        }


@dataclass(slots=True)
class PipelineSummary:
    config: PipelineConfig
    tokenizer: str
    vocab_size: int
    dtype: str
    started_at: datetime
    finished_at: datetime
    total_seconds: float
    merge_seconds: float
    documents: int
    tokens: int
    partitions: list[PartitionResult] = field(default_factory=list)

    @property
    def output_prefix(self) -> Path:
        return self.config.output_prefix

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    @property
    def reused_partitions(self) -> int:
        return sum(p.reused for p in self.partitions)

    @property
    def empty_documents(self) -> int:
        return sum(p.stats.empty_documents for p in self.partitions)

    def to_dict(self) -> dict[str, Any]:
        input_files = sum(len(p.files) for p in self.partitions)
        input_bytes = sum(f.size for p in self.partitions for f in p.files)
        return {
            "tool": {"name": "oellm-gigatokenize", "version": __version__},
            "output": {
                "bin": str(bin_path(self.output_prefix).resolve()),
                "idx": str(idx_path(self.output_prefix).resolve()),
                "dtype": self.dtype,
            },
            "config": {
                **self.config.to_dict(),
                "tokenizer": self.tokenizer,
                "vocab_size": self.vocab_size,
            },
            "timing": {
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat(),
                "total_seconds": self.total_seconds,
                "merge_seconds": self.merge_seconds,
                "tokens_per_second": self.tokens / self.total_seconds if self.total_seconds > 0 else None,
            },
            "totals": {
                "input_files": input_files,
                "input_bytes": input_bytes,
                "partitions": self.num_partitions,
                "reused_partitions": self.reused_partitions,
                "documents": self.documents,
                "tokens": self.tokens,
                "empty_documents": self.empty_documents,
            },
            "partitions": [p.to_dict() for p in self.partitions],
        }

    def write(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        tmp.replace(path)


def tokenize_partition(
    files: Sequence[InputFile],
    prefix: Path,
    encoder: Encoder,
    config: PipelineConfig,
) -> PartitionStats:
    """Tokenize ``files`` into a dataset at ``prefix``, streaming batch by batch.

    Only one batch of text and its tokens is held in memory at a time.
    """
    started = time.perf_counter()
    stats = PartitionStats()
    dtype = optimal_dtype(encoder.vocab_size)
    with DatasetWriter(prefix, dtype) as writer:
        for file in files:
            documents = iter_documents(file.path, config.text_key)
            for texts in iter_batches(documents, config.batch_bytes):
                batch = encoder.encode_batch(texts)
                non_empty = batch.drop_empty()
                stats.empty_documents += batch.num_documents - non_empty.num_documents
                if config.eod_id is not None:
                    non_empty = non_empty.with_eod(config.eod_id)
                writer.add_documents(non_empty.tokens, non_empty.lengths)
                stats.documents += non_empty.num_documents
                stats.tokens += len(non_empty.tokens)
        writer.finalize()
    stats.seconds = time.perf_counter() - started
    return stats


def run(config: PipelineConfig, encoder: Encoder) -> PipelineSummary:
    """Run the full pipeline, write the summary to ``<output_prefix>.json`` and return it."""
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    if config.eod_id is not None and not 0 <= config.eod_id < encoder.vocab_size:
        raise ValueError(f"eod id {config.eod_id} is outside the vocabulary (size {encoder.vocab_size})")

    files = discover_files(config.input_dir, config.pattern)
    if not files:
        raise NoInputFilesError(f"no files matching {config.pattern!r} in {config.input_dir}")
    partitions = split_into_partitions(files, config.num_partitions)
    dtype = optimal_dtype(encoder.vocab_size)
    partition_dir = config.partition_dir
    partition_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Tokenizing %d files (%.2f GiB compressed) in %d partitions with %s, token dtype %s",
        len(files),
        sum(f.size for f in files) / 2**30,
        len(partitions),
        encoder.fingerprint,
        dtype,
    )

    prefixes: list[Path] = []
    results: list[PartitionResult] = []
    for number, partition_files in enumerate(partitions):
        prefix = partition_dir / f"part-{number:05d}"
        prefixes.append(prefix)
        manifest = PartitionManifest(
            files=[(str(f.path.resolve()), f.size) for f in partition_files],
            tokenizer=encoder.fingerprint,
            text_key=config.text_key,
            eod_id=config.eod_id,
            dtype=str(dtype),
        )
        label = f"partition {number + 1}/{len(partitions)}"
        stored_stats = manifest.load_matching_stats(prefix)
        if stored_stats is not None:
            logger.info("Reusing existing %s at %s", label, prefix)
            results.append(PartitionResult(number, partition_files, reused=True, stats=stored_stats))
            continue

        stats = tokenize_partition(partition_files, prefix, encoder, config)
        manifest.save(prefix, stats)
        results.append(PartitionResult(number, partition_files, reused=False, stats=stats))
        logger.info(
            "Finished %s: %d files, %d documents, %d tokens in %.1fs (%.0f tokens/s)",
            label,
            len(partition_files),
            stats.documents,
            stats.tokens,
            stats.seconds,
            stats.tokens / stats.seconds if stats.seconds > 0 else 0.0,
        )

    logger.info("Merging %d partitions into %s", len(prefixes), config.output_prefix)
    merge_started = time.perf_counter()
    merged = merge_datasets(prefixes, config.output_prefix)
    merge_seconds = time.perf_counter() - merge_started

    if not config.keep_partitions:
        _remove_partitions(prefixes, partition_dir)

    summary = PipelineSummary(
        config=config,
        tokenizer=encoder.name,
        vocab_size=encoder.vocab_size,
        dtype=str(dtype),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        total_seconds=time.perf_counter() - started,
        merge_seconds=merge_seconds,
        documents=merged.num_sequences,
        tokens=merged.num_tokens,
        partitions=results,
    )
    summary.write(summary_path(config.output_prefix))
    return summary


def _remove_partitions(prefixes: Sequence[Path], partition_dir: Path) -> None:
    for prefix in prefixes:
        remove_dataset(prefix)
        manifest_path(prefix).unlink(missing_ok=True)
    try:
        partition_dir.rmdir()
    except OSError:
        logger.debug("Partition directory %s not removed (not empty)", partition_dir)
