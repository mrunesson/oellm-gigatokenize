"""The end-to-end tokenization pipeline: partition, tokenize each partition, merge."""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from oellm_gigatokenize.indexed_dataset import (
    DatasetWriter,
    dataset_exists,
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


def manifest_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".json")


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Everything that determines a partition's contents.

    Stored next to each finished partition so that a re-run can reuse partitions that
    are still valid and redo those whose inputs or settings changed.
    """

    files: list[tuple[str, int]]
    tokenizer: str
    text_key: str
    eod_id: int | None
    dtype: str

    def save(self, prefix: Path) -> None:
        manifest_path(prefix).write_text(json.dumps(asdict(self), indent=2))

    def matches(self, prefix: Path) -> bool:
        path = manifest_path(prefix)
        if not (path.is_file() and dataset_exists(prefix)):
            return False
        try:
            stored = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        return stored == json.loads(json.dumps(asdict(self)))


@dataclass(slots=True)
class PartitionStats:
    documents: int = 0
    tokens: int = 0
    empty_documents: int = 0


@dataclass(slots=True)
class PipelineSummary:
    output_prefix: Path
    num_partitions: int
    reused_partitions: int
    documents: int
    tokens: int
    empty_documents: int
    partition_stats: list[PartitionStats] = field(default_factory=list)


def tokenize_partition(
    files: Sequence[InputFile],
    prefix: Path,
    encoder: Encoder,
    config: PipelineConfig,
) -> PartitionStats:
    """Tokenize ``files`` into a dataset at ``prefix``, streaming batch by batch.

    Only one batch of text and its tokens is held in memory at a time.
    """
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
    return stats


def run(config: PipelineConfig, encoder: Encoder) -> PipelineSummary:
    """Run the full pipeline and return a summary of what was produced."""
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
    all_stats: list[PartitionStats] = []
    reused = 0
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
        if manifest.matches(prefix):
            logger.info("Reusing existing %s at %s", label, prefix)
            reused += 1
            continue

        started = time.perf_counter()
        stats = tokenize_partition(partition_files, prefix, encoder, config)
        manifest.save(prefix)
        all_stats.append(stats)
        elapsed = time.perf_counter() - started
        logger.info(
            "Finished %s: %d files, %d documents, %d tokens in %.1fs (%.0f tokens/s)",
            label,
            len(partition_files),
            stats.documents,
            stats.tokens,
            elapsed,
            stats.tokens / elapsed if elapsed > 0 else 0.0,
        )

    logger.info("Merging %d partitions into %s", len(prefixes), config.output_prefix)
    merged = merge_datasets(prefixes, config.output_prefix)

    if not config.keep_partitions:
        _remove_partitions(prefixes, partition_dir)

    return PipelineSummary(
        output_prefix=config.output_prefix,
        num_partitions=len(partitions),
        reused_partitions=reused,
        documents=merged.num_sequences,
        tokens=merged.num_tokens,
        empty_documents=sum(s.empty_documents for s in all_stats),
        partition_stats=all_stats,
    )


def _remove_partitions(prefixes: Sequence[Path], partition_dir: Path) -> None:
    for prefix in prefixes:
        remove_dataset(prefix)
        manifest_path(prefix).unlink(missing_ok=True)
    try:
        partition_dir.rmdir()
    except OSError:
        logger.debug("Partition directory %s not removed (not empty)", partition_dir)
