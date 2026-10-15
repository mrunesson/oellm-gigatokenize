"""The end-to-end tokenization pipeline.

Every input file becomes one output dataset. With ``num_partitions = n > 1`` a
file's documents are dealt round robin over n partitions (document ``i`` goes to
partition ``i % n``). Each partition is tokenized into its own dataset, and the
partitions are concatenated in partition order into the file's output.
Partitions and merges are tasks run by an executor: inline for one worker,
otherwise on a thread pool.

Finished partitions and files leave manifests describing their inputs, so a re-run
reuses whatever is still valid:

* ``<output_dir>/<name>.json`` next to each ``<name>.bin``/``.idx`` (a finished file),
* ``<work_dir>/<name>/part-NNNNN.json`` next to each partition (until merged).
"""

import logging
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oellm_gigatokenize.executor import make_executor
from oellm_gigatokenize.indexed_dataset import (
    DatasetWriter,
    bin_path,
    dataset_exists,
    idx_path,
    merge_datasets,
    optimal_dtype,
    remove_dataset,
)
from oellm_gigatokenize.reader import DEFAULT_PATTERN, InputFile, discover_files, iter_batches, iter_documents
from oellm_gigatokenize.results import (
    FileResult,
    PartitionResult,
    PartitionStats,
    RunSummary,
    load_if_matching,
    write_json,
)
from oellm_gigatokenize.tokenization import Encoder

logger = logging.getLogger(__name__)

DEFAULT_BATCH_BYTES = 256 * 1024 * 1024
SUMMARY_NAME = "_summary.json"


class NoInputFilesError(FileNotFoundError):
    """Raised when the input directory contains no matching files."""


class CancelledError(Exception):
    """Raised inside a task when the run is being aborted."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    num_partitions: int = 1
    workers: int = 1
    text_key: str = "text"
    pattern: str = DEFAULT_PATTERN
    batch_bytes: int = DEFAULT_BATCH_BYTES
    eod_id: int | None = None
    work_dir: Path | None = None
    keep_partitions: bool = False

    def __post_init__(self) -> None:
        if self.num_partitions < 1:
            raise ValueError("num_partitions must be at least 1")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

    @property
    def partition_dir(self) -> Path:
        return self.work_dir if self.work_dir is not None else self.output_dir / ".partitions"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / SUMMARY_NAME

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, with paths made absolute."""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value.resolve())
        values["work_dir"] = str(self.partition_dir.resolve())
        return values


def output_name(file: InputFile, input_dir: Path) -> Path:
    """The input's path relative to ``input_dir`` without ``.zst`` and ``.jsonl``/``.json``."""
    relative = file.path.relative_to(input_dir)
    name = relative.name.removesuffix(".zst")
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    return relative.with_name(name)


def _json_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + ".json")


@dataclass(frozen=True, slots=True)
class FileJob:
    """One input file and where everything derived from it goes."""

    input: InputFile
    name: Path
    config: PipelineConfig
    tokenizer: str
    dtype: str

    @property
    def output_prefix(self) -> Path:
        return self.config.output_dir / self.name

    @property
    def manifest_path(self) -> Path:
        return _json_path(self.output_prefix)

    @property
    def partition_dir(self) -> Path:
        return self.config.partition_dir / self.name

    def partition_prefix(self, number: int) -> Path:
        return self.partition_dir / f"part-{number:05d}"

    def inputs(self) -> dict[str, Any]:
        """Everything that determines the output's contents."""
        return {
            "file": str(self.input.path.resolve()),
            "size": self.input.size,
            "num_partitions": self.config.num_partitions,
            "tokenizer": self.tokenizer,
            "text_key": self.config.text_key,
            "eod_id": self.config.eod_id,
            "dtype": self.dtype,
        }

    def partition_inputs(self, number: int) -> dict[str, Any]:
        return {**self.inputs(), "partition": number}

    def load_finished(self) -> FileResult | None:
        if not dataset_exists(self.output_prefix):
            return None
        stored = load_if_matching(self.manifest_path, self.inputs())
        return None if stored is None else FileResult.from_dict(stored, reused=True)


def tokenize_partition(
    job: FileJob,
    number: int,
    encoder: Encoder,
    stop: threading.Event | None = None,
) -> PartitionResult:
    """Tokenize the documents of one round-robin partition into a partition dataset.

    Only one batch of text and its tokens is held in memory at a time. A partition
    that is already finished with the same inputs is reused instead.
    """
    config = job.config
    prefix = job.partition_prefix(number)
    inputs = job.partition_inputs(number)
    if dataset_exists(prefix) and (stored := load_if_matching(_json_path(prefix), inputs)) is not None:
        logger.info("Reusing partition %d of %s", number, job.name)
        return PartitionResult(number, reused=True, stats=PartitionStats(**stored))

    started = time.perf_counter()
    stats = PartitionStats()
    with DatasetWriter(prefix, optimal_dtype(encoder.vocab_size)) as writer:
        documents = iter_documents(job.input.path, config.text_key, number, config.num_partitions)
        for texts in iter_batches(documents, config.batch_bytes):
            if stop is not None and stop.is_set():
                raise CancelledError
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
    write_json(_json_path(prefix), {"inputs": inputs, "result": asdict(stats)})
    logger.info(
        "Tokenized partition %d of %s: %d documents, %d tokens in %.1fs",
        number,
        job.name,
        stats.documents,
        stats.tokens,
        stats.seconds,
    )
    return PartitionResult(number, reused=False, stats=stats)


def finalize_file(job: FileJob, partitions: list[PartitionResult]) -> FileResult:
    """Combine a file's partitions into its output dataset and record its manifest."""
    started = time.perf_counter()
    prefixes = [job.partition_prefix(p.number) for p in partitions]
    job.manifest_path.unlink(missing_ok=True)
    if len(prefixes) == 1 and not job.config.keep_partitions:
        _move_dataset(prefixes[0], job.output_prefix)
    else:
        merge_datasets(prefixes, job.output_prefix)
    result = FileResult(
        input=job.input,
        output_prefix=job.output_prefix,
        reused=False,
        merge_seconds=time.perf_counter() - started,
        partitions=partitions,
    )
    write_json(job.manifest_path, {"inputs": job.inputs(), "result": result.to_dict()})
    if not job.config.keep_partitions:
        _remove_partitions(job, prefixes)
    logger.info("Finished %s: %d documents, %d tokens", job.name, result.documents, result.tokens)
    return result


def _move_dataset(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    remove_dataset(target)
    shutil.move(bin_path(source), bin_path(target))
    shutil.move(idx_path(source), idx_path(target))


def _remove_partitions(job: FileJob, prefixes: list[Path]) -> None:
    for prefix in prefixes:
        remove_dataset(prefix)
        _json_path(prefix).unlink(missing_ok=True)
    directory = job.partition_dir
    while directory != job.config.partition_dir:
        try:
            directory.rmdir()
        except OSError:
            break
        directory = directory.parent


class _Scheduler:
    """Submits tasks and runs each task's continuation in the calling thread once it finishes.

    Continuations run in submission order, which keeps the inline executor's
    execution order deterministic.
    """

    def __init__(self, executor: Executor) -> None:
        self._executor = executor
        self._pending: dict[Future[Any], Callable[[Any], None]] = {}

    def submit[T](self, then: Callable[[T], None], fn: Callable[..., T], *args: object) -> None:
        self._pending[self._executor.submit(fn, *args)] = then

    def run(self) -> None:
        while self._pending:
            done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
            for future in [f for f in self._pending if f in done]:
                self._pending.pop(future)(future.result())


def _make_jobs(config: PipelineConfig, encoder: Encoder, files: list[InputFile]) -> list[FileJob]:
    dtype = str(optimal_dtype(encoder.vocab_size))
    jobs = [FileJob(f, output_name(f, config.input_dir), config, encoder.fingerprint, dtype) for f in files]
    seen: dict[Path, InputFile] = {}
    for job in jobs:
        if job.name in seen:
            raise ValueError(f"{seen[job.name].path} and {job.input.path} would both be written to {job.name}")
        if job.manifest_path == config.summary_path:
            raise ValueError(f"{job.input.path} would overwrite the run summary {SUMMARY_NAME}")
        seen[job.name] = job.input
    return jobs


def run(config: PipelineConfig, encoder: Encoder) -> RunSummary:
    """Tokenize every input file, write the run summary to ``<output_dir>/_summary.json`` and return it."""
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    if config.eod_id is not None and not 0 <= config.eod_id < encoder.vocab_size:
        raise ValueError(f"eod id {config.eod_id} is outside the vocabulary (size {encoder.vocab_size})")

    files = discover_files(config.input_dir, config.pattern)
    if not files:
        raise NoInputFilesError(f"no files matching {config.pattern!r} in {config.input_dir}")
    jobs = _make_jobs(config, encoder, files)
    logger.info(
        "Tokenizing %d files (%.2f GiB compressed) into %s, %d partitions per file, %d workers, with %s",
        len(files),
        sum(f.size for f in files) / 2**30,
        config.output_dir,
        config.num_partitions,
        config.workers,
        encoder.fingerprint,
    )

    results: dict[int, FileResult] = {}
    todo: list[int] = []
    for index, job in enumerate(jobs):
        if (finished := job.load_finished()) is not None:
            logger.info("Reusing finished output for %s", job.name)
            results[index] = finished
        else:
            todo.append(index)

    stop = threading.Event()
    executor = make_executor(config.workers)
    scheduler = _Scheduler(executor)
    next_file = 0
    outstanding = 0  # partitions submitted whose results have not been handled yet

    def start_file(index: int) -> None:
        nonlocal outstanding
        partitions: dict[int, PartitionResult] = {}

        def on_partition(result: PartitionResult) -> None:
            nonlocal outstanding
            outstanding -= 1
            partitions[result.number] = result
            if len(partitions) == config.num_partitions:
                ordered = [partitions[n] for n in sorted(partitions)]
                scheduler.submit(lambda r: results.__setitem__(index, r), finalize_file, jobs[index], ordered)
            fill()

        outstanding += config.num_partitions
        for number in range(config.num_partitions):
            scheduler.submit(on_partition, tokenize_partition, jobs[index], number, encoder, stop)

    def fill() -> None:
        # Start files only while fewer partitions than workers are outstanding, so
        # every worker has work but partitions of few files sit on disk at a time.
        nonlocal next_file
        while next_file < len(todo) and outstanding < config.workers:
            start_file(todo[next_file])
            next_file += 1

    try:
        fill()
        scheduler.run()
    except BaseException:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    _remove_empty_dir(config.partition_dir)

    summary = RunSummary(
        config=config.to_dict(),
        tokenizer=encoder.name,
        vocab_size=encoder.vocab_size,
        dtype=str(optimal_dtype(encoder.vocab_size)),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        total_seconds=time.perf_counter() - started,
        files=[results[i] for i in range(len(jobs))],
    )
    summary.write(config.summary_path)
    return summary


def _remove_empty_dir(directory: Path) -> None:
    try:
        directory.rmdir()
    except OSError:
        logger.debug("Partition directory %s not removed (missing or not empty)", directory)
