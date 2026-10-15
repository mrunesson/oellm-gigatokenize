"""Splitting the input files into partitions that are tokenized one at a time."""

import logging
from collections.abc import Sequence

from oellm_gigatokenize.reader import InputFile

logger = logging.getLogger(__name__)


def split_into_partitions(files: Sequence[InputFile], num_partitions: int) -> list[list[InputFile]]:
    """Split ``files`` into contiguous, non-empty partitions of roughly equal total size.

    File order is preserved, so concatenating the partitions reproduces the input
    order. Files are never split, which means the number of partitions is capped at
    the number of files.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")
    if not files:
        return []
    if num_partitions > len(files):
        logger.warning(
            "Requested %d partitions but only %d input files exist; using %d partitions",
            num_partitions,
            len(files),
            len(files),
        )
        num_partitions = len(files)

    total = sum(f.size for f in files)
    partitions: list[list[InputFile]] = []
    current: list[InputFile] = []
    accumulated = 0
    for index, file in enumerate(files):
        current.append(file)
        accumulated += file.size
        remaining_files = len(files) - index - 1
        remaining_partitions = num_partitions - len(partitions) - 1
        if remaining_partitions == 0:
            continue
        reached_target = accumulated >= total * (len(partitions) + 1) / num_partitions
        if reached_target or remaining_files == remaining_partitions:
            partitions.append(current)
            current = []
    partitions.append(current)
    return partitions
