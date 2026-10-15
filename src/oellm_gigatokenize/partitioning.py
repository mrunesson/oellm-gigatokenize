"""Round-robin partitioning of an input file's documents."""

import itertools
from collections.abc import Iterable, Iterator


def round_robin[T](items: Iterable[T], partition: int, num_partitions: int) -> Iterator[T]:
    """The items of ``partition`` when items are dealt round robin over ``num_partitions``.

    Item ``i`` (counting from 0) goes to partition ``i % num_partitions``, so
    partition 0 gets items 0, n, 2n, ..., partition 1 gets items 1, n+1, ... and so on.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")
    if not 0 <= partition < num_partitions:
        raise ValueError(f"partition {partition} is outside [0, {num_partitions})")
    return itertools.islice(items, partition, None, num_partitions)
