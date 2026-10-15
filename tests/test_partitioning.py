from pathlib import Path

import pytest

from oellm_gigatokenize.partitioning import split_into_partitions
from oellm_gigatokenize.reader import InputFile


def make_files(*sizes: int) -> list[InputFile]:
    return [InputFile(path=Path(f"f{i}"), size=size) for i, size in enumerate(sizes)]


def sizes(partitions: list[list[InputFile]]) -> list[list[int]]:
    return [[f.size for f in p] for p in partitions]


def test_single_partition_contains_everything() -> None:
    files = make_files(1, 2, 3)
    assert split_into_partitions(files, 1) == [files]


def test_equal_sizes_split_evenly() -> None:
    assert sizes(split_into_partitions(make_files(*[10] * 6), 3)) == [[10, 10], [10, 10], [10, 10]]


def test_balances_by_size() -> None:
    assert sizes(split_into_partitions(make_files(50, 10, 10, 10, 10, 10), 2)) == [[50], [10, 10, 10, 10, 10]]


def test_caps_partitions_at_number_of_files(caplog: pytest.LogCaptureFixture) -> None:
    partitions = split_into_partitions(make_files(1, 2), 5)
    assert sizes(partitions) == [[1], [2]]
    assert "only 2 input files" in caplog.text


def test_never_produces_empty_partitions_even_with_skewed_sizes() -> None:
    partitions = split_into_partitions(make_files(1000, 1, 1, 1), 4)
    assert sizes(partitions) == [[1000], [1], [1], [1]]


def test_zero_sized_files() -> None:
    assert sizes(split_into_partitions(make_files(0, 0, 0, 0), 2)) == [[0], [0, 0, 0]]


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 7, 10, 25])
def test_partitions_are_contiguous_and_complete(num_partitions: int) -> None:
    files = make_files(*[(i * 37) % 11 + 1 for i in range(10)])
    partitions = split_into_partitions(files, num_partitions)
    assert len(partitions) == min(num_partitions, len(files))
    assert all(partitions)
    assert [f for p in partitions for f in p] == files


def test_no_files() -> None:
    assert split_into_partitions([], 3) == []


@pytest.mark.parametrize("num_partitions", [0, -1])
def test_rejects_invalid_partition_count(num_partitions: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        split_into_partitions(make_files(1), num_partitions)
