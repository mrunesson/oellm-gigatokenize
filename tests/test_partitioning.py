import pytest

from oellm_gigatokenize.partitioning import round_robin


def test_single_partition_gets_everything() -> None:
    assert list(round_robin(range(5), 0, 1)) == [0, 1, 2, 3, 4]


def test_deals_items_round_robin() -> None:
    items = ["d0", "d1", "d2", "d3", "d4", "d5", "d6"]
    assert [list(round_robin(items, k, 3)) for k in range(3)] == [
        ["d0", "d3", "d6"],
        ["d1", "d4"],
        ["d2", "d5"],
    ]


def test_more_partitions_than_items() -> None:
    assert [list(round_robin([1, 2], k, 4)) for k in range(4)] == [[1], [2], [], []]


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 7, 25])
def test_partitions_cover_every_item_once(num_partitions: int) -> None:
    items = list(range(23))
    parts = [list(round_robin(items, k, num_partitions)) for k in range(num_partitions)]
    assert sorted(i for part in parts for i in part) == items
    assert all(i % num_partitions == k for k, part in enumerate(parts) for i in part)


def test_is_lazy() -> None:
    def endless():
        n = 0
        while True:
            yield n
            n += 1

    iterator = round_robin(endless(), 1, 2)
    assert [next(iterator) for _ in range(3)] == [1, 3, 5]


@pytest.mark.parametrize(("partition", "num_partitions"), [(0, 0), (-1, 2), (2, 2)])
def test_rejects_invalid_arguments(partition: int, num_partitions: int) -> None:
    with pytest.raises(ValueError, match=r"num_partitions|outside"):
        round_robin([], partition, num_partitions)
