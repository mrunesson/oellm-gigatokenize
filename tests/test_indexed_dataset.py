import struct
from pathlib import Path

import numpy as np
import pytest

from oellm_gigatokenize.indexed_dataset import (
    DatasetIndex,
    DatasetWriter,
    IndexFormatError,
    bin_path,
    dataset_exists,
    idx_path,
    merge_datasets,
    optimal_dtype,
    read_documents,
)


def write_dataset(prefix: Path, documents: list[list[int]], dtype: type = np.uint16) -> None:
    with DatasetWriter(prefix, np.dtype(dtype)) as writer:
        tokens = np.array([t for doc in documents for t in doc], dtype=np.int64)
        writer.add_documents(tokens, np.array([len(d) for d in documents]))
        writer.finalize()


def as_lists(prefix: Path) -> list[list[int]]:
    return [doc.tolist() for doc in read_documents(prefix)]


def parse_megatron_index(path: Path) -> dict:
    """Independent parser following megatron.core.datasets.indexed_dataset._IndexReader."""
    with path.open("rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00"
        (version,) = struct.unpack("<Q", f.read(8))
        (dtype_code,) = struct.unpack("<B", f.read(1))
        (sequence_count,) = struct.unpack("<Q", f.read(8))
        (document_count,) = struct.unpack("<Q", f.read(8))
        offset = f.tell()
    buffer = path.read_bytes()
    lengths = np.frombuffer(buffer, np.int32, sequence_count, offset)
    pointers = np.frombuffer(buffer, np.int64, sequence_count, offset + lengths.nbytes)
    documents = np.frombuffer(buffer, np.int64, document_count, offset + lengths.nbytes + pointers.nbytes)
    assert offset + lengths.nbytes + pointers.nbytes + documents.nbytes == len(buffer)
    return {
        "version": version,
        "dtype_code": dtype_code,
        "lengths": lengths.tolist(),
        "pointers": pointers.tolist(),
        "documents": documents.tolist(),
    }


@pytest.mark.parametrize(
    ("vocab_size", "expected"),
    [(256, np.uint16), (50257, np.uint16), (65499, np.uint16), (65500, np.int32), (151669, np.int32)],
)
def test_optimal_dtype_matches_megatron(vocab_size: int, expected: type) -> None:
    assert optimal_dtype(vocab_size) == np.dtype(expected)


def test_roundtrip(tmp_path: Path) -> None:
    documents = [[1, 2, 3], [4], [5, 6, 7, 8, 65535]]
    write_dataset(tmp_path / "ds", documents)
    assert as_lists(tmp_path / "ds") == documents


def test_multiple_add_calls_append(tmp_path: Path) -> None:
    prefix = tmp_path / "ds"
    with DatasetWriter(prefix, np.dtype(np.int32)) as writer:
        writer.add_documents(np.array([1, 2, 3]), np.array([2, 1]))
        writer.add_documents(np.array([100_000]), np.array([1]))
        writer.finalize()
    assert as_lists(prefix) == [[1, 2], [3], [100_000]]


@pytest.mark.parametrize(("dtype", "code", "itemsize"), [(np.uint16, 8, 2), (np.int32, 4, 4)])
def test_file_layout_matches_megatron_spec(tmp_path: Path, dtype: type, code: int, itemsize: int) -> None:
    prefix = tmp_path / "ds"
    write_dataset(prefix, [[1, 2, 3], [4], [5, 6]], dtype)

    parsed = parse_megatron_index(idx_path(prefix))

    assert parsed == {
        "version": 1,
        "dtype_code": code,
        "lengths": [3, 1, 2],
        "pointers": [0, 3 * itemsize, 4 * itemsize],
        "documents": [0, 1, 2, 3],
    }
    assert bin_path(prefix).read_bytes() == np.array([1, 2, 3, 4, 5, 6], dtype=f"<{np.dtype(dtype).char}").tobytes()


def test_empty_dataset(tmp_path: Path) -> None:
    write_dataset(tmp_path / "ds", [])
    index = DatasetIndex.read(idx_path(tmp_path / "ds"))
    assert index.num_sequences == 0
    assert index.document_indices.tolist() == [0]
    assert as_lists(tmp_path / "ds") == []


def test_empty_documents_have_no_sequence(tmp_path: Path) -> None:
    prefix = tmp_path / "ds"
    write_dataset(prefix, [[], [1, 2], [], [], [3], []])

    parsed = parse_megatron_index(idx_path(prefix))

    assert parsed["lengths"] == [2, 1]
    assert parsed["pointers"] == [0, 4]
    assert parsed["documents"] == [0, 0, 1, 1, 1, 2, 2]
    assert as_lists(prefix) == [[1, 2], [3]]


def test_unfinalized_writer_leaves_no_dataset(tmp_path: Path) -> None:
    prefix = tmp_path / "ds"
    writer = DatasetWriter(prefix, np.dtype(np.uint16))
    writer.add_documents(np.array([1]), np.array([1]))
    assert not dataset_exists(prefix)
    writer.abort()
    assert list(tmp_path.iterdir()) == []


def test_exception_inside_context_removes_temporary_files(tmp_path: Path) -> None:
    def write_then_fail() -> None:
        with DatasetWriter(tmp_path / "ds", np.dtype(np.uint16)) as writer:
            writer.add_documents(np.array([1]), np.array([1]))
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_then_fail()
    assert list(tmp_path.iterdir()) == []


def test_rejects_tokens_that_do_not_fit_dtype(tmp_path: Path) -> None:
    with DatasetWriter(tmp_path / "ds", np.dtype(np.uint16)) as writer, pytest.raises(ValueError, match="does not fit"):
        writer.add_documents(np.array([70_000]), np.array([1]))


def test_rejects_inconsistent_lengths(tmp_path: Path) -> None:
    with DatasetWriter(tmp_path / "ds", np.dtype(np.uint16)) as writer, pytest.raises(ValueError, match="lengths"):
        writer.add_documents(np.array([1, 2]), np.array([3]))


def test_rejects_unsupported_dtype(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        DatasetWriter(tmp_path / "ds", np.dtype(np.uint32))


class TestMerge:
    def test_concatenates_in_order(self, tmp_path: Path) -> None:
        parts = [[[1, 2], [3]], [], [[4, 5, 6]], [[7], [8], [9]]]
        prefixes = [tmp_path / f"part{i}" for i in range(len(parts))]
        for prefix, docs in zip(prefixes, parts, strict=True):
            write_dataset(prefix, docs)

        index = merge_datasets(prefixes, tmp_path / "out" / "merged")

        assert as_lists(tmp_path / "out" / "merged") == [doc for docs in parts for doc in docs]
        assert index.num_tokens == 9
        parsed = parse_megatron_index(idx_path(tmp_path / "out" / "merged"))
        assert parsed["documents"] == [0, 1, 2, 3, 4, 5, 6]
        assert parsed["pointers"] == [0, 4, 6, 12, 14, 16]

    def test_matches_writing_everything_at_once(self, tmp_path: Path) -> None:
        documents = [[i, i + 1] * (i % 4 + 1) for i in range(20)]
        write_dataset(tmp_path / "direct", documents)
        write_dataset(tmp_path / "a", documents[:7])
        write_dataset(tmp_path / "b", documents[7:])

        merge_datasets([tmp_path / "a", tmp_path / "b"], tmp_path / "merged")

        for suffix in (".bin", ".idx"):
            assert (tmp_path / f"merged{suffix}").read_bytes() == (tmp_path / f"direct{suffix}").read_bytes()

    def test_matches_writing_everything_at_once_with_empty_documents(self, tmp_path: Path) -> None:
        documents = [[1], [], [2, 3], [], [], [4], []]
        write_dataset(tmp_path / "direct", documents)
        write_dataset(tmp_path / "a", documents[:2])
        write_dataset(tmp_path / "b", documents[2:3])
        write_dataset(tmp_path / "c", documents[3:5])
        write_dataset(tmp_path / "d", documents[5:])

        merge_datasets([tmp_path / name for name in "abcd"], tmp_path / "merged")

        for suffix in (".bin", ".idx"):
            assert (tmp_path / f"merged{suffix}").read_bytes() == (tmp_path / f"direct{suffix}").read_bytes()

    def test_preserves_multi_sequence_documents(self, tmp_path: Path) -> None:
        index = DatasetIndex(np.dtype(np.uint16), np.array([1, 1, 1], np.int32), np.array([0, 2, 3], np.int64))
        for name in ("a", "b"):
            index.write(idx_path(tmp_path / name))
            bin_path(tmp_path / name).write_bytes(np.array([1, 2, 3], np.uint16).tobytes())

        merged = merge_datasets([tmp_path / "a", tmp_path / "b"], tmp_path / "m")

        assert merged.document_indices.tolist() == [0, 2, 3, 5, 6]

    def test_rejects_mixed_dtypes(self, tmp_path: Path) -> None:
        write_dataset(tmp_path / "a", [[1]], np.uint16)
        write_dataset(tmp_path / "b", [[1]], np.int32)
        with pytest.raises(ValueError, match="different dtypes"):
            merge_datasets([tmp_path / "a", tmp_path / "b"], tmp_path / "m")
        assert not dataset_exists(tmp_path / "m")

    def test_rejects_empty_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no datasets"):
            merge_datasets([], tmp_path / "m")


class TestReadIndex:
    def test_rejects_bad_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "x.idx"
        path.write_bytes(b"NOTANIDX" + bytes(40))
        with pytest.raises(IndexFormatError, match="not a Megatron index"):
            DatasetIndex.read(path)

    def test_rejects_truncated_file(self, tmp_path: Path) -> None:
        write_dataset(tmp_path / "ds", [[1, 2], [3]])
        path = idx_path(tmp_path / "ds")
        path.write_bytes(path.read_bytes()[:-3])
        with pytest.raises(IndexFormatError, match="expected"):
            DatasetIndex.read(path)

    def test_rejects_unknown_version(self, tmp_path: Path) -> None:
        write_dataset(tmp_path / "ds", [[1]])
        path = idx_path(tmp_path / "ds")
        data = bytearray(path.read_bytes())
        data[9:17] = struct.pack("<Q", 2)
        path.write_bytes(bytes(data))
        with pytest.raises(IndexFormatError, match="version"):
            DatasetIndex.read(path)


def test_compatible_with_megatron_reader(tmp_path: Path) -> None:
    """Cross-check against Megatron itself when megatron-core is installed."""
    indexed_dataset = pytest.importorskip("megatron.core.datasets.indexed_dataset")
    documents = [[1, 2, 3], [4], [5, 6]]
    write_dataset(tmp_path / "ds", documents)
    dataset = indexed_dataset.IndexedDataset(str(tmp_path / "ds"))
    assert [dataset[i].tolist() for i in range(len(dataset))] == documents


def test_byte_identical_to_megatron_builder(tmp_path: Path) -> None:
    """Write the same documents, including empty ones, with Megatron's builder when it is installed."""
    indexed_dataset = pytest.importorskip("megatron.core.datasets.indexed_dataset")
    torch = pytest.importorskip("torch")
    documents = [[1, 2, 3], [], [4], [], [5, 6], []]
    write_dataset(tmp_path / "ours", documents)

    builder = indexed_dataset.IndexedDatasetBuilder(str(bin_path(tmp_path / "theirs")), dtype=np.uint16)
    for doc in documents:
        # preprocess_data.py adds every document, with no sequence lengths when it has no tokens.
        builder.add_document(torch.tensor(doc, dtype=torch.int64), [len(doc)] if doc else [])
    builder.finalize(str(idx_path(tmp_path / "theirs")))

    for suffix in (".bin", ".idx"):
        assert (tmp_path / f"ours{suffix}").read_bytes() == (tmp_path / f"theirs{suffix}").read_bytes()
