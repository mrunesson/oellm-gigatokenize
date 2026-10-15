from pathlib import Path

import pytest
import zstandard

from oellm_gigatokenize.reader import InputFormatError, discover_files, iter_batches, iter_documents
from tests.conftest import write_jsonl_zst


class TestDiscoverFiles:
    def test_finds_matching_files_recursively_in_sorted_order(self, tmp_path: Path) -> None:
        for name in ["b.jsonl.zst", "a.jsonl.zst", "sub/c.jsonl.zst", "ignored.jsonl", "notes.txt"]:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_bytes(b"x" * len(name))

        files = discover_files(tmp_path)

        assert [f.path.relative_to(tmp_path).as_posix() for f in files] == [
            "a.jsonl.zst",
            "b.jsonl.zst",
            "sub/c.jsonl.zst",
        ]
        assert [f.size for f in files] == [11, 11, 15]

    def test_custom_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "a.zst").write_bytes(b"")
        (tmp_path / "b.jsonl.zst").write_bytes(b"")
        assert [f.path.name for f in discover_files(tmp_path, "*.zst")] == ["a.zst", "b.jsonl.zst"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            discover_files(tmp_path / "missing")


class TestIterDocuments:
    def test_yields_text_of_each_line(self, tmp_path: Path) -> None:
        path = write_jsonl_zst(tmp_path / "f.jsonl.zst", [{"text": "hello", "id": 1}, {"text": "wörld ✓"}])
        assert list(iter_documents(path)) == ["hello", "wörld ✓"]

    def test_custom_text_key(self, tmp_path: Path) -> None:
        path = write_jsonl_zst(tmp_path / "f.jsonl.zst", [{"content": "abc"}])
        assert list(iter_documents(path, text_key="content")) == ["abc"]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "f.jsonl.zst"
        path.write_bytes(zstandard.ZstdCompressor().compress(b'{"text": "a"}\n\n  \n{"text": "b"}'))
        assert list(iter_documents(path)) == ["a", "b"]

    def test_reads_multiple_concatenated_frames(self, tmp_path: Path) -> None:
        compressor = zstandard.ZstdCompressor()
        path = tmp_path / "f.jsonl.zst"
        path.write_bytes(compressor.compress(b'{"text": "a"}\n') + compressor.compress(b'{"text": "b"}\n'))
        assert list(iter_documents(path)) == ["a", "b"]

    def test_streams_large_files(self, tmp_path: Path) -> None:
        docs = [f"document {i} " * 50 for i in range(5000)]
        path = write_jsonl_zst(tmp_path / "f.jsonl.zst", ({"text": d} for d in docs))
        assert list(iter_documents(path)) == docs

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (b"{not json}\n", "invalid JSON"),
            (b'{"other": "x"}\n', "missing key 'text'"),
            (b'["text"]\n', "missing key 'text'"),
            (b'{"text": 42}\n', "is not a string"),
        ],
    )
    def test_invalid_lines_raise_with_location(self, tmp_path: Path, payload: bytes, message: str) -> None:
        path = tmp_path / "bad.jsonl.zst"
        path.write_bytes(zstandard.ZstdCompressor().compress(b'{"text": "ok"}\n' + payload))
        with pytest.raises(InputFormatError, match=rf"bad\.jsonl\.zst:2: .*{message}"):
            list(iter_documents(path))


class TestIterBatches:
    def test_batches_respect_budget(self) -> None:
        batches = list(iter_batches(["aa", "bb", "cc", "dd", "e"], max_batch_bytes=4))
        assert batches == [["aa", "bb"], ["cc", "dd"], ["e"]]

    def test_oversized_document_forms_its_own_batch(self) -> None:
        assert list(iter_batches(["a", "x" * 100, "b"], max_batch_bytes=10)) == [["a", "x" * 100], ["b"]]

    def test_empty_input(self) -> None:
        assert list(iter_batches([], max_batch_bytes=10)) == []

    def test_preserves_all_documents_in_order(self) -> None:
        docs = [str(i) * (i % 7) for i in range(100)]
        batches = list(iter_batches(docs, max_batch_bytes=13))
        assert [d for b in batches for d in b] == docs

    def test_rejects_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            list(iter_batches(["a"], max_batch_bytes=0))
