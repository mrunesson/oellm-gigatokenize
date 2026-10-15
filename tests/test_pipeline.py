import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from oellm_gigatokenize.indexed_dataset import DatasetIndex, dataset_exists, idx_path, read_documents
from oellm_gigatokenize.pipeline import NoInputFilesError, PipelineConfig, run, summary_path
from tests.conftest import ByteEncoder, write_corpus

CORPUS = [
    ["first document", "second", "ünïcödé ✓"],
    ["fourth"],
    ["fifth document is a bit longer " * 5, "sixth"],
    ["seventh", "eighth", "ninth", "tenth"],
]
DOCUMENTS = [d for docs in CORPUS for d in docs]


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "input"
    write_corpus(directory, CORPUS)
    return directory


def make_config(tmp_path: Path, input_dir: Path, **overrides: object) -> PipelineConfig:
    settings: dict = {"input_dir": input_dir, "output_prefix": tmp_path / "out" / "data", "num_partitions": 2}
    settings.update(overrides)
    return PipelineConfig(**settings)


def decoded(prefix: Path, encoder: ByteEncoder, eod: int | None = None) -> list[str]:
    documents = []
    for tokens in read_documents(prefix):
        ids = tokens.tolist()
        if eod is not None:
            assert ids[-1] == eod
            ids = ids[:-1]
        documents.append(bytes(t - 1 for t in ids).decode())
    return documents


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 4, 10])
def test_output_contains_all_documents_in_order(
    tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder, num_partitions: int
) -> None:
    config = make_config(tmp_path, input_dir, num_partitions=num_partitions)

    summary = run(config, byte_encoder)

    assert decoded(config.output_prefix, byte_encoder) == DOCUMENTS
    assert summary.documents == len(DOCUMENTS)
    assert summary.tokens == sum(len(d.encode()) for d in DOCUMENTS)
    assert summary.num_partitions == min(num_partitions, len(CORPUS))


def test_output_is_independent_of_partitioning_and_batching(
    tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder
) -> None:
    reference = make_config(tmp_path, input_dir, output_prefix=tmp_path / "a", num_partitions=1)
    other = make_config(tmp_path, input_dir, output_prefix=tmp_path / "b", num_partitions=3, batch_bytes=7)
    run(reference, byte_encoder)
    run(other, byte_encoder)
    for suffix in (".bin", ".idx"):
        assert (tmp_path / f"a{suffix}").read_bytes() == (tmp_path / f"b{suffix}").read_bytes()


def test_small_batches_stream_through_encoder(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    run(make_config(tmp_path, input_dir, batch_bytes=1), byte_encoder)
    assert byte_encoder.calls == len(DOCUMENTS)


def test_appends_eod(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    config = make_config(tmp_path, input_dir, eod_id=0)
    summary = run(config, byte_encoder)
    assert decoded(config.output_prefix, byte_encoder, eod=0) == DOCUMENTS
    assert summary.tokens == sum(len(d.encode()) + 1 for d in DOCUMENTS)


def test_rejects_eod_outside_vocabulary(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    with pytest.raises(ValueError, match="outside the vocabulary"):
        run(make_config(tmp_path, input_dir, eod_id=byte_encoder.vocab_size), byte_encoder)


def test_skips_empty_documents(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_corpus(input_dir, [["a", "", "b"], [""]])
    config = make_config(tmp_path, input_dir)
    summary = run(config, byte_encoder)
    assert decoded(config.output_prefix, byte_encoder) == ["a", "b"]
    assert summary.empty_documents == 2


def test_uses_int32_for_large_vocabularies(tmp_path: Path, input_dir: Path) -> None:
    config = make_config(tmp_path, input_dir)
    run(config, ByteEncoder(vocab_size=200_000))
    assert DatasetIndex.read(idx_path(config.output_prefix)).dtype == np.dtype(np.int32)


def test_removes_partitions_by_default(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    config = make_config(tmp_path, input_dir)
    run(config, byte_encoder)
    assert not config.partition_dir.exists()
    assert sorted(p.name for p in config.output_prefix.parent.iterdir()) == ["data.bin", "data.idx", "data.json"]


def test_keep_partitions(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    config = make_config(tmp_path, input_dir, keep_partitions=True, work_dir=tmp_path / "work")
    run(config, byte_encoder)
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == [
        "part-00000.bin",
        "part-00000.idx",
        "part-00000.json",
        "part-00001.bin",
        "part-00001.idx",
        "part-00001.json",
    ]
    manifest = json.loads((tmp_path / "work" / "part-00000.json").read_text())
    assert manifest["inputs"]["tokenizer"] == byte_encoder.fingerprint
    assert len(manifest["inputs"]["files"]) > 0
    assert manifest["stats"]["documents"] > 0


class TestResume:
    def test_reuses_finished_partitions(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        run(config, byte_encoder)
        first_output = (tmp_path / "out" / "data.bin").read_bytes()

        rerun_encoder = ByteEncoder()
        summary = run(config, rerun_encoder)

        assert summary.reused_partitions == 2
        assert rerun_encoder.calls == 0
        assert (tmp_path / "out" / "data.bin").read_bytes() == first_output
        assert summary.documents == len(DOCUMENTS)

    def test_reused_partitions_keep_their_stats(self, tmp_path: Path, byte_encoder: ByteEncoder) -> None:
        input_dir = tmp_path / "input"
        write_corpus(input_dir, [["a", ""], ["b", "", ""]])
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        first = run(config, byte_encoder)

        second = run(config, ByteEncoder())

        assert second.reused_partitions == 2
        assert second.empty_documents == first.empty_documents == 3
        assert [p.stats for p in second.partitions] == [p.stats for p in first.partitions]

    def test_ignores_manifest_in_old_format(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        run(config, byte_encoder)
        (config.partition_dir / "part-00000.json").write_text('{"files": []}')

        assert run(config, ByteEncoder()).reused_partitions == 1

    def test_redoes_only_missing_partitions(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        run(config, byte_encoder)
        (config.partition_dir / "part-00001.idx").unlink()

        summary = run(config, ByteEncoder())

        assert summary.reused_partitions == 1
        assert decoded(config.output_prefix, byte_encoder) == DOCUMENTS

    def test_redoes_partitions_when_tokenizer_changes(
        self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder
    ) -> None:
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        run(config, byte_encoder)
        summary = run(config, ByteEncoder(name="other"))
        assert summary.reused_partitions == 0

    def test_redoes_partitions_when_settings_change(
        self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder
    ) -> None:
        config = make_config(tmp_path, input_dir, keep_partitions=True)
        run(config, byte_encoder)
        summary = run(make_config(tmp_path, input_dir, keep_partitions=True, eod_id=0), byte_encoder)
        assert summary.reused_partitions == 0
        assert decoded(config.output_prefix, byte_encoder, eod=0) == DOCUMENTS

    def test_failed_partition_leaves_no_complete_dataset(self, tmp_path: Path, input_dir: Path) -> None:
        class FailingEncoder(ByteEncoder):
            def encode_batch(self, texts: list[str]):  # type: ignore[override]
                if any("fifth" in t for t in texts):
                    raise RuntimeError("tokenizer crashed")
                return super().encode_batch(texts)

        config = make_config(tmp_path, input_dir, num_partitions=len(CORPUS))  # one file per partition
        with pytest.raises(RuntimeError, match="crashed"):
            run(config, FailingEncoder())

        assert dataset_exists(config.partition_dir / "part-00001")
        assert not dataset_exists(config.partition_dir / "part-00002")
        assert not dataset_exists(config.output_prefix)
        assert not list(config.partition_dir.glob("*.tmp"))


class TestSummaryFile:
    def test_written_next_to_dataset(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, eod_id=0, num_partitions=3, batch_bytes=1000)

        summary = run(config, byte_encoder)

        data = json.loads(summary_path(config.output_prefix).read_text())
        assert summary_path(config.output_prefix) == tmp_path / "out" / "data.json"
        assert data == summary.to_dict()
        assert data["tool"]["name"] == "oellm-gigatokenize"
        assert data["output"] == {
            "bin": str((tmp_path / "out" / "data.bin").resolve()),
            "idx": str((tmp_path / "out" / "data.idx").resolve()),
            "dtype": "uint16",
        }

    def test_contains_config(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, eod_id=0, num_partitions=3, batch_bytes=1000, text_key="text")

        config_data = run(config, byte_encoder).to_dict()["config"]

        assert config_data == {
            "input_dir": str(input_dir.resolve()),
            "output_prefix": str((tmp_path / "out" / "data").resolve()),
            "num_partitions": 3,
            "text_key": "text",
            "pattern": "*.jsonl.zst",
            "batch_bytes": 1000,
            "eod_id": 0,
            "work_dir": str((tmp_path / "out" / "data.partitions").resolve()),
            "keep_partitions": False,
            "tokenizer": "bytes",
            "vocab_size": 300,
        }

    def test_contains_totals_and_partitions(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        data = run(make_config(tmp_path, input_dir, num_partitions=2), byte_encoder).to_dict()

        totals = data["totals"]
        assert totals["input_files"] == len(CORPUS)
        assert totals["input_bytes"] == sum(f.stat().st_size for f in input_dir.iterdir())
        assert totals["partitions"] == 2
        assert totals["reused_partitions"] == 0
        assert totals["documents"] == len(DOCUMENTS)
        assert totals["tokens"] == sum(len(d.encode()) for d in DOCUMENTS)
        assert totals["empty_documents"] == 0
        partitions = data["partitions"]
        assert [p["number"] for p in partitions] == [0, 1]
        assert sum(len(p["input_files"]) for p in partitions) == len(CORPUS)
        assert sum(p["documents"] for p in partitions) == len(DOCUMENTS)
        assert all(not p["reused"] and p["seconds"] >= 0 for p in partitions)

    def test_contains_timing(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        before = datetime.now(UTC)
        timing = run(make_config(tmp_path, input_dir), byte_encoder).to_dict()["timing"]
        after = datetime.now(UTC)

        started, finished = datetime.fromisoformat(timing["started_at"]), datetime.fromisoformat(timing["finished_at"])
        assert before <= started <= finished <= after
        assert 0 <= timing["merge_seconds"] <= timing["total_seconds"] <= (after - before).total_seconds()
        assert timing["tokens_per_second"] > 0


def test_no_input_files(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(NoInputFilesError):
        run(make_config(tmp_path, tmp_path / "empty"), byte_encoder)
