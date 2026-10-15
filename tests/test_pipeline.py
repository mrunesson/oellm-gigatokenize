import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from oellm_gigatokenize.indexed_dataset import DatasetIndex, dataset_exists, idx_path, read_documents
from oellm_gigatokenize.pipeline import NoInputFilesError, PipelineConfig, output_name, run
from oellm_gigatokenize.reader import InputFile
from oellm_gigatokenize.tokenization import EncodedBatch
from tests.conftest import ByteEncoder, write_corpus, write_jsonl_zst

CORPUS = [
    ["first document", "second", "ünïcödé ✓"],
    ["fourth"],
    ["fifth document is a bit longer " * 5, "sixth", *[f"filler {i}" for i in range(30)]],
    ["seventh", "eighth", "ninth", "tenth"],
]
NAMES = [f"shard-{i:03d}" for i in range(len(CORPUS))]


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "input"
    write_corpus(directory, CORPUS)
    return directory


def make_config(tmp_path: Path, input_dir: Path, **overrides: object) -> PipelineConfig:
    settings: dict = {"input_dir": input_dir, "output_dir": tmp_path / "out"}
    settings.update(overrides)
    return PipelineConfig(**settings)


def decoded(prefix: Path, eod: int | None = None) -> list[str]:
    documents = []
    for tokens in read_documents(prefix):
        ids = tokens.tolist()
        if eod is not None:
            assert ids[-1] == eod
            ids = ids[:-1]
        documents.append(bytes(t - 1 for t in ids).decode())
    return documents


def round_robin_order(documents: list[str], num_partitions: int) -> list[str]:
    """The order documents end up in: partition 0's documents, then partition 1's, ..."""
    return [doc for k in range(num_partitions) for doc in documents[k::num_partitions]]


def expected(num_partitions: int = 1) -> list[list[str]]:
    return [round_robin_order(docs, num_partitions) for docs in CORPUS]


def outputs(output_dir: Path) -> list[list[str]]:
    return [decoded(output_dir / name) for name in NAMES]


class SlowEncoder(ByteEncoder):
    """Slows each call down a little so that work spreads over the worker threads."""

    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        time.sleep(0.01)
        return super().encode_batch(texts)


class FailingEncoder(ByteEncoder):
    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        if any("fifth" in t for t in texts):
            raise RuntimeError("tokenizer crashed")
        return super().encode_batch(texts)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 7, 50])
def test_each_input_file_becomes_one_dataset(
    tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder, num_partitions: int, workers: int
) -> None:
    config = make_config(tmp_path, input_dir, num_partitions=num_partitions, workers=workers)

    summary = run(config, byte_encoder)

    assert outputs(config.output_dir) == expected(num_partitions)
    assert summary.documents == sum(map(len, CORPUS))
    assert [len(f.partitions) for f in summary.files] == [num_partitions] * len(CORPUS)
    assert [f.documents for f in summary.files] == [len(docs) for docs in CORPUS]


def test_partitions_are_dealt_round_robin(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_corpus(input_dir, [[f"d{i}" for i in range(8)]])
    config = make_config(tmp_path, input_dir, num_partitions=3, keep_partitions=True)

    summary = run(config, byte_encoder)

    parts = [decoded(config.partition_dir / "shard-000" / f"part-{k:05d}") for k in range(3)]
    assert parts == [["d0", "d3", "d6"], ["d1", "d4", "d7"], ["d2", "d5"]]
    assert decoded(config.output_dir / "shard-000") == ["d0", "d3", "d6", "d1", "d4", "d7", "d2", "d5"]
    assert [p.stats.documents for p in summary.files[0].partitions] == [3, 3, 2]


def test_output_is_independent_of_workers_and_batching(
    tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder
) -> None:
    run(make_config(tmp_path, input_dir, output_dir=tmp_path / "a", num_partitions=4), byte_encoder)
    run(
        make_config(tmp_path, input_dir, output_dir=tmp_path / "b", num_partitions=4, workers=3, batch_bytes=7),
        byte_encoder,
    )
    for name in NAMES:
        for suffix in (".bin", ".idx"):
            assert (tmp_path / "a" / f"{name}{suffix}").read_bytes() == (
                tmp_path / "b" / f"{name}{suffix}"
            ).read_bytes()


def test_mirrors_input_directory_structure(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_jsonl_zst(input_dir / "en" / "web" / "part-1.jsonl.zst", [{"text": "hello"}])
    write_jsonl_zst(input_dir / "sv" / "part-1.jsonl.zst", [{"text": "hej"}])
    config = make_config(tmp_path, input_dir, num_partitions=2)

    run(config, byte_encoder)

    assert decoded(config.output_dir / "en" / "web" / "part-1") == ["hello"]
    assert decoded(config.output_dir / "sv" / "part-1") == ["hej"]


def test_output_directory_contents(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    config = make_config(tmp_path, input_dir, num_partitions=3)
    run(config, byte_encoder)
    expected = sorted(["_summary.json", *(f"{n}{s}" for n in NAMES for s in (".bin", ".idx", ".json"))])
    assert sorted(p.name for p in config.output_dir.iterdir()) == expected


@pytest.mark.parametrize("num_partitions", [1, 3])
def test_keep_partitions(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder, num_partitions: int) -> None:
    config = make_config(
        tmp_path, input_dir, num_partitions=num_partitions, keep_partitions=True, work_dir=tmp_path / "work"
    )
    run(config, byte_encoder)
    for name in NAMES:
        assert sorted(p.name for p in (tmp_path / "work" / name).iterdir()) == sorted(
            f"part-{k:05d}{s}" for k in range(num_partitions) for s in (".bin", ".idx", ".json")
        )
    assert outputs(config.output_dir) == expected(num_partitions)


def test_uses_worker_threads(tmp_path: Path, input_dir: Path) -> None:
    encoder = SlowEncoder()
    run(make_config(tmp_path, input_dir, num_partitions=4, workers=3, batch_bytes=1), encoder)
    assert len(encoder.threads) > 1
    assert all(name.startswith("tokenize") for name in encoder.threads)


class PartitionCountingEncoder(ByteEncoder):
    """Records the most partition datasets that existed on disk at any one time."""

    def __init__(self, partition_dir: Path) -> None:
        super().__init__()
        self.partition_dir = partition_dir
        self.most_partitions = 0

    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        on_disk = len(list(self.partition_dir.rglob("*.bin*")))
        self.most_partitions = max(self.most_partitions, on_disk)
        return super().encode_batch(texts)


@pytest.mark.parametrize("workers", [1, 2])
def test_processes_few_files_at_a_time(tmp_path: Path, workers: int) -> None:
    input_dir = tmp_path / "input"
    write_corpus(input_dir, [[f"file {i} doc {j}" for j in range(6)] for i in range(10)])
    config = make_config(tmp_path, input_dir, num_partitions=3, workers=workers)
    encoder = PartitionCountingEncoder(config.partition_dir)

    run(config, encoder)

    # Without a bound, all 30 partitions would be written before any file is merged.
    assert encoder.most_partitions <= 2 * 3 + workers


def test_single_worker_runs_in_main_thread(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    run(make_config(tmp_path, input_dir, num_partitions=4), byte_encoder)
    assert byte_encoder.threads == {"MainThread"}


def test_small_batches_stream_through_encoder(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    run(make_config(tmp_path, input_dir, batch_bytes=1), byte_encoder)
    assert byte_encoder.calls == sum(map(len, CORPUS))


def test_appends_eod(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    config = make_config(tmp_path, input_dir, eod_id=0, num_partitions=2)
    summary = run(config, byte_encoder)
    assert [decoded(config.output_dir / name, eod=0) for name in NAMES] == expected(2)
    assert summary.tokens == sum(len(d.encode()) + 1 for docs in CORPUS for d in docs)


def test_rejects_eod_outside_vocabulary(tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
    with pytest.raises(ValueError, match="outside the vocabulary"):
        run(make_config(tmp_path, input_dir, eod_id=byte_encoder.vocab_size), byte_encoder)


def test_skips_empty_documents(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_corpus(input_dir, [["a", "", "b"], [""]])
    config = make_config(tmp_path, input_dir, num_partitions=2)
    summary = run(config, byte_encoder)
    assert decoded(config.output_dir / "shard-000") == ["a", "b"]  # partition 0: "a", "b"; partition 1: ""
    assert decoded(config.output_dir / "shard-001") == []
    assert summary.empty_documents == 2


def test_uses_int32_for_large_vocabularies(tmp_path: Path, input_dir: Path) -> None:
    config = make_config(tmp_path, input_dir)
    run(config, ByteEncoder(vocab_size=200_000))
    assert DatasetIndex.read(idx_path(config.output_dir / NAMES[0])).dtype == np.dtype(np.int32)


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("a.jsonl.zst", "a"),
        ("a.json.zst", "a"),
        ("a.zst", "a"),
        ("sub/dir/a.b.jsonl.zst", "sub/dir/a.b"),
        ("a.txt.zst", "a.txt"),
    ],
)
def test_output_name(tmp_path: Path, relative: str, expected: str) -> None:
    assert output_name(InputFile(tmp_path / relative, 0), tmp_path) == Path(expected)


def test_rejects_inputs_with_the_same_output_name(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_jsonl_zst(input_dir / "a.jsonl.zst", [{"text": "x"}])
    write_jsonl_zst(input_dir / "a.json.zst", [{"text": "y"}])
    with pytest.raises(ValueError, match="would both be written to"):
        run(make_config(tmp_path, input_dir, pattern="*.zst"), byte_encoder)


def test_rejects_input_that_would_overwrite_summary(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    input_dir = tmp_path / "input"
    write_jsonl_zst(input_dir / "_summary.jsonl.zst", [{"text": "x"}])
    with pytest.raises(ValueError, match="overwrite the run summary"):
        run(make_config(tmp_path, input_dir), byte_encoder)


def test_no_input_files(tmp_path: Path, byte_encoder: ByteEncoder) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(NoInputFilesError):
        run(make_config(tmp_path, tmp_path / "empty"), byte_encoder)


@pytest.mark.parametrize(("num_partitions", "workers"), [(0, 1), (1, 0)])
def test_rejects_invalid_config(tmp_path: Path, num_partitions: int, workers: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_config(tmp_path, tmp_path, num_partitions=num_partitions, workers=workers)


class TestResume:
    def test_reuses_finished_files(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, num_partitions=2)
        first = run(config, byte_encoder)

        rerun_encoder = ByteEncoder()
        second = run(config, rerun_encoder)

        assert rerun_encoder.calls == 0
        assert second.reused_files == len(CORPUS)
        assert [f.documents for f in second.files] == [f.documents for f in first.files]
        assert second.empty_documents == first.empty_documents
        assert outputs(config.output_dir) == expected(2)

    def test_redoes_only_missing_files(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir)
        run(config, byte_encoder)
        (config.output_dir / f"{NAMES[1]}.idx").unlink()

        summary = run(config, ByteEncoder())

        assert [f.reused for f in summary.files] == [True, False, True, True]
        assert outputs(config.output_dir) == expected()

    def test_reuses_finished_partitions_of_unfinished_files(
        self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder
    ) -> None:
        config = make_config(tmp_path, input_dir, num_partitions=3, keep_partitions=True)
        run(config, byte_encoder)
        (config.output_dir / f"{NAMES[2]}.json").unlink()
        (config.partition_dir / NAMES[2] / "part-00001.idx").unlink()

        summary = run(config, ByteEncoder())

        file = summary.files[2]
        assert not file.reused
        assert [p.reused for p in file.partitions] == [True, False, True]
        assert outputs(config.output_dir) == expected(3)

    @pytest.mark.parametrize(
        "change",
        [{"num_partitions": 2}, {"eod_id": 0}, {"text_key": "text", "encoder": "other"}],
    )
    def test_redoes_files_when_settings_change(
        self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder, change: dict
    ) -> None:
        run(make_config(tmp_path, input_dir), byte_encoder)
        encoder = ByteEncoder(name=change.pop("encoder", "bytes"))
        summary = run(make_config(tmp_path, input_dir, **change), encoder)
        assert summary.reused_files == 0

    def test_failure_keeps_finished_work_and_leaves_no_partial_output(self, tmp_path: Path, input_dir: Path) -> None:
        config = make_config(tmp_path, input_dir, num_partitions=2)
        with pytest.raises(RuntimeError, match="crashed"):
            run(config, FailingEncoder())

        assert dataset_exists(config.output_dir / NAMES[0])
        assert dataset_exists(config.output_dir / NAMES[1])
        assert not dataset_exists(config.output_dir / NAMES[2])
        assert not (config.output_dir / "_summary.json").exists()
        assert not list(tmp_path.rglob("*.tmp"))

        summary = run(config, ByteEncoder())
        assert [f.reused for f in summary.files] == [True, True, False, False]
        assert outputs(config.output_dir) == expected(2)

    def test_failure_with_threads_stops_the_run(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        write_corpus(input_dir, [["fifth"], *[[f"doc {i}"] for i in range(20)]])
        config = make_config(tmp_path, input_dir, workers=2)

        class SlowFailingEncoder(FailingEncoder, SlowEncoder):
            pass

        with pytest.raises(RuntimeError, match="crashed"):
            run(config, SlowFailingEncoder())

        assert len(list(config.output_dir.glob("*.bin"))) < 20
        assert not list(tmp_path.rglob("*.tmp"))


class TestSummaryFiles:
    def test_run_summary(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, num_partitions=3, workers=2, eod_id=0, batch_bytes=1000)
        before = datetime.now(UTC)

        summary = run(config, byte_encoder)

        data = json.loads((config.output_dir / "_summary.json").read_text())
        assert data == summary.to_dict()
        assert data["tool"]["name"] == "oellm-gigatokenize"
        assert data["dtype"] == "uint16"
        assert data["config"] == {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(config.output_dir.resolve()),
            "num_partitions": 3,
            "workers": 2,
            "text_key": "text",
            "pattern": "*.jsonl.zst",
            "batch_bytes": 1000,
            "eod_id": 0,
            "work_dir": str((config.output_dir / ".partitions").resolve()),
            "keep_partitions": False,
            "tokenizer": "bytes",
            "vocab_size": 300,
        }
        assert data["totals"] == {
            "input_files": len(CORPUS),
            "input_bytes": sum(f.stat().st_size for f in input_dir.iterdir()),
            "reused_files": 0,
            "partitions": 3 * len(CORPUS),
            "reused_partitions": 0,
            "documents": sum(map(len, CORPUS)),
            "tokens": sum(len(d.encode()) + 1 for docs in CORPUS for d in docs),
            "empty_documents": 0,
        }
        timing = data["timing"]
        started, finished = datetime.fromisoformat(timing["started_at"]), datetime.fromisoformat(timing["finished_at"])
        assert before <= started <= finished <= datetime.now(UTC)
        assert timing["total_seconds"] > 0
        assert timing["tokens_per_second"] > 0
        assert [Path(f["input"]).name for f in data["files"]] == [f"{n}.jsonl.zst" for n in NAMES]

    def test_file_manifest(self, tmp_path: Path, input_dir: Path, byte_encoder: ByteEncoder) -> None:
        config = make_config(tmp_path, input_dir, num_partitions=2)
        summary = run(config, byte_encoder)

        manifest = json.loads((config.output_dir / f"{NAMES[2]}.json").read_text())

        assert manifest["inputs"]["num_partitions"] == 2
        assert manifest["inputs"]["tokenizer"] == byte_encoder.fingerprint
        result = manifest["result"]
        assert result == summary.files[2].to_dict()
        assert result["output_bin"] == str((config.output_dir / f"{NAMES[2]}.bin").resolve())
        assert result["documents"] == len(CORPUS[2])
        partitions = result["partitions"]
        assert [p["number"] for p in partitions] == [0, 1]
        assert [p["documents"] for p in partitions] == [len(CORPUS[2][0::2]), len(CORPUS[2][1::2])]
