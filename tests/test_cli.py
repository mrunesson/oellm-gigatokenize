import json
import subprocess
import sys
from pathlib import Path

import pytest

from oellm_gigatokenize.cli import main
from oellm_gigatokenize.indexed_dataset import read_documents
from oellm_gigatokenize.tokenization import GigatokenEncoder, TokenizerError
from tests.conftest import EOT, write_corpus

CORPUS = [["ab c", "hello"], ["second file"], ["ab", "third file"]]


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    write_corpus(tmp_path / "input", CORPUS)
    return tmp_path / "input"


def documents(prefix: Path) -> list[list[int]]:
    return [doc.tolist() for doc in read_documents(prefix)]


def test_end_to_end_with_gigatoken(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    output = tmp_path / "out"
    args = [str(input_dir), str(output), "-t", str(tiny_tokenizer_json), "-n", "2", "-w", "2", "--eod-token", EOT]

    assert main(args) == 0

    assert documents(output / "shard-000") == [[256, ord(" "), ord("c"), 257], [*b"hello", 257]]
    assert documents(output / "shard-001") == [[*b"second file", 257]]
    assert documents(output / "shard-002") == [[256, 257], [*b"third file", 257]]
    summary = json.loads((output / "_summary.json").read_text())
    assert summary["config"]["tokenizer"] == str(tiny_tokenizer_json)
    assert summary["config"]["eod_id"] == 257
    assert summary["config"]["num_partitions"] == 2
    assert summary["config"]["workers"] == 2
    assert summary["totals"]["documents"] == 5


def test_defaults_to_one_partition_and_one_worker(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    assert main([str(input_dir), str(tmp_path / "out"), "-t", str(tiny_tokenizer_json)]) == 0
    config = json.loads((tmp_path / "out" / "_summary.json").read_text())["config"]
    assert (config["num_partitions"], config["workers"]) == (1, 1)


def test_eos_is_the_default_eod_token(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    assert main([str(input_dir), str(tmp_path / "out"), "-t", str(tiny_tokenizer_json)]) == 0
    assert all(doc[-1] == 258 for doc in documents(tmp_path / "out" / "shard-000"))


def test_eod_token_id(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    assert main([str(input_dir), str(tmp_path / "out"), "-t", str(tiny_tokenizer_json), "--eod-token-id", "0"]) == 0
    assert all(doc[-1] == 0 for doc in documents(tmp_path / "out" / "shard-000"))


def test_without_eod(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    assert main([str(input_dir), str(tmp_path / "out"), "-t", str(tiny_tokenizer_json), "-n", "3", "--no-eod"]) == 0
    assert documents(tmp_path / "out" / "shard-000")[1] == list(b"hello")


def test_keep_partitions_and_work_dir(tmp_path: Path, input_dir: Path, tiny_tokenizer_json: Path) -> None:
    args = [str(input_dir), str(tmp_path / "out"), "-t", str(tiny_tokenizer_json), "-n", "3"]
    assert main([*args, "--keep-partitions", "--work-dir", str(tmp_path / "work")]) == 0
    assert len(list((tmp_path / "work").rglob("*.idx"))) == 3 * len(CORPUS)


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--eod-token", "not-a-single-token"], "not a single token"),
        (["--text-key", "missing"], "missing key"),
        (["--pattern", "*.parquet"], "no files matching"),
        (["--eod-token-id", "100000"], "outside the vocabulary"),
    ],
)
def test_errors_are_reported(
    tmp_path: Path,
    input_dir: Path,
    tiny_tokenizer_json: Path,
    caplog: pytest.LogCaptureFixture,
    extra_args: list[str],
    message: str,
) -> None:
    args = [str(input_dir), str(tmp_path / "corpus"), "-t", str(tiny_tokenizer_json), "-n", "1", *extra_args]
    assert main(args) == 1
    assert message in caplog.text


def test_unknown_tokenizer(tmp_path: Path, input_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert main([str(input_dir), str(tmp_path / "c"), "-t", str(tmp_path / "nope.json"), "-n", "1"]) == 1
    assert "could not load tokenizer" in caplog.text


def test_missing_input_directory(tmp_path: Path, tiny_tokenizer_json: Path) -> None:
    assert main([str(tmp_path / "missing"), str(tmp_path / "c"), "-t", str(tiny_tokenizer_json), "-n", "1"]) == 1


@pytest.mark.parametrize(
    "args",
    [
        ["in"],  # missing output directory
        ["in", "out", "-t", "tok", "-n", "0"],
        ["in", "out", "-t", "tok", "-w", "0"],
        ["in", "out", "-t", "tok", "-n", "2", "--eod-token", "x", "--eod-token-id", "1"],
        ["in", "out", "-t", "tok", "-n", "2", "--eod-token", "x", "--no-eod"],
    ],
)
def test_invalid_arguments_exit_with_usage_error(args: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(args)
    assert excinfo.value.code == 2


def test_tokenizer_defaults_to_openeurollm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded: list[str] = []

    def fake_load(name: str) -> GigatokenEncoder:
        loaded.append(name)
        raise TokenizerError("stop here")

    monkeypatch.setattr(GigatokenEncoder, "load", staticmethod(fake_load))
    assert main([str(tmp_path), str(tmp_path / "out"), "-n", "1"]) == 1
    assert loaded == ["openeurollm/tokenizer-256k-v2"]


def test_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "oellm_gigatokenize", "--help"], capture_output=True, text=True, check=True
    )
    assert "--num-partitions" in result.stdout
    assert "--workers" in result.stdout
