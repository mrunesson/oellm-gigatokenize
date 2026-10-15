"""Shared test fixtures and helpers."""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import orjson
import pytest
import zstandard

from oellm_gigatokenize.tokenization import EncodedBatch

EOT = "<|endoftext|>"
EOS = "<eos>"


def write_jsonl_zst(path: Path, records: Iterable[object]) -> Path:
    """Write ``records`` as zstd-compressed JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(orjson.dumps(record) + b"\n" for record in records)
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    return path


def write_corpus(directory: Path, documents_per_file: list[list[str]]) -> list[str]:
    """Write one file per inner list and return all documents in file order."""
    for number, documents in enumerate(documents_per_file):
        write_jsonl_zst(directory / f"shard-{number:03d}.jsonl.zst", ({"text": d} for d in documents))
    return [doc for documents in documents_per_file for doc in documents]


class ByteEncoder:
    """A deterministic fake encoder: one token per UTF-8 byte, offset by one so 0 is free."""

    def __init__(self, vocab_size: int = 300, name: str = "bytes") -> None:
        self.vocab_size = vocab_size
        self.fingerprint = f"fake:{name}"
        self.calls = 0

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode(), dtype=np.uint8).astype(np.uint32) + 1

    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        self.calls += 1
        encoded = [self.encode(t) for t in texts]
        tokens = np.concatenate(encoded) if encoded else np.empty(0, np.uint32)
        return EncodedBatch(tokens=tokens, lengths=np.array([len(e) for e in encoded], dtype=np.int64))


@pytest.fixture
def byte_encoder() -> ByteEncoder:
    return ByteEncoder()


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode character mapping."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars, strict=True)}


@pytest.fixture(scope="session")
def tiny_tokenizer_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal byte-level BPE tokenizer.json: 256 byte tokens, one merge ("ab"), EOT and EOS.

    Lets the real Gigatoken backend be tested without network access.
    """
    b2u = _bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab[b2u[ord("a")] + b2u[ord("b")]] = 256
    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 257,
                "content": EOT,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 258,
                "content": EOS,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [[b2u[ord("a")], b2u[ord("b")]]],
        },
    }
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(json.dumps(config))
    return path
