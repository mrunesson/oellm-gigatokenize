"""Turning batches of text into flat token arrays, backed by Gigatoken."""

from dataclasses import dataclass
from typing import Protocol

import awkward as ak
import gigatoken
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    """A batch of tokenized documents: all tokens concatenated plus each document's length."""

    tokens: npt.NDArray[np.integer]
    lengths: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if int(self.lengths.sum()) != len(self.tokens):
            raise ValueError("sum of lengths does not match the number of tokens")

    @property
    def num_documents(self) -> int:
        return len(self.lengths)

    def drop_empty(self) -> "EncodedBatch":
        """Remove documents without tokens (tokens are untouched since they have none)."""
        return EncodedBatch(tokens=self.tokens, lengths=self.lengths[self.lengths > 0])

    def with_eod(self, eod_id: int) -> "EncodedBatch":
        """Append ``eod_id`` to the end of every document."""
        ends = np.cumsum(self.lengths)
        tokens = np.insert(self.tokens, ends, np.asarray(eod_id, dtype=self.tokens.dtype))
        return EncodedBatch(tokens=tokens, lengths=self.lengths + 1)


class Encoder(Protocol):
    """Anything that can tokenize a batch of documents."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def fingerprint(self) -> str:
        """Identifies the tokenizer configuration, used to detect stale partitions."""
        ...

    def encode_batch(self, texts: list[str]) -> EncodedBatch: ...


class TokenizerError(ValueError):
    """Raised when a tokenizer cannot be loaded or configured as requested."""


class GigatokenEncoder:
    """An :class:`Encoder` using a Gigatoken tokenizer."""

    def __init__(self, tokenizer: gigatoken.Tokenizer, name: str) -> None:
        self._tokenizer = tokenizer
        self._name = name

    @classmethod
    def load(cls, name_or_path: str) -> "GigatokenEncoder":
        """Load a tokenizer from a HuggingFace Hub repo id or a local tokenizer path."""
        try:
            tokenizer = gigatoken.Tokenizer(name_or_path)
        except Exception as exc:  # gigatoken raises a variety of exception types
            raise TokenizerError(f"could not load tokenizer {name_or_path!r}: {exc}") from exc
        return cls(tokenizer, name_or_path)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def fingerprint(self) -> str:
        return f"gigatoken:{self._name}:vocab={self.vocab_size}"

    def token_to_id(self, token: str) -> int:
        """The id of a token, e.g. a special end-of-document token such as ``<|endoftext|>``."""
        ids = self._tokenizer.encode(token)
        if len(ids) != 1:
            raise TokenizerError(f"{token!r} is not a single token in {self._name!r} (encodes to {ids.tolist()})")
        return int(ids[0])

    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        encoded = self._tokenizer.encode_batch(texts)
        lengths = ak.num(encoded, axis=1).to_numpy().astype(np.int64)
        tokens = ak.flatten(encoded, axis=1).to_numpy()
        return EncodedBatch(tokens=tokens, lengths=lengths)
