# oellm-gigatokenize

Tokenize a directory of zstd-compressed JSON Lines files with
[Gigatoken](https://github.com/marcelroed/gigatoken) and write a single
Megatron-LM `bin/idx` (`MMapIndexedDataset`) dataset.

## How it works

1. **Discover** all files matching `--pattern` (default `*.jsonl.zst`) under the input directory, recursively and in sorted order.
2. **Partition** the files into `--num-partitions` contiguous groups of roughly equal compressed size.
3. **Tokenize** one partition at a time. Each file is decompressed as a stream and tokenized in batches of about `--batch-bytes` of text. Tokens are written straight to that partition's `part-NNNNN.bin/.idx`, so memory use depends on the batch size, not on the corpus size.
4. **Merge** the partitions, in order, into `<output_prefix>.bin` and `<output_prefix>.idx`. Partitions are then deleted unless `--keep-partitions` is given.

Each partition is written to temporary files and only moved into place when it is complete. Next to it goes a manifest recording its input files, tokenizer and settings. If a run is interrupted, re-running the same command reuses every finished partition whose manifest still matches.

## Usage

```bash
uv sync
uv run oellm-gigatokenize INPUT_DIR OUTPUT_PREFIX --num-partitions 16
```

| Option | Description |
|---|---|
| `-t, --tokenizer` | HuggingFace Hub repo id or a local `tokenizer.json` / directory (default `openeurollm/tokenizer-256k-v2`) |
| `-n, --num-partitions` | Number of partitions (required; capped at the number of files) |
| `--eod-token` / `--eod-token-id` | Token appended after every document (default `<eos>`) |
| `--no-eod` | Do not append an end-of-document token |
| `--text-key` | JSON field containing the text (default `text`) |
| `--pattern` | Glob for input files (default `*.jsonl.zst`) |
| `--batch-bytes` | Text per tokenizer call; bounds memory (default 256 MiB) |
| `--work-dir` | Where partitions go (default `<output_prefix>.partitions`) |
| `--keep-partitions` | Keep the partition files after merging |

Notes:

* The token dtype follows Megatron's rule: `uint16` when the vocabulary has fewer than 65,500 tokens, otherwise `int32`.
* Every document is stored as its own sequence. Documents that tokenize to nothing are skipped.
* Gigatoken matches special-token strings (such as `<|endoftext|>`) that appear in the raw text and encodes them as special tokens.
* Files are never split, so the number of partitions cannot exceed the number of input files.

## Development

```bash
uv sync                         # install runtime and dev dependencies
uv run pytest                   # run the tests
uv run pytest -m "not network"  # skip the tests that download tokenizers from the Hub
uv run ruff check . && uv run ruff format --check .
uvx ty check src tests          # type check
```

The tests use a small byte-level BPE `tokenizer.json` built in `tests/conftest.py`, so Gigatoken is exercised without network access. If `megatron-core` is installed, an extra test reads the output with Megatron's own `IndexedDataset`.

### Layout

```
src/oellm_gigatokenize/
  cli.py              argument parsing and entry point
  pipeline.py         partition -> tokenize -> merge orchestration, resume manifests
  reader.py           file discovery and streaming zstd/JSONL reading
  partitioning.py     size-balanced, order-preserving file partitioning
  tokenization.py     Encoder protocol, Gigatoken backend, EOD/empty-document handling
  indexed_dataset.py  Megatron bin/idx writer, reader and merger
```
