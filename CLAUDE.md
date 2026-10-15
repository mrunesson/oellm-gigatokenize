# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                   # install runtime + dev dependencies
uv run pytest                             # full test suite
uv run pytest -m "not network"            # skip tests that download tokenizers from the HF Hub
uv run pytest tests/test_pipeline.py::TestResume::test_reuses_finished_partitions   # single test
uv run ruff format . && uv run ruff check .
uvx ty check src tests                    # type check
uv run oellm-gigatokenize INPUT_DIR OUT_PREFIX -n 16   # run the CLI
```

## What the tool does

`oellm-gigatokenize` tokenizes a directory of `*.jsonl.zst` files with Gigatoken and writes one Megatron-LM `MMapIndexedDataset` (`<prefix>.bin` + `<prefix>.idx`). Defaults: tokenizer `openeurollm/tokenizer-256k-v2`, and `<eos>` appended after every document (`--no-eod` disables it).

## Architecture

The data flow is `cli.main` → `pipeline.run(config, encoder)`:

1. `reader.discover_files` finds the inputs (recursive, sorted). `partitioning.split_into_partitions` groups them into contiguous, order-preserving partitions balanced by compressed size. Files are never split, so the partition count is capped at the file count.
2. For each partition, `pipeline.tokenize_partition` streams documents (`reader.iter_documents` → `iter_batches` by `--batch-bytes`) through the encoder and appends each batch to a `DatasetWriter`. Memory is bounded by the batch size, not the partition size.
3. `indexed_dataset.merge_datasets` concatenates the partition `.bin` files and rebuilds the index, then the partitions are removed unless `--keep-partitions` is set.

Key design points that span files:

- **Encoder protocol** (`tokenization.Encoder`): the pipeline depends only on `vocab_size`, `fingerprint` and `encode_batch(list[str]) -> EncodedBatch`. `GigatokenEncoder` is the real implementation. Tests inject `tests.conftest.ByteEncoder` (one token per UTF-8 byte, +1 offset). `EncodedBatch` is a flat token array plus per-document lengths. EOD appending (`with_eod`) and empty-document dropping (`drop_empty`) are applied by the pipeline, not by the encoder.
- **Resume/atomicity**: `DatasetWriter` writes `*.bin.tmp`/`*.idx.tmp` and renames them into place on `finalize()`, `.bin` first and then `.idx`. Each finished partition gets a `part-NNNNN.json` `PartitionManifest` (input file paths and sizes, encoder `fingerprint`, text key, eod id, dtype). A partition is reused only when its manifest matches exactly, so any change to these fields must be reflected in the manifest or stale partitions will be reused.
- **Megatron format**: `indexed_dataset.py` documents the `.idx` layout in its module docstring. Every document is exactly one sequence (`document_indices = [0..N]`), but `merge_datasets` preserves multi-sequence documents from external datasets. The token dtype comes from `optimal_dtype`, which mirrors Megatron: `uint16` if vocab < 65500, else `int32` (so the default 262k tokenizer produces `int32`).
- **Gigatoken specifics**: `gigatoken.Tokenizer(name_or_path)` loads HF Hub ids or local `tokenizer.json`. `encode_batch` returns an awkward array, which is flattened with `ak.flatten` / `ak.num`. The HF compat layer does not expose `eos_token_id`, so EOD ids are resolved by encoding the token string and requiring exactly one id (`GigatokenEncoder.token_to_id`). Special-token strings that appear in raw text are encoded as special tokens.

## Testing notes

- `tests/conftest.py` builds a tiny byte-level BPE `tokenizer.json` offline: 256 byte tokens, merge `ab`→256, `<|endoftext|>`=257, `<eos>`=258. CLI and tokenization tests run real Gigatoken against it without network access.
- Tests marked `@pytest.mark.network` download real tokenizers (gpt2, openeurollm). A Megatron cross-check test is skipped unless `megatron-core` is installed.
- Test helpers are imported as `from tests.conftest import ...` (`tests/` is a package).
