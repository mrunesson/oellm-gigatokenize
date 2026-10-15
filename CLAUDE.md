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
uv run oellm-gigatokenize INPUT_DIR OUT_DIR -n 8 -w 4   # run the CLI
```

## What the tool does

`oellm-gigatokenize` tokenizes a directory of `*.jsonl.zst` files with Gigatoken. Each input file becomes one Megatron-LM `MMapIndexedDataset` in the output directory (`a/b.jsonl.zst` → `OUT/a/b.bin`, `.idx`, `.json`), plus `OUT/_summary.json` for the run. `-n` deals each file's documents round robin over n partitions. Defaults: tokenizer `openeurollm/tokenizer-256k-v2`, `<eos>` appended after every document (`--no-eod` disables it), 1 partition per file, 1 worker.

## Architecture

The data flow is `cli.main` → `pipeline.run(config, encoder)`. Each file is split into `num_partitions` tasks plus a merge task, run by an executor from `executor.make_executor` (`InlineExecutor` in the main thread for `workers == 1`, else a `ThreadPoolExecutor`):

1. `tokenize_partition`: partition k streams the whole file but keeps only non-blank rows `i` with `i % n == k` (`reader.iter_rows` → `partitioning.round_robin` → `parse_document`). It tokenizes them in batches into a partition `DatasetWriter`, and checks a `stop` event between batches so a failure elsewhere aborts it.
2. `finalize_file`: concatenates the partitions in partition order into the file's output (or moves the single partition), writes `<name>.json`, and removes the partitions. The output order is round robin (0, n, 2n, …, 1, n+1, …), not the input order. Tests rely on this via `round_robin_order`.

`_Scheduler` submits tasks and runs their continuations in the main thread, in submission order, which keeps the inline mode deterministic. `fill()` starts a new file only while fewer than `workers` partitions are outstanding, which bounds how many partitions are on disk. On any exception the stop event is set and queued futures are cancelled. Results and JSON shapes live in `results.py` (`PartitionResult`, `FileResult`, `RunSummary`, each with `to_dict`).

Key design points that span files:

- **Encoder protocol** (`tokenization.Encoder`): the pipeline depends only on `name`, `vocab_size`, `fingerprint` and `encode_batch(list[str]) -> EncodedBatch`. `GigatokenEncoder` is the real implementation. Tests inject `tests.conftest.ByteEncoder` (one token per UTF-8 byte, +1 offset). `EncodedBatch` is a flat token array plus per-document lengths. EOD appending (`with_eod`, which skips empty documents) is applied by the pipeline, not by the encoder.
- **Resume/atomicity**: `DatasetWriter` writes `*.bin.tmp`/`*.idx.tmp` and renames them into place on `finalize()`, `.bin` first and then `.idx`. Manifests are `{"inputs": ..., "result": ...}` JSON (`results.write_json` / `load_if_matching`). There are two levels: `OUT/<name>.json` per finished file (`FileJob.inputs`, which includes `num_partitions`) and `<work_dir>/<name>/part-NNNNN.json` per partition (`FileJob.partition_inputs`, which adds the partition number). Work is reused only when `inputs` match exactly, so any new setting that affects output must be added to `FileJob.inputs`, or stale results will be reused. `batch_bytes` and `workers` are deliberately excluded because they don't change the output.
- **Megatron format**: `indexed_dataset.py` documents the `.idx` layout in its module docstring. Output must stay byte-identical to Megatron's `preprocess_data.py`: every document is one sequence, except that an empty document has no sequence and no EOD but still gets a `document_indices` entry (`[a], [], [b]` → `[0, 1, 1, 2]`). `merge_datasets` preserves arbitrary `document_indices`, including multi-sequence documents from external datasets. `pipeline.OUTPUT_FORMAT` is part of `FileJob.inputs`. Bump it whenever the same inputs produce different files. The token dtype comes from `optimal_dtype`, which mirrors Megatron: `uint16` if vocab < 65500, else `int32` (so the default 262k tokenizer produces `int32`).
- **Gigatoken specifics**: `gigatoken.Tokenizer(name_or_path)` loads HF Hub ids or local `tokenizer.json`. `encode_batch` returns an awkward array, which is flattened with `ak.flatten` / `ak.num`. The HF compat layer does not expose `eos_token_id`, so EOD ids are resolved by encoding the token string and requiring exactly one id (`GigatokenEncoder.token_to_id`). Special-token strings that appear in raw text are encoded as special tokens.

## Testing notes

- `tests/conftest.py` builds a tiny byte-level BPE `tokenizer.json` offline: 256 byte tokens, merge `ab`→256, `<|endoftext|>`=257, `<eos>`=258. CLI and tokenization tests run real Gigatoken against it without network access.
- Tests marked `@pytest.mark.network` download real tokenizers (gpt2, openeurollm). A Megatron cross-check test is skipped unless `megatron-core` is installed.
- Test helpers are imported as `from tests.conftest import ...` (`tests/` is a package).
