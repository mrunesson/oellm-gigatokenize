# oellm-gigatokenize

Tokenize a directory of zstd-compressed JSON Lines files with
[Gigatoken](https://github.com/marcelroed/gigatoken). Every input file becomes
one Megatron-LM `bin/idx` (`MMapIndexedDataset`) dataset.

## How it works

1. **Discover** all files matching `--pattern` (default `*.jsonl.zst`) under the input directory, recursively and in sorted order. Input `a/b.jsonl.zst` is written to `OUTPUT_DIR/a/b.bin` and `b.idx`.
2. **Partition** each file's documents round robin over `--num-partitions` partitions (default 1, no partitioning). Partitioning exists mainly so results can be compared with older datasets made with Megatron's `preprocess_data.py --partitions`: with the same n, the output is the same as Megatron's. Every non-blank row is a document, and document `i` (counting from 0) goes to partition `i % n`. So with 3 partitions, partition 0 gets documents 0, 3, 6, …, partition 1 gets 1, 4, 7, …, and partition 2 gets 2, 5, 8, ….
3. **Tokenize** the partitions on `--workers` threads (default 1, which runs everything in the main thread). Each partition streams the file and parses only its own rows. It tokenizes them in batches of about `--batch-bytes` of text, straight into its own partition `bin/idx`.
4. **Merge** each file's partitions into that file's output, partition 0 first. The output order is therefore documents 0, n, 2n, …, 1, n+1, …, not the input order. Partitions are then deleted unless `--keep-partitions` is given. For a given `--num-partitions`, the output is byte-identical whatever the number of workers or the batch size.
5. **Summarize**:
   * `OUTPUT_DIR/<name>.json` next to each dataset records that file's inputs, documents, tokens, timing and its partitions (counts, seconds).
   * `OUTPUT_DIR/_summary.json` covers the whole run:
     * `config`: every setting used, including the tokenizer and its vocabulary size.
     * `timing`: start and finish time (UTC), total seconds, and tokens per second.
     * `totals`: files, bytes, partitions, documents, tokens and empty documents, and how many files and partitions were reused.
     * `files`: the per-file results.

Partitions and outputs are written to temporary files and only moved into place when complete. Each one gets a manifest recording its input file, tokenizer and settings. If a run is interrupted, re-running the same command skips finished files and reuses finished partitions of unfinished ones, as long as the manifests still match. If one task fails, the remaining queued tasks are cancelled and running ones stop at their next batch.

## Usage

```bash
uv sync
uv run oellm-gigatokenize INPUT_DIR OUTPUT_DIR --num-partitions 8 --workers 4
```

| Option | Description |
|---|---|
| `-t, --tokenizer` | HuggingFace Hub repo id or a local `tokenizer.json` / directory (default `openeurollm/tokenizer-256k-v2`) |
| `-n, --num-partitions` | Round-robin partitions per input file (default 1) |
| `-w, --workers` | Worker threads processing partitions; 1 runs without threads (default 1) |
| `--eod-token` / `--eod-token-id` | Token appended after every document (default `<eos>`) |
| `--no-eod` | Do not append an end-of-document token |
| `--text-key` | JSON field containing the text (default `text`) |
| `--pattern` | Glob for input files (default `*.jsonl.zst`) |
| `--batch-bytes` | Text per tokenizer call, per worker; bounds memory (default 256 MiB) |
| `--work-dir` | Where partitions go (default `OUTPUT_DIR/.partitions`) |
| `--keep-partitions` | Keep the partition files after merging |

Notes:

* The token dtype follows Megatron's rule: `uint16` when the vocabulary has fewer than 65,500 tokens, otherwise `int32`.
* Every document is stored as its own sequence. A document that tokenizes to nothing gets no sequence and no end-of-document token, but still an entry in the index's document list. This matches Megatron's `preprocess_data.py`, so the `.idx` files are byte-identical to Megatron's.
* Gigatoken matches special-token strings (such as `<|endoftext|>`) that appear in the raw text and encodes them as special tokens.
* Every partition decompresses the whole file, so a file is decompressed n times, while each row is parsed and tokenized only once.
* Gigatoken already tokenizes each batch on all CPU cores, so the recommended number of threads per file is 1, which is the default. Extra workers mainly overlap decompression and JSON parsing with tokenization.
* New files are started only while fewer partitions than workers are in progress, so only a few files' partitions are on disk at any time.
* Memory use grows with the number of workers, roughly `workers × batch-bytes` of text plus its tokens.

## Running on a Slurm cluster

Submit from the repository root, with your own time limit:

```bash
sbatch --time=02:00:00 --account=PROJECT scripts/tokenize.sbatch INPUT_DIR OUTPUT_DIR [tool options...]
```

The job takes a whole node and runs the tool with `uv run`. Options after the two directories go to `oellm-gigatokenize`. For the time limit: a 27,828,971,364-byte input file became 10,058,027,380 tokens in 600 s, about 46 MB of input per second, or roughly 1 hour per 167 GB. If the job runs out of time, submit it again, and finished files are reused.

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
  pipeline.py         per-file plan -> tokenize partitions -> merge, task scheduling, resume
  executor.py         inline (no threads) or thread-pool executor
  results.py          partition/file/run results, manifests and JSON summaries
  reader.py           file discovery, streaming zstd/JSONL reading of one partition
  partitioning.py     round-robin assignment of documents to partitions
  tokenization.py     Encoder protocol, Gigatoken backend, EOD/empty-document handling
  indexed_dataset.py  Megatron bin/idx writer, reader and merger
```
