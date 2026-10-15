"""Command line interface."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from oellm_gigatokenize import __version__
from oellm_gigatokenize.pipeline import DEFAULT_BATCH_BYTES, PipelineConfig, run
from oellm_gigatokenize.reader import DEFAULT_PATTERN, InputFormatError
from oellm_gigatokenize.tokenization import GigatokenEncoder, TokenizerError

logger = logging.getLogger("oellm_gigatokenize")

DEFAULT_TOKENIZER = "openeurollm/tokenizer-256k-v2"
DEFAULT_EOD_TOKEN = "<eos>"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oellm-gigatokenize",
        description=(
            "Tokenize a directory of zstd-compressed JSON Lines files with Gigatoken and write "
            "a Megatron-LM bin/idx dataset. Files are tokenized in partitions that are stored "
            "one at a time and merged at the end; re-running reuses finished partitions."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="directory containing *.jsonl.zst files (searched recursively)")
    parser.add_argument(
        "output_prefix",
        type=Path,
        help="output path without extension; writes <prefix>.bin and <prefix>.idx",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HuggingFace Hub repo id or path to a tokenizer.json / directory (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-partitions",
        type=_positive_int,
        required=True,
        help="number of partitions to split the input files into (capped at the number of files)",
    )
    parser.add_argument(
        "--text-key", default="text", help="JSON field holding the document text (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="glob pattern for input files (default: %(default)s)"
    )
    eod = parser.add_mutually_exclusive_group()
    eod.add_argument(
        "--eod-token",
        help=f"token appended after every document (default: {DEFAULT_EOD_TOKEN})",
    )
    eod.add_argument("--eod-token-id", type=int, help="token id appended after every document")
    eod.add_argument("--no-eod", action="store_true", help="do not append an end-of-document token")
    parser.add_argument(
        "--batch-bytes",
        type=_positive_int,
        default=DEFAULT_BATCH_BYTES,
        help="approximate characters of text tokenized per batch; bounds memory use (default: %(default)s)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="where partitions are stored (default: <output_prefix>.partitions)",
    )
    parser.add_argument("--keep-partitions", action="store_true", help="keep the partition files after merging")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _resolve_eod_id(args: argparse.Namespace, encoder: GigatokenEncoder) -> int | None:
    if args.no_eod:
        return None
    if args.eod_token_id is not None:
        return args.eod_token_id
    return encoder.token_to_id(args.eod_token or DEFAULT_EOD_TOKEN)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        encoder = GigatokenEncoder.load(args.tokenizer)
        eod_id = _resolve_eod_id(args, encoder)
        config = PipelineConfig(
            input_dir=args.input_dir,
            output_prefix=args.output_prefix,
            num_partitions=args.num_partitions,
            text_key=args.text_key,
            pattern=args.pattern,
            batch_bytes=args.batch_bytes,
            eod_id=eod_id,
            work_dir=args.work_dir,
            keep_partitions=args.keep_partitions,
        )
        summary = run(config, encoder)
    except (TokenizerError, InputFormatError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted; finished partitions are kept and will be reused on the next run")
        return 130

    logger.info(
        "Wrote %d documents and %d tokens to %s.{bin,idx,json} in %.1fs "
        "(%d of %d partitions reused, %d empty documents skipped)",
        summary.documents,
        summary.tokens,
        summary.output_prefix,
        summary.total_seconds,
        summary.reused_partitions,
        summary.num_partitions,
        summary.empty_documents,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
