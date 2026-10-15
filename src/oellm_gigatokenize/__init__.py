"""Tokenize zstd-compressed JSON Lines corpora with Gigatoken into Megatron bin/idx datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oellm-gigatokenize")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
