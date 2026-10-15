import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from oellm_gigatokenize.executor import InlineExecutor, make_executor


def test_inline_executor_runs_in_calling_thread() -> None:
    future = InlineExecutor().submit(lambda x: (x * 2, threading.current_thread()), 21)
    assert future.done()
    assert future.result() == (42, threading.current_thread())


def test_inline_executor_captures_exceptions() -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    future = InlineExecutor().submit(fail)
    with pytest.raises(RuntimeError, match="boom"):
        future.result()


def test_make_executor() -> None:
    assert isinstance(make_executor(1), InlineExecutor)
    with make_executor(3) as executor:
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor._max_workers == 3


def test_make_executor_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_executor(0)
