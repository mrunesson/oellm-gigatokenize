"""Executors for running pipeline tasks inline or in worker threads."""

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor


class InlineExecutor(Executor):
    """Runs every task immediately in the calling thread; used when no worker threads are wanted."""

    def submit[T](self, fn: Callable[..., T], /, *args: object, **kwargs: object) -> Future[T]:
        future: Future[T] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def make_executor(workers: int) -> Executor:
    """An inline executor for one worker, otherwise a thread pool with ``workers`` threads."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return InlineExecutor()
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tokenize")
