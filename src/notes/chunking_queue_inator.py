# cerebrum_core/utils/task_queue_inator.py import asyncio
import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("cerebrum")


@dataclass
class QueuedJob:
    file_fingerprint: str
    original_name: str
    fn: Callable[[], None]


class SingleWorkerQueue:
    """
    Runs at most one file's convert/chunk/embed pipeline at a time,
    process-wide. Ollama chat-model calls (sanitization) and embedding-model
    calls, plus FAISS index writes, all compete for the same CPU/RAM —
    running them one file at a time avoids piling concurrent load onto
    the machine when several files are queued close together.
    """

    def __init__(self):
        self._queue: asyncio.Queue[QueuedJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self.current: Optional[str] = None
        self.pending: list[str] = []

    def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("[QUEUE] worker started")

    async def enqueue(self, job: QueuedJob):
        self.pending.append(job.file_fingerprint)
        await self._queue.put(job)
        logger.info(f"[QUEUE] enqueued {job.original_name}")

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            self.current = job.file_fingerprint
            if job.file_fingerprint in self.pending:
                self.pending.remove(job.file_fingerprint)
            try:
                logger.info(f"[QUEUE] starting {job.original_name}")
                # fn() is sync + CPU/IO heavy — off the event loop, but the
                # queue guarantees only one runs at a time regardless.
                await loop.run_in_executor(None, job.fn)
                logger.info(f"[QUEUE] finished {job.original_name}")
            except Exception as e:
                logger.error(f"[QUEUE] failed {job.original_name}: {e}")
            finally:
                self.current = None
                self._queue.task_done()

    def status(self) -> dict:
        return {
            "current": self.current,
            "pending": list(self.pending),
            "pending_count": len(self.pending),
        }


file_processing_queue = SingleWorkerQueue()
