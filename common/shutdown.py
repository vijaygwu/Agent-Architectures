from __future__ import annotations

"""
Graceful Shutdown Infrastructure for Agent Systems
===================================================

Provides production-ready shutdown patterns:
- GracefulShutdown: Signal handling for SIGTERM/SIGINT with asyncio.Event
- ResourceManager: Track and cleanup async tasks with timeout
- Context managers for clean resource lifecycle

These patterns are essential for production deployments where agents need to:
- Handle Kubernetes pod termination gracefully
- Complete in-flight requests before shutdown
- Clean up resources (connections, files, locks) properly
- Avoid data loss during restarts or scaling events

Shutdown Flow
-------------
1. Signal received (SIGTERM from K8s, SIGINT from Ctrl+C)
2. GracefulShutdown sets event and notifies callbacks
3. Application stops accepting new work
4. ResourceManager waits for in-flight tasks (with timeout)
5. Cleanup callbacks run (close connections, flush buffers)
6. Application exits cleanly

Usage:
    from common.shutdown import GracefulShutdown, ResourceManager

    # Basic signal handling
    shutdown = GracefulShutdown()

    async def main():
        async with shutdown:
            while not shutdown.is_shutting_down:
                await process_work()

    # With resource tracking
    resources = ResourceManager(shutdown_timeout=30.0)

    async def handle_request(request):
        async with resources.track_task("request-123"):
            await process_request(request)

    # In shutdown handler
    await resources.shutdown()  # Waits for tracked tasks
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, Final, Protocol, runtime_checkable

__all__ = [
    "CleanupProtocol",
    "GracefulShutdown",
    "ResourceManager",
    "managed_resources",
    "ShutdownTimeout",
]

logger = logging.getLogger(__name__)

# Default signals to handle for graceful shutdown
DEFAULT_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGTERM, signal.SIGINT)

# Default timeout values (in seconds)
DEFAULT_SHUTDOWN_TIMEOUT: Final[float] = 30.0
DEFAULT_CANCEL_TIMEOUT: Final[float] = 5.0

# Polling interval for shutdown wait loop (in seconds)
SHUTDOWN_POLL_INTERVAL: Final[float] = 0.5


@runtime_checkable
class CleanupProtocol(Protocol):
    """Protocol for resources that need cleanup."""
    async def cleanup(self) -> None: ...


class ShutdownTimeout(Exception):
    """
    Exception raised when shutdown timeout is exceeded.

    Indicates that some tasks did not complete within the allowed
    shutdown window, which may indicate hung tasks or too-short timeout.
    """

    def __init__(self, pending_count: int, timeout: float):
        self.pending_count = pending_count
        self.timeout = timeout
        super().__init__(
            f"Shutdown timeout ({timeout}s) exceeded with {pending_count} tasks pending"
        )


class GracefulShutdown:
    """
    Signal handler for graceful shutdown in async applications.

    Captures SIGTERM and SIGINT signals and sets an asyncio.Event
    that application code can await. Supports cleanup callbacks
    that run before the application exits.

    This pattern is essential for:
    - Kubernetes deployments (SIGTERM on pod termination)
    - Docker containers (SIGTERM on docker stop)
    - Interactive development (SIGINT from Ctrl+C)

    Args:
        signals: Signals to handle (default: SIGTERM, SIGINT)
        logger: Logger for shutdown messages

    Example:
        shutdown = GracefulShutdown()

        # Register cleanup callback
        shutdown.on_shutdown(lambda: print("Cleaning up..."))

        async def main():
            # Start signal handling
            async with shutdown:
                # Main loop - exits when signal received
                while not shutdown.is_shutting_down:
                    await asyncio.sleep(1)
                    await do_work()

                # Cleanup happens automatically via callbacks

        # Or check manually
        async def worker():
            while True:
                if shutdown.is_shutting_down:
                    break
                await process_item()
    """

    def __init__(
        self,
        signals: tuple[signal.Signals, ...] = DEFAULT_SIGNALS,
        log: logging.Logger | None = None
    ) -> None:
        """
        Initialize graceful shutdown handler.

        Args:
            signals: Tuple of signals to capture
            log: Logger for shutdown events
        """
        self.signals = signals
        self.log = log or logger
        self._shutdown_event = asyncio.Event()
        self._shutdown_callbacks: list[Callable[[], Awaitable[None] | None]] = []
        self._signal_received: signal.Signals | None = None
        self._original_handlers: dict[signal.Signals, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._shutdown_event.is_set()

    @property
    def shutdown_event(self) -> asyncio.Event:
        """Get the shutdown event for awaiting."""
        return self._shutdown_event

    async def wait_for_shutdown(self) -> None:
        """
        Wait until shutdown signal is received.

        Useful for simple main loops that just need to wait.

        Example:
            async def main():
                async with shutdown:
                    # Start background tasks
                    asyncio.create_task(worker())

                    # Wait for shutdown
                    await shutdown.wait_for_shutdown()
        """
        await self._shutdown_event.wait()

    def on_shutdown(
        self,
        callback: Callable[[], Awaitable[None] | None]
    ) -> None:
        """
        Register a callback to run during shutdown.

        Callbacks are run in reverse registration order (LIFO)
        to properly unwind resource dependencies.

        Args:
            callback: Sync or async function to call during shutdown

        Example:
            shutdown.on_shutdown(lambda: db.close())
            shutdown.on_shutdown(async_cleanup_function)
        """
        self._shutdown_callbacks.append(callback)

    def _handle_signal(self, sig: signal.Signals) -> None:
        """
        Internal signal handler.

        Sets the shutdown event and logs the received signal.
        """
        self._signal_received = sig
        self.log.info(
            f"Received {sig.name}, initiating graceful shutdown..."
        )
        self._shutdown_event.set()

    def _install_handlers(self) -> None:
        """Install signal handlers, saving originals for restoration."""
        self._loop = asyncio.get_running_loop()

        for sig in self.signals:
            # Save original handler
            self._original_handlers[sig] = signal.getsignal(sig)

            # Install our handler
            # Use add_signal_handler for proper async integration
            try:
                self._loop.add_signal_handler(
                    sig,
                    lambda s=sig: self._handle_signal(s)
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for all signals
                signal.signal(
                    sig,
                    lambda s, f, sig=sig: self._handle_signal(sig)  # type: ignore[arg-type]
                )

    def _remove_handlers(self) -> None:
        """Remove our signal handlers and restore originals."""
        for sig, original in self._original_handlers.items():
            try:
                if self._loop:
                    self._loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

            # Restore original handler
            if original is not None:
                try:
                    signal.signal(sig, original)
                except (ValueError, OSError):
                    pass

        self._original_handlers.clear()

    async def _run_callbacks(self) -> None:
        """Run shutdown callbacks in reverse order."""
        # Run in reverse order (LIFO) for proper dependency unwinding
        for callback in reversed(self._shutdown_callbacks):
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self.log.error(f"Error in shutdown callback: {e}")

    async def __aenter__(self) -> "GracefulShutdown":
        """
        Async context manager entry - install signal handlers.

        Example:
            async with GracefulShutdown() as shutdown:
                while not shutdown.is_shutting_down:
                    await do_work()
        """
        self._install_handlers()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any
    ) -> bool:
        """
        Async context manager exit - run callbacks and restore handlers.

        Returns False to propagate any exceptions.
        """
        # Run cleanup callbacks
        await self._run_callbacks()

        # Restore original signal handlers
        self._remove_handlers()

        return False

    def trigger_shutdown(self) -> None:
        """
        Manually trigger shutdown (for testing or programmatic shutdown).

        Example:
            # In a health check handler that detects fatal condition
            if health_check_failed():
                shutdown.trigger_shutdown()
        """
        self.log.info("Programmatic shutdown triggered")
        self._shutdown_event.set()


@dataclass
class TrackedTask:
    """
    Internal representation of a tracked async task.

    Stores metadata about the task for monitoring and debugging.
    """
    task_id: str
    task: asyncio.Task | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


class ResourceManager:
    """
    Manager for tracking async tasks and ensuring cleanup during shutdown.

    Provides:
    - Task tracking with unique IDs for monitoring
    - Graceful shutdown with configurable timeout
    - Context manager for automatic task lifecycle

    Essential for ensuring all in-flight work completes before shutdown,
    preventing data loss and ensuring consistent state.

    Args:
        shutdown_timeout: Seconds to wait for tasks during shutdown
        log: Logger for resource management events

    Example:
        resources = ResourceManager(shutdown_timeout=30.0)

        # Track a task
        async def handle_request(request):
            async with resources.track_task(f"request-{request.id}"):
                await process_request(request)

        # During shutdown
        await resources.shutdown()  # Waits up to 30s for tasks

        # Or use with GracefulShutdown
        shutdown = GracefulShutdown()
        shutdown.on_shutdown(resources.shutdown)
    """

    def __init__(
        self,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        log: logging.Logger | None = None
    ) -> None:
        """
        Initialize resource manager.

        Args:
            shutdown_timeout: Maximum seconds to wait during shutdown
            log: Logger for events
        """
        self.shutdown_timeout = shutdown_timeout
        self.log = log or logger
        self._tracked_tasks: dict[str, TrackedTask] = {}
        self._lock = asyncio.Lock()
        self._shutting_down = False

    @property
    def active_task_count(self) -> int:
        """Get number of currently tracked tasks."""
        return len(self._tracked_tasks)

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down

    def get_active_tasks(self) -> list[dict[str, Any]]:
        """
        Get information about all active tasks.

        Useful for monitoring and debugging.

        Returns:
            List of task info dicts
        """
        return [
            {
                "task_id": task_id,
                "started_at": tracked.started_at.isoformat(),
                "description": tracked.description,
                "running_seconds": (
                    datetime.now(timezone.utc) - tracked.started_at
                ).total_seconds(),
            }
            for task_id, tracked in self._tracked_tasks.items()
        ]

    async def register_task(
        self,
        task_id: str,
        task: asyncio.Task | None = None,
        description: str = ""
    ) -> None:
        """
        Register a task for tracking.

        Args:
            task_id: Unique identifier for the task
            task: Optional asyncio.Task object
            description: Human-readable description

        Raises:
            RuntimeError: If shutdown is in progress
        """
        if self._shutting_down:
            raise RuntimeError("Cannot register task during shutdown")

        async with self._lock:
            self._tracked_tasks[task_id] = TrackedTask(
                task_id=task_id,
                task=task,
                description=description
            )
            self.log.debug(f"Registered task: {task_id}")

    async def unregister_task(self, task_id: str) -> None:
        """
        Unregister a completed task.

        Args:
            task_id: ID of task to unregister
        """
        async with self._lock:
            if task_id in self._tracked_tasks:
                del self._tracked_tasks[task_id]
                self.log.debug(f"Unregistered task: {task_id}")

    @asynccontextmanager
    async def track_task(self, task_id: str, description: str = "") -> AsyncIterator[None]:
        """
        Context manager for automatically tracking task lifecycle.

        Registers the task on entry, unregisters on exit (even if exception).

        Args:
            task_id: Unique identifier for the task
            description: Human-readable description

        Example:
            async with resources.track_task("request-123", "Processing user request"):
                await handle_request()
        """
        await self.register_task(task_id, description=description)
        try:
            yield
        finally:
            await self.unregister_task(task_id)

    async def shutdown(self, timeout: float | None = None) -> None:
        """
        Gracefully shutdown, waiting for tracked tasks to complete.

        Args:
            timeout: Override default timeout (seconds)

        Raises:
            ShutdownTimeout: If timeout exceeded with tasks still pending
        """
        self._shutting_down = True
        effective_timeout = timeout if timeout is not None else self.shutdown_timeout

        self.log.info(
            f"Shutdown initiated, waiting up to {effective_timeout}s "
            f"for {len(self._tracked_tasks)} tasks"
        )

        start_time = asyncio.get_event_loop().time()

        # Wait for tasks to complete
        while self._tracked_tasks:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= effective_timeout:
                pending = len(self._tracked_tasks)
                self.log.warning(
                    f"Shutdown timeout reached with {pending} tasks pending: "
                    f"{list(self._tracked_tasks.keys())}"
                )
                raise ShutdownTimeout(pending, effective_timeout)

            # Log progress periodically
            remaining = effective_timeout - elapsed
            self.log.debug(
                f"Waiting for {len(self._tracked_tasks)} tasks, "
                f"{remaining:.1f}s remaining"
            )

            await asyncio.sleep(SHUTDOWN_POLL_INTERVAL)

        self.log.info("All tasks completed, shutdown complete")

    async def cancel_all(self, timeout: float = DEFAULT_CANCEL_TIMEOUT) -> int:
        """
        Cancel all tracked tasks (forceful shutdown).

        Use this as a fallback when graceful shutdown times out.

        Args:
            timeout: Seconds to wait for cancellation to complete

        Returns:
            Number of tasks that were cancelled
        """
        cancelled = 0

        async with self._lock:
            for task_id, tracked in list(self._tracked_tasks.items()):
                if tracked.task and not tracked.task.done():
                    tracked.task.cancel()
                    cancelled += 1
                    self.log.warning(f"Cancelled task: {task_id}")

        # Wait briefly for cancellations to propagate
        if cancelled > 0:
            await asyncio.sleep(min(timeout, 1.0))

        return cancelled


@asynccontextmanager
async def managed_resources(
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    handle_signals: bool = True
) -> AsyncIterator[tuple[GracefulShutdown | None, ResourceManager]]:
    """
    Combined context manager for shutdown handling and resource management.

    Provides a convenient way to set up both signal handling and resource
    tracking in one context manager.

    Args:
        shutdown_timeout: Seconds to wait for tasks during shutdown
        handle_signals: Whether to install signal handlers

    Yields:
        Tuple of (GracefulShutdown, ResourceManager)

    Example:
        async def main():
            async with managed_resources(shutdown_timeout=30) as (shutdown, resources):
                # Start workers
                for i in range(3):
                    asyncio.create_task(worker(i, shutdown, resources))

                # Wait for shutdown signal
                await shutdown.wait_for_shutdown()

        async def worker(worker_id, shutdown, resources):
            while not shutdown.is_shutting_down:
                async with resources.track_task(f"worker-{worker_id}-task"):
                    await process_item()
    """
    shutdown = GracefulShutdown() if handle_signals else None
    resources = ResourceManager(shutdown_timeout=shutdown_timeout)

    # Register resource shutdown as cleanup callback
    if shutdown:
        shutdown.on_shutdown(resources.shutdown)

    try:
        if shutdown:
            async with shutdown:
                yield shutdown, resources
        else:
            yield None, resources
    finally:
        # Ensure shutdown runs even without signal handling
        if not shutdown and not resources.is_shutting_down:
            try:
                await resources.shutdown()
            except ShutdownTimeout:
                await resources.cancel_all()
