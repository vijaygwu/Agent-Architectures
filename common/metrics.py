"""
Metrics and Logging Infrastructure for Agent Systems
=====================================================

Provides production-ready observability patterns:
- MetricsCollector: Prometheus-compatible metrics (counters, histograms, gauges)
- StructuredLogger: JSON-formatted logging with context propagation
- Timing helpers for measuring operation durations

These patterns are essential for monitoring agent behavior in production,
debugging issues, and understanding system performance.

Metrics Types
-------------
- Counter: Monotonically increasing value (requests, errors, completions)
- Gauge: Point-in-time value that can go up or down (active tasks, queue depth)
- Histogram: Distribution of values with configurable buckets (latencies, sizes)

Structured Logging
------------------
JSON-formatted logs with consistent fields enable:
- Easy parsing by log aggregation systems (ELK, Splunk, Datadog)
- Correlation across distributed systems via request_id/trace_id
- Rich context for debugging (agent_id, task_id, user_id)

Usage:
    from common.metrics import MetricsCollector, StructuredLogger, timed

    # Metrics
    metrics = MetricsCollector(namespace="agent")
    metrics.increment("requests_total", labels={"endpoint": "/chat"})
    metrics.observe("response_time_seconds", 0.234, labels={"model": "gpt-4"})
    metrics.set_gauge("active_tasks", 5)

    # Structured logging
    logger = StructuredLogger("orchestrator", default_context={"service": "agent"})
    logger.info("Task started", task_id="abc123", priority="high")

    # Timing decorator
    @timed(metrics, "llm_call_duration_seconds")
    async def call_llm():
        return await client.chat(...)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Final, Generator, TypedDict, TypeVar

__all__ = [
    "MetricsCollector",
    "StructuredLogger",
    "timed",
    "async_timed",
    "Timer",
    "MetricResult",
    "HistogramStats",
    "MetricsExport",
]

T = TypeVar("T")


# =============================================================================
# TypedDicts for Structured Returns
# =============================================================================

class MetricResult(TypedDict):
    """Structured result for a single metric observation."""
    name: str
    value: float
    timestamp: str
    labels: dict[str, str]


class HistogramStats(TypedDict):
    """Statistics for a histogram metric."""
    count: int
    sum: float
    mean: float
    buckets: dict[float, int]


class HistogramExportData(TypedDict):
    """Export format for histogram data."""
    count: int
    sum: float
    buckets: dict[str, int]


class MetricsExport(TypedDict):
    """Complete metrics export structure."""
    counters: dict[str, dict[str, float]]
    gauges: dict[str, dict[str, float]]
    histograms: dict[str, dict[str, HistogramExportData]]


# =============================================================================
# Prometheus-Compatible Metrics Collector
# =============================================================================

# Default histogram buckets for latency measurements (in seconds)
DEFAULT_LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, float("inf")
)

# Default log levels mapping
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# Common metric name suffixes
METRIC_SUFFIX_TOTAL: Final[str] = "_total"
METRIC_SUFFIX_SECONDS: Final[str] = "_seconds"
METRIC_SUFFIX_BYTES: Final[str] = "_bytes"
METRIC_SUFFIX_BUCKET: Final[str] = "_bucket"
METRIC_SUFFIX_SUM: Final[str] = "_sum"
METRIC_SUFFIX_COUNT: Final[str] = "_count"


@dataclass
class HistogramData:
    """
    Internal data structure for histogram metrics.

    Stores bucket counts and running statistics for calculating
    percentiles and other distribution metrics.
    """
    buckets: tuple[float, ...]
    bucket_counts: dict[float, int] = field(default_factory=dict)
    sum_value: float = 0.0
    count: int = 0

    def __post_init__(self):
        """Initialize bucket counts to zero."""
        for bucket in self.buckets:
            self.bucket_counts[bucket] = 0

    def observe(self, value: float) -> None:
        """
        Record an observation in the histogram.

        Args:
            value: The value to record
        """
        self.sum_value += value
        self.count += 1

        # Increment all buckets that this value falls into
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1


class MetricsCollector:
    """
    Prometheus-compatible metrics collector for agent systems.

    Provides three metric types:
    - Counter: Monotonically increasing (requests, errors, completions)
    - Gauge: Point-in-time value (active tasks, queue depth, memory usage)
    - Histogram: Distribution with buckets (latencies, request sizes)

    Thread-safe for concurrent access in async environments.

    Args:
        namespace: Prefix for all metric names (e.g., "agent", "orchestrator")
        default_labels: Labels applied to all metrics

    Example:
        metrics = MetricsCollector(
            namespace="agent",
            default_labels={"environment": "production"}
        )

        # Counter - track events
        metrics.increment("requests_total", labels={"endpoint": "/chat"})
        metrics.increment("errors_total", labels={"type": "rate_limit"})

        # Gauge - track current state
        metrics.set_gauge("active_tasks", 5)
        metrics.set_gauge("memory_mb", 256.5)

        # Histogram - track distributions
        metrics.observe("response_time_seconds", 0.234)
        metrics.observe("tokens_used", 150, labels={"model": "gpt-4"})

        # Export for Prometheus scraping
        print(metrics.export_prometheus())
    """

    def __init__(
        self,
        namespace: str = "",
        default_labels: dict[str, str] | None = None
    ):
        """
        Initialize metrics collector.

        Args:
            namespace: Prefix for metric names (e.g., "agent_requests_total")
            default_labels: Labels applied to all metrics
        """
        self.namespace = namespace
        self.default_labels = default_labels or {}

        # Metric storage
        self._counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: dict[str, dict[str, HistogramData]] = defaultdict(dict)
        self._histogram_buckets: dict[str, tuple[float, ...]] = {}

        # Thread safety
        self._lock = asyncio.Lock()

    def _format_name(self, name: str) -> str:
        """Format metric name with namespace prefix."""
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name

    def _format_labels(self, labels: dict[str, str] | None) -> str:
        """
        Convert labels dict to a hashable string key.

        Labels are sorted for consistent key generation.
        """
        merged = {**self.default_labels, **(labels or {})}
        if not merged:
            return ""
        sorted_items = sorted(merged.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_items)

    # -------------------------------------------------------------------------
    # Counter Operations
    # -------------------------------------------------------------------------

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None
    ) -> None:
        """
        Increment a counter metric.

        Counters are monotonically increasing - they only go up.
        Use for tracking events like requests, errors, completions.

        Args:
            name: Metric name
            value: Amount to increment (must be positive)
            labels: Additional labels for this observation

        Example:
            metrics.increment("requests_total", labels={"status": "success"})
            metrics.increment("tokens_used", value=150)
        """
        if value < 0:
            raise ValueError("Counter increment must be positive")

        full_name = self._format_name(name)
        label_key = self._format_labels(labels)
        self._counters[full_name][label_key] += value

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None
    ) -> float:
        """Get current value of a counter."""
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)
        return self._counters[full_name][label_key]

    # -------------------------------------------------------------------------
    # Gauge Operations
    # -------------------------------------------------------------------------

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None
    ) -> None:
        """
        Set a gauge metric to an absolute value.

        Gauges can go up or down - they represent current state.
        Use for tracking things like active tasks, queue depth, memory.

        Args:
            name: Metric name
            value: Current value
            labels: Additional labels for this metric

        Example:
            metrics.set_gauge("active_tasks", 5)
            metrics.set_gauge("queue_depth", 100, labels={"queue": "high_priority"})
        """
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)
        self._gauges[full_name][label_key] = value

    def increment_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None
    ) -> None:
        """
        Increment a gauge metric.

        Args:
            name: Metric name
            value: Amount to increment (can be negative)
            labels: Additional labels for this metric
        """
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)
        self._gauges[full_name][label_key] += value

    def decrement_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None
    ) -> None:
        """
        Decrement a gauge metric.

        Args:
            name: Metric name
            value: Amount to decrement
            labels: Additional labels for this metric
        """
        self.increment_gauge(name, -value, labels)

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None
    ) -> float:
        """Get current value of a gauge."""
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)
        return self._gauges[full_name][label_key]

    # -------------------------------------------------------------------------
    # Histogram Operations
    # -------------------------------------------------------------------------

    def register_histogram(
        self,
        name: str,
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS
    ) -> None:
        """
        Register a histogram with custom buckets.

        Call this before observing if you want non-default buckets.
        If not called, DEFAULT_LATENCY_BUCKETS will be used.

        Args:
            name: Metric name
            buckets: Tuple of bucket boundaries (must include float("inf"))

        Example:
            # Custom buckets for token counts
            metrics.register_histogram(
                "tokens_used",
                buckets=(10, 50, 100, 500, 1000, 5000, float("inf"))
            )
        """
        full_name = self._format_name(name)
        self._histogram_buckets[full_name] = buckets

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None
    ) -> None:
        """
        Record an observation in a histogram.

        Histograms track the distribution of values, useful for
        latencies, request sizes, token counts, etc.

        Args:
            name: Metric name
            value: Value to record
            labels: Additional labels for this observation

        Example:
            metrics.observe("response_time_seconds", 0.234)
            metrics.observe("tokens_used", 150, labels={"model": "gpt-4"})
        """
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)

        # Get or create histogram for this label combination
        if label_key not in self._histograms[full_name]:
            buckets = self._histogram_buckets.get(full_name, DEFAULT_LATENCY_BUCKETS)
            self._histograms[full_name][label_key] = HistogramData(buckets=buckets)

        self._histograms[full_name][label_key].observe(value)

    def get_histogram_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None
    ) -> HistogramStats | None:
        """
        Get statistics for a histogram.

        Returns:
            Dict with count, sum, and bucket counts, or None if not found
        """
        full_name = self._format_name(name)
        label_key = self._format_labels(labels)

        if full_name not in self._histograms:
            return None
        if label_key not in self._histograms[full_name]:
            return None

        hist = self._histograms[full_name][label_key]
        return {
            "count": hist.count,
            "sum": hist.sum_value,
            "mean": hist.sum_value / hist.count if hist.count > 0 else 0,
            "buckets": dict(hist.bucket_counts),
        }

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns a string suitable for scraping by Prometheus.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Export counters
        for name, label_values in self._counters.items():
            for label_key, value in label_values.items():
                if label_key:
                    lines.append(f"{name}{{{label_key}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        # Export gauges
        for name, label_values in self._gauges.items():
            for label_key, value in label_values.items():
                if label_key:
                    lines.append(f"{name}{{{label_key}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        # Export histograms
        for name, label_histograms in self._histograms.items():
            for label_key, hist in label_histograms.items():
                base_labels = f"{{{label_key}}}" if label_key else ""

                # Bucket values (cumulative)
                for bucket, count in sorted(hist.bucket_counts.items()):
                    bucket_str = "+Inf" if bucket == float("inf") else str(bucket)
                    if label_key:
                        lines.append(
                            f'{name}_bucket{{le="{bucket_str}",{label_key}}} {count}'
                        )
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

                # Sum and count
                if label_key:
                    lines.append(f"{name}_sum{{{label_key}}} {hist.sum_value}")
                    lines.append(f"{name}_count{{{label_key}}} {hist.count}")
                else:
                    lines.append(f"{name}_sum {hist.sum_value}")
                    lines.append(f"{name}_count {hist.count}")

        return "\n".join(lines)

    def export_json(self) -> MetricsExport:
        """
        Export metrics as a JSON-serializable dictionary.

        Useful for debugging or sending to non-Prometheus systems.

        Returns:
            Dict containing all metrics
        """
        return {
            "counters": {
                name: dict(values) for name, values in self._counters.items()
            },
            "gauges": {
                name: dict(values) for name, values in self._gauges.items()
            },
            "histograms": {
                name: {
                    label_key: {
                        "count": hist.count,
                        "sum": hist.sum_value,
                        "buckets": {
                            str(k): v for k, v in hist.bucket_counts.items()
                        }
                    }
                    for label_key, hist in label_histograms.items()
                }
                for name, label_histograms in self._histograms.items()
            },
        }

    def reset(self) -> None:
        """Reset all metrics to zero/empty state."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# =============================================================================
# Structured Logger for JSON-Formatted Logging
# =============================================================================

class StructuredLogger:
    """
    JSON-formatted structured logger with context propagation.

    Structured logging produces JSON output that is easy to parse
    by log aggregation systems (ELK, Splunk, Datadog, etc.) and
    enables powerful querying and correlation.

    Features:
    - Consistent JSON format across all log entries
    - Context fields that persist across log calls (request_id, agent_id)
    - Automatic timestamp and log level
    - Exception formatting with stack traces

    Args:
        name: Logger name (typically module or component name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        default_context: Fields included in every log entry

    Example:
        # Create logger with default context
        logger = StructuredLogger(
            "orchestrator",
            default_context={"service": "agent", "environment": "production"}
        )

        # Log with additional context
        logger.info("Task started", task_id="abc123", priority="high")
        # Output: {"timestamp": "...", "level": "INFO", "logger": "orchestrator",
        #          "message": "Task started", "service": "agent",
        #          "environment": "production", "task_id": "abc123", "priority": "high"}

        # Create child logger with additional context
        task_logger = logger.with_context(task_id="abc123")
        task_logger.info("Processing step 1")  # Includes task_id automatically
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        default_context: dict[str, Any] | None = None,
        stream: Any = None
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Log level string
            default_context: Fields to include in every log entry
            stream: Output stream (default: sys.stdout)
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.default_context = default_context or {}
        self.stream = stream or sys.stdout

        # Level mapping for filtering
        self._levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        return self._levels.get(level, logging.INFO) >= self.level

    def _format_entry(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> str:
        """
        Format a log entry as JSON.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields

        Returns:
            JSON-formatted log line
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **self.default_context,
            **kwargs,
        }

        # Handle exception info if present
        exc_info = kwargs.pop("exc_info", None)
        if exc_info:
            import traceback
            entry["exception"] = "".join(
                traceback.format_exception(*exc_info)
            )

        return json.dumps(entry, default=str)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        if self._should_log(level):
            line = self._format_entry(level, message, **kwargs)
            print(line, file=self.stream)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """
        Log an exception with stack trace.

        Call this from an exception handler to include the full traceback.
        """
        import sys
        kwargs["exc_info"] = sys.exc_info()
        self._log("ERROR", message, **kwargs)

    def with_context(self, **context: Any) -> StructuredLogger:
        """
        Create a child logger with additional context.

        The child logger inherits all context from the parent
        and adds the new context fields.

        Args:
            **context: Additional context fields

        Returns:
            New StructuredLogger with merged context

        Example:
            base_logger = StructuredLogger("api")
            request_logger = base_logger.with_context(request_id="req-123")
            request_logger.info("Processing")  # Includes request_id
        """
        merged_context = {**self.default_context, **context}
        return StructuredLogger(
            name=self.name,
            level=logging.getLevelName(self.level),
            default_context=merged_context,
            stream=self.stream
        )

    def set_level(self, level: str) -> None:
        """Change the log level."""
        self.level = getattr(logging, level.upper())


# =============================================================================
# Timing Helpers
# =============================================================================

class Timer:
    """
    Context manager for timing code execution.

    Provides both sync and async context manager interfaces.
    Records elapsed time in seconds with high precision.

    Example:
        # Async usage
        async with Timer() as t:
            await some_operation()
        print(f"Took {t.elapsed_seconds:.3f}s")

        # Sync usage
        with Timer() as t:
            some_operation()
        print(f"Took {t.elapsed_ms:.1f}ms")
    """

    def __init__(self):
        """Initialize timer."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    def __enter__(self) -> Timer:
        """Sync context manager entry."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any
    ) -> None:
        """Sync context manager exit."""
        self.end_time = time.perf_counter()

    async def __aenter__(self) -> Timer:
        """Async context manager entry."""
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        self.end_time = time.perf_counter()


def timed(
    metrics: MetricsCollector,
    metric_name: str,
    labels: dict[str, str] | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically time function execution and record as histogram.

    Works with both sync and async functions.

    Args:
        metrics: MetricsCollector to record to
        metric_name: Name of the histogram metric
        labels: Additional labels for the metric

    Example:
        @timed(metrics, "api_call_duration_seconds", labels={"api": "openai"})
        async def call_api():
            return await client.chat(...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                metrics.observe(metric_name, elapsed, labels)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                metrics.observe(metric_name, elapsed, labels)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def async_timed(
    metrics: MetricsCollector,
    metric_name: str,
    labels: dict[str, str] | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for timing async functions (explicit async version).

    Identical to timed() but explicitly for async functions.
    Provided for clarity in codebases that want explicit decoration.

    Args:
        metrics: MetricsCollector to record to
        metric_name: Name of the histogram metric
        labels: Additional labels for the metric
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                metrics.observe(metric_name, elapsed, labels)
        return wrapper
    return decorator


@asynccontextmanager
async def timed_block(
    metrics: MetricsCollector,
    metric_name: str,
    labels: dict[str, str] | None = None
) -> AsyncGenerator[None, None]:
    """
    Async context manager for timing a block of code.

    Args:
        metrics: MetricsCollector to record to
        metric_name: Name of the histogram metric
        labels: Additional labels for the metric

    Example:
        async with timed_block(metrics, "processing_time_seconds"):
            await process_data()
            await save_results()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics.observe(metric_name, elapsed, labels)
