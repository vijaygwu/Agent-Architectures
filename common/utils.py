"""
Common utilities used across Book 1 code examples.
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime as ISO 8601 string."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.

    Note: Prefer `with_retry` for new code - it provides more configuration
    options and better logging. This function is retained for backward
    compatibility.
    """
    return with_retry(
        max_attempts=max_attempts,
        base_delay=delay,
        exponential_base=backoff,
        retryable_exceptions=exceptions
    )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exceptions that trigger a retry

    Example:
        @with_retry(max_attempts=3, base_delay=1.0)
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # Should never reach here
        return wrapper
    return decorator


class Timer:
    """Context manager for timing code execution."""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0

    async def __aenter__(self):
        self.start_time = asyncio.get_running_loop().time()
        return self

    async def __aexit__(self, *args):
        self.end_time = asyncio.get_running_loop().time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    logger_name: str | None = None
) -> logging.Logger:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output logs as JSON; otherwise use standard format
        logger_name: Name for the logger; None for root logger

    Returns:
        Configured logger instance

    Example:
        logger = configure_logging(level="INFO", json_output=True)
        logger.info("Task started", extra={"extra_fields": {"task_id": "123"}})
    """
    log = logging.getLogger(logger_name)
    log.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    log.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    log.addHandler(handler)
    return log


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# =============================================================================
# OpenTelemetry Tracing Support
# =============================================================================

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

# Try to import OTLP exporter (optional)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    OTLPSpanExporter = None


def configure_tracing(
    service_name: str,
    endpoint: str | None = None,
    console_export: bool = False
) -> Any:
    """
    Configure OpenTelemetry distributed tracing.

    Args:
        service_name: Name of this service (e.g., 'orchestrator', 'council')
        endpoint: OTLP collector endpoint (e.g., 'http://localhost:4317')
        console_export: If True, also export spans to console (for debugging)

    Returns:
        Tracer instance, or None if OpenTelemetry not available

    Example:
        tracer = configure_tracing('orchestrator', endpoint='http://jaeger:4317')
        with tracer.start_as_current_span('my_operation') as span:
            span.set_attribute('task.id', task_id)
            # ... do work ...
    """
    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry not available. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )
        return None

    provider = TracerProvider()

    # Add OTLP exporter if endpoint provided and available
    if endpoint and OTLP_AVAILABLE:
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    elif endpoint:
        logger.warning(
            "OTLP exporter not available. Install with: "
            "pip install opentelemetry-exporter-otlp"
        )

    # Add console exporter for debugging
    if console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer(name: str) -> Any:
    """
    Get a tracer instance for the given service name.

    Returns a no-op tracer if OpenTelemetry is not installed,
    allowing code to use tracing without requiring the dependency.

    Args:
        name: Service/component name for the tracer

    Returns:
        Tracer instance (real or no-op)

    Example:
        tracer = get_tracer('orchestrator')
        with tracer.start_as_current_span('process_task') as span:
            span.set_attribute('task.id', '123')
    """
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return NoOpTracer()


class NoOpSpan:
    """No-op span for when OpenTelemetry is not installed."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op: Set span attribute."""
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """No-op: Add event to span."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op: Set span status."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op: Record exception."""
        pass


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        """Return a no-op span context manager."""
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        """Return a no-op span."""
        return NoOpSpan()
