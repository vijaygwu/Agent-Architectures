"""
Common utilities for Book 1: Design Patterns & Implementation

This package provides shared infrastructure for production-ready agent systems:

- agent_types: Core data types (AgentMessage, AgentState, TaskResult)
- utils: General utilities (generate_id, with_retry, format_timestamp, logging, tracing)
- resilience: Resilience patterns (CircuitBreaker, RateLimiter, RetryWithBackoff)
- metrics: Observability (MetricsCollector, StructuredLogger, timing helpers)
- shutdown: Graceful shutdown (GracefulShutdown, ResourceManager)
"""

from .agent_types import AgentMessage, AgentState, TaskResult
from .utils import generate_id, with_retry, format_timestamp

# Resilience patterns
from .resilience import (
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreaker,
    CircuitBreakerRegistry,
    RateLimiter,
    RateLimitExceeded,
    retry_with_backoff,
    RetryWithBackoff,
    RetryConfig,
    RetryExhausted,
)

# Metrics and logging
from .metrics import (
    MetricsCollector,
    StructuredLogger,
    timed,
    async_timed,
    Timer,
)

# Graceful shutdown
from .shutdown import (
    GracefulShutdown,
    ResourceManager,
    managed_resources,
    ShutdownTimeout,
)

__all__ = [
    # Agent types
    "AgentMessage",
    "AgentState",
    "TaskResult",
    # Utils
    "generate_id",
    "with_retry",
    "format_timestamp",
    # Resilience
    "CircuitState",
    "CircuitBreakerOpen",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "RateLimiter",
    "RateLimitExceeded",
    "retry_with_backoff",
    "RetryWithBackoff",
    "RetryConfig",
    "RetryExhausted",
    # Metrics
    "MetricsCollector",
    "StructuredLogger",
    "timed",
    "async_timed",
    "Timer",
    # Shutdown
    "GracefulShutdown",
    "ResourceManager",
    "managed_resources",
    "ShutdownTimeout",
]
