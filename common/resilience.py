from __future__ import annotations

"""
Common Resilience Patterns
==========================

Provides production-ready resilience patterns for agent systems:
- Circuit Breaker: Fail fast when a service is unavailable
- Rate Limiter: Control request throughput using token bucket algorithm
- Retry with backoff: Handle transient failures

These patterns are essential for building robust agent systems that can handle
API failures, rate limits, and service degradation gracefully.

Circuit Breaker Pattern
-----------------------
The circuit breaker has three states:

1. CLOSED (normal): Requests flow through normally
   - Failures are counted
   - When failure threshold is reached, transitions to OPEN

2. OPEN (failing fast): All requests fail immediately without attempting the call
   - Prevents cascading failures and gives the downstream service time to recover
   - After timeout period, transitions to HALF_OPEN

3. HALF_OPEN (testing recovery): A limited number of test requests are allowed
   - If test requests succeed, transitions back to CLOSED
   - If test requests fail, transitions back to OPEN

Rate Limiter Pattern (Token Bucket)
-----------------------------------
The token bucket algorithm allows controlled bursting while maintaining
a sustainable average rate:
- Bucket holds up to max_tokens tokens (burst capacity)
- Tokens are added at refill_rate per second
- Each request consumes tokens from the bucket
- Requests are blocked/delayed when bucket is empty

Retry Pattern (Exponential Backoff)
----------------------------------
Handles transient failures by progressively increasing the delay between
retries, giving failing services time to recover. Adding jitter prevents
multiple clients from retrying simultaneously (thundering herd problem).

Usage:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
    from common.resilience import RateLimiter, RateLimitExceeded
    from common.resilience import retry_with_backoff

    # Circuit breaker
    breaker = CircuitBreaker(name="api", failure_threshold=5, recovery_timeout=30.0)

    @breaker.protect
    async def call_api():
        return await make_api_call()

    # Rate limiter
    limiter = RateLimiter(name="openai", max_tokens=60, refill_rate=1.0)

    async def call_llm():
        await limiter.acquire()
        return await client.chat(...)
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Final, Generic, TypeVar

# Type variable for generic return types
T = TypeVar('T')

# Default configuration constants
DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
DEFAULT_RECOVERY_TIMEOUT: Final[float] = 30.0
DEFAULT_HALF_OPEN_MAX_CALLS: Final[int] = 3
DEFAULT_MAX_TOKENS: Final[float] = 10.0
DEFAULT_REFILL_RATE: Final[float] = 1.0
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_BASE_DELAY: Final[float] = 1.0
DEFAULT_MAX_DELAY: Final[float] = 60.0
DEFAULT_EXPONENTIAL_BASE: Final[float] = 2.0
DEFAULT_JITTER: Final[float] = 0.1

# Callback type aliases
RetryCallback = Callable[[int, Exception, float], None]
AsyncFunc = Callable[..., Awaitable[T]]

__all__ = [
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerOpen",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    # Rate Limiter
    "RateLimiter",
    "RateLimitExceeded",
    # Retry
    "retry_with_backoff",
    "RetryWithBackoff",
    "RetryConfig",
    "RetryExhausted",
]


class CircuitState(Enum):
    """
    Circuit breaker states.

    The state machine transitions:
    CLOSED -> OPEN (on failure threshold)
    OPEN -> HALF_OPEN (after recovery timeout)
    HALF_OPEN -> CLOSED (on success) or HALF_OPEN -> OPEN (on failure)
    """
    CLOSED: Final[str] = "closed"       # Normal operation, requests pass through
    OPEN: Final[str] = "open"           # Failing fast, no requests allowed
    HALF_OPEN: Final[str] = "half_open" # Testing if service has recovered


class CircuitBreakerOpen(Exception):
    """
    Exception raised when circuit breaker is open.

    This allows callers to handle the "fail fast" case differently
    from actual service failures.
    """
    def __init__(self, breaker_name: str, time_until_retry: float):
        self.breaker_name = breaker_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


@dataclass
class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for resilience against cascading failures.

    This pattern prevents an application from repeatedly trying to execute
    an operation that's likely to fail, allowing it to continue without
    waiting for the fault to be fixed or wasting CPU cycles.

    Args:
        name: Identifier for this circuit breaker (used in logging/metrics)
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Seconds to wait before testing recovery (half-open)
        half_open_max_calls: Max calls allowed in half-open state for testing

    Example:
        # Create a circuit breaker for an external API
        api_breaker = CircuitBreaker(
            name="payment-api",
            failure_threshold=5,
            recovery_timeout=30.0
        )

        # Use as async context manager
        async with api_breaker:
            response = await http_client.post(url, data=payload)

        # Or use as decorator
        @api_breaker.protect
        async def call_payment_api(amount):
            return await http_client.post(...)
    """
    name: str
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT
    half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_CALLS

    # Internal state (not part of initialization)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    def allow(self) -> bool:
        """Check if requests are allowed through the circuit.

        This is a convenience method that returns True if the circuit
        is not in the OPEN state. In HALF_OPEN state, some requests
        are allowed for testing recovery.

        Returns:
            True if requests can proceed, False if circuit is open.
        """
        return not self.is_open

    def record_success(self) -> None:
        """
        Record a successful call.

        In HALF_OPEN state, enough successes will close the circuit.
        In CLOSED state, this resets the failure count.
        """
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            # If enough successes in half-open, close the circuit
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed call.

        In CLOSED state, failures are counted toward the threshold.
        In HALF_OPEN state, any failure reopens the circuit.
        """
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def _check_state(self) -> None:
        """
        Check if calls are allowed in current state.

        Raises:
            CircuitBreakerOpen: If circuit is open and calls are not allowed
        """
        current_state = self.state  # This triggers automatic OPEN->HALF_OPEN

        if current_state == CircuitState.OPEN:
            time_until_retry = (
                self.recovery_timeout - (time.time() - self._last_failure_time)
            )
            raise CircuitBreakerOpen(self.name, max(0, time_until_retry))

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                # Too many test calls in progress, fail fast
                raise CircuitBreakerOpen(self.name, self.recovery_timeout)
            self._half_open_calls += 1

    async def __aenter__(self) -> "CircuitBreaker":
        """
        Async context manager entry - checks if calls are allowed.

        Usage:
            async with circuit_breaker:
                await make_api_call()
        """
        async with self._lock:
            self._check_state()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any
    ) -> bool:
        """
        Async context manager exit - records success or failure.

        Returns False to propagate any exception.
        """
        async with self._lock:
            if exc_type is None:
                self.record_success()
            else:
                # Don't count CircuitBreakerOpen as a failure
                if not isinstance(exc_val, CircuitBreakerOpen):
                    self.record_failure()
        return False  # Don't suppress exceptions

    def protect(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """
        Decorator to protect an async function with this circuit breaker.

        Usage:
            @circuit_breaker.protect
            async def call_api():
                return await http_client.get(url)
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with self:
                return await func(*args, **kwargs)
        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """
        Get current circuit breaker statistics.

        Useful for monitoring and debugging.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_since_last_failure": (
                time.time() - self._last_failure_time
                if self._last_failure_time > 0
                else None
            )
        }

    def reset(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        Use with caution - typically for testing or admin override.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = 0.0


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers by name,
    useful for monitoring and management.

    Usage:
        registry = CircuitBreakerRegistry()

        # Get or create a circuit breaker
        breaker = registry.get_or_create("api-service", failure_threshold=3)

        # Get all breaker stats for monitoring
        all_stats = registry.get_all_stats()
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker[Any]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT,
        half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_CALLS
    ) -> CircuitBreaker[Any]:
        """
        Get an existing circuit breaker or create a new one.

        Thread-safe method to ensure only one breaker per name.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker[Any] | None:
        """Get a circuit breaker by name, or None if not found."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()


# Convenience function for retry with exponential backoff
async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
) -> T:
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Async function to execute (should be a zero-arg callable)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exceptions that should trigger retry

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail

    Example:
        result = await retry_with_backoff(
            lambda: http_client.get(url),
            max_retries=3,
            retryable_exceptions=(httpx.RequestError, TimeoutError)
        )
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                await asyncio.sleep(delay)

    raise last_exception


# =============================================================================
# Rate Limiter (Token Bucket Algorithm)
# =============================================================================

class RateLimitExceeded(Exception):
    """
    Exception raised when rate limit is exceeded.

    This allows callers to handle rate limiting differently from other errors,
    for example by implementing backpressure or queueing.

    Attributes:
        limiter_name: Name of the rate limiter that was exceeded
        retry_after: Seconds until tokens will be available
    """

    def __init__(self, limiter_name: str, retry_after: float):
        self.limiter_name = limiter_name
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for '{limiter_name}', retry after {retry_after:.2f}s"
        )


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for controlling request throughput.

    The token bucket algorithm allows for controlled bursting while
    maintaining a sustainable average rate:
    - Bucket holds up to max_tokens tokens (burst capacity)
    - Tokens are added at refill_rate per second
    - Each request consumes one or more tokens
    - Requests are blocked when bucket is empty

    This is ideal for API rate limiting where you want to allow
    short bursts while preventing sustained overload.

    Args:
        name: Identifier for this limiter (used in logging and errors)
        max_tokens: Maximum tokens in bucket (burst capacity)
        refill_rate: Tokens added per second (sustainable rate)

    Example:
        # Allow bursts of 60 requests, sustainable rate of 1 req/sec
        limiter = RateLimiter(name="openai_api", max_tokens=60, refill_rate=1.0)

        # Simple usage - blocks if no tokens
        if await limiter.acquire():
            await call_api()
        else:
            print(f"Rate limited, retry in {limiter.get_wait_time():.1f}s")

        # Wait for tokens (with timeout)
        if await limiter.acquire_or_wait(timeout=30.0):
            await call_api()

        # Use as decorator
        @limiter.limit
        async def call_api():
            return await client.chat(...)
    """
    name: str
    max_tokens: float = DEFAULT_MAX_TOKENS    # Maximum tokens (burst capacity)
    refill_rate: float = DEFAULT_REFILL_RATE    # Tokens per second

    # Internal state (not part of initialization)
    _tokens: float = field(init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        """Initialize tokens to max capacity."""
        self._tokens = self.max_tokens

    async def _refill(self) -> None:
        """
        Refill tokens based on elapsed time.

        Must be called while holding self._lock.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.max_tokens, self._tokens + tokens_to_add)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        This is a non-blocking operation that returns immediately.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens acquired, False if insufficient tokens
        """
        async with self._lock:
            await self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire_or_wait(
        self,
        tokens: float = 1.0,
        timeout: float | None = None
    ) -> bool:
        """
        Acquire tokens, waiting if necessary.

        This method will block until tokens are available or timeout
        is exceeded. Useful when you want to throttle rather than reject.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if tokens acquired, False if timeout exceeded
        """
        start = time.monotonic()
        while True:
            if await self.acquire(tokens):
                return True

            wait_time = self.get_wait_time(tokens)
            if timeout is not None:
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            await asyncio.sleep(min(wait_time, 0.1))  # Poll at most every 100ms

    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Calculate time until requested tokens will be available.

        Useful for providing retry-after information to clients.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds until tokens available (0 if already available)
        """
        # Estimate current tokens without locking (approximate)
        elapsed = time.monotonic() - self._last_refill
        current_tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)

        if current_tokens >= tokens:
            return 0.0
        tokens_needed = tokens - current_tokens
        return tokens_needed / self.refill_rate

    def get_available_tokens(self) -> float:
        """
        Get current number of available tokens (approximate).

        This is an estimate without locking, useful for monitoring.
        """
        elapsed = time.monotonic() - self._last_refill
        return min(self.max_tokens, self._tokens + elapsed * self.refill_rate)

    def limit(
        self,
        tokens: float = 1.0,
        wait: bool = False,
        timeout: float | None = None
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """
        Decorator to apply rate limiting to an async function.

        Args:
            tokens: Tokens consumed per call (default 1)
            wait: If True, wait for tokens; if False, raise immediately
            timeout: Maximum wait time (only used if wait=True)

        Example:
            @limiter.limit(tokens=1, wait=True, timeout=30)
            async def call_api():
                return await client.request()
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                if wait:
                    acquired = await self.acquire_or_wait(tokens, timeout)
                else:
                    acquired = await self.acquire(tokens)

                if not acquired:
                    raise RateLimitExceeded(self.name, self.get_wait_time(tokens))

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def get_stats(self) -> dict[str, Any]:
        """
        Get current rate limiter statistics.

        Useful for monitoring and debugging.
        """
        return {
            "name": self.name,
            "available_tokens": self.get_available_tokens(),
            "max_tokens": self.max_tokens,
            "refill_rate": self.refill_rate,
            "requests_per_second": self.refill_rate,
        }


# =============================================================================
# Enhanced Retry with Exponential Backoff (Class-based)
# =============================================================================

class RetryExhausted(Exception):
    """
    Exception raised when all retry attempts are exhausted.

    Attributes:
        attempts: Number of attempts made
        last_exception: The final exception that caused the last failure
    """

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted. Last error: {last_exception}"
        )


@dataclass
class RetryConfig:
    """
    Configuration for retry with exponential backoff.

    Attributes:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries (caps exponential growth)
        exponential_base: Multiplier for exponential backoff (typically 2)
        jitter: Random factor to prevent thundering herd (0.0 to 1.0)
        retryable_exceptions: Exception types that should trigger retry

    Example:
        # 5 attempts, starting at 1s delay, doubling up to 30s max
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=0.1  # Add 10% randomness
        )
    """
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE
    jitter: float = DEFAULT_JITTER  # 0.0 to 1.0 - adds randomness to prevent thundering herd
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )


class RetryWithBackoff:
    """
    Class-based retry mechanism with exponential backoff.

    Provides more control than the simple retry_with_backoff function,
    including callbacks for retry events and configurable behavior.

    Exponential backoff progressively increases the delay between retries,
    giving failing services time to recover. Adding jitter prevents multiple
    clients from retrying simultaneously (thundering herd problem).

    Delay calculation:
        delay = min(base_delay * (exponential_base ** attempt), max_delay)
        delay = delay * (1 + random(-jitter, +jitter))

    Example:
        retry = RetryWithBackoff(RetryConfig(max_attempts=3, base_delay=1.0))

        # Use as decorator
        @retry
        async def call_api():
            return await client.request()

        # Or call execute directly
        result = await retry.execute(call_api)

        # With retry callbacks for logging
        retry.on_retry(lambda attempt, exc, delay: print(f"Retry {attempt}: {exc}"))
    """

    def __init__(self, config: RetryConfig | None = None):
        """
        Initialize retry handler.

        Args:
            config: Configuration parameters (uses defaults if not provided)
        """
        self.config: RetryConfig = config or RetryConfig()
        self._on_retry_callbacks: list[RetryCallback] = []

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds (with jitter applied)
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter > 0:
            jitter_range = delay * self.config.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def on_retry(self, callback: RetryCallback) -> None:
        """
        Register a callback for retry events.

        The callback is called before each retry sleep with:
        - attempt: The attempt number (1 for first retry, 2 for second, etc.)
        - exception: The exception that triggered the retry
        - delay: The delay before the next attempt

        Args:
            callback: Function called with (attempt, exception, delay)
        """
        self._on_retry_callbacks.append(callback)

    def _notify_retry(self, attempt: int, exception: Exception, delay: float) -> None:
        """Notify all callbacks of a retry event."""
        for callback in self._on_retry_callbacks:
            try:
                callback(attempt, exception, delay)
            except Exception:
                pass  # Don't let callback errors affect retry logic

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            RetryExhausted: If all attempts fail
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e

                # Don't delay after last attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    self._notify_retry(attempt + 1, e, delay)
                    await asyncio.sleep(delay)

        raise RetryExhausted(self.config.max_attempts, last_exception)

    def __call__(
        self,
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        """
        Use as decorator for async functions.

        Example:
            @RetryWithBackoff(RetryConfig(max_attempts=3))
            async def call_api():
                return await client.request()
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.execute(func, *args, **kwargs)
        return wrapper
