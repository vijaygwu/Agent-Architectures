"""
Chapter 4: Agent Communication Protocols
MCP Client Implementation
=========================================

A client for connecting to MCP servers.

Usage:
    python mcp_client.py
"""

__all__ = [
    "MCPError",
    "MCPTool",
    "MCPClient",
    "calculate_backoff_with_jitter",
]

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any

# Structured logging - use StructuredLogger if available, fallback to standard logging
try:
    from common.metrics import StructuredLogger
    logger = StructuredLogger("mcp_client")
except ImportError:
    import logging

    class _KwargsLoggerShim:
        """Adapts stdlib logging to the StructuredLogger kwargs API."""

        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def _format(self, message: str, kwargs: dict) -> str:
            if kwargs:
                context = " ".join(
                    f"{key}={value}" for key, value in kwargs.items()
                )
                return f"{message} | {context}"
            return message

        def debug(self, message: str, **kwargs):
            self._logger.debug(self._format(message, kwargs))

        def info(self, message: str, **kwargs):
            self._logger.info(self._format(message, kwargs))

        def warning(self, message: str, **kwargs):
            self._logger.warning(self._format(message, kwargs))

        def error(self, message: str, **kwargs):
            self._logger.error(self._format(message, kwargs))

    logger = _KwargsLoggerShim("mcp_client")

# Circuit breaker for connection resilience
try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
except ImportError:
    # Inline fallback if module not available
    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        def __init__(self, breaker_name: str, time_until_retry: float):
            self.breaker_name = breaker_name
            self.time_until_retry = time_until_retry
            super().__init__(
                f"Circuit breaker '{breaker_name}' is OPEN. "
                f"Retry in {time_until_retry:.1f}s"
            )

    class CircuitBreaker:
        """Minimal inline circuit breaker implementation."""
        def __init__(self, name: str, failure_threshold: int = 5,
                     recovery_timeout: float = 30.0):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._is_open = False

        def allow(self) -> bool:
            """Return True if the circuit allows a request.

            Matches the public API of common.resilience.CircuitBreaker.
            """
            if self._is_open:
                elapsed = time.time() - self._last_failure_time
                if elapsed < self.recovery_timeout:
                    return False
                # Recovery timeout passed, allow test request
                self._is_open = False
                self._failure_count = 0
            return True

        def record_success(self) -> None:
            """Record a successful call."""
            self._failure_count = 0
            self._is_open = False

        def record_failure(self) -> None:
            """Record a failed call."""
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._is_open = True


# Module-level circuit breaker for connection attempts
_connection_circuit_breaker = CircuitBreaker(
    name="mcp_connection",
    failure_threshold=3,
    recovery_timeout=30.0
)


def calculate_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_factor: float = 0.5
) -> float:
    """
    Exponential backoff with jitter to prevent thundering herd.

    Args:
        attempt: Zero-based attempt number
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter_factor: Random variation (0.5 = ±50% of delay)

    Returns:
        Delay in seconds with random jitter applied

    Example: attempt=1, base=1.0, jitter=0.5
        exponential = 1.0 * 2^1 = 2.0
        jitter range = 2.0 * 0.5 = 1.0
        final delay = random between 1.0 and 3.0
    """
    exponential_delay = base_delay * (2 ** attempt)
    capped_delay = min(exponential_delay, max_delay)

    jitter_range = capped_delay * jitter_factor
    jittered_delay = capped_delay + random.uniform(-jitter_range, jitter_range)

    return max(0.1, jittered_delay)  # Minimum 100ms


class MCPError(Exception):
    """Error from MCP server."""
    pass


# Maximum number of tools to cache to prevent unbounded memory growth
MAX_CACHED_TOOLS = 1000


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict

class MCPClient:
    def __init__(self, max_tools: int = 1000, timeout: float = 30.0):
        self.request_id = 0
        self.reader = None
        self.writer = None
        self.process = None
        self.tools: dict[str, MCPTool] = {}
        self._max_tools = max_tools
        self.timeout = timeout

    async def connect_stdio(self, command: list[str],
                            max_retries: int = 3, base_delay: float = 1.0,
                            jitter_factor: float = 0.5):
        """Connect to an MCP server via stdio with retry logic and circuit breaker.

        Args:
            command: Command and arguments to launch the MCP server.
            max_retries: Maximum number of connection attempts (default: 3).
            base_delay: Base delay for exponential backoff in seconds (default: 1.0).
            jitter_factor: Random variation for backoff (0.5 = ±50% of delay).

        Raises:
            ConnectionError: If all connection attempts fail.
            CircuitBreakerOpen: If circuit breaker is open due to repeated failures.
        """
        # Check circuit breaker before attempting connection
        if not _connection_circuit_breaker.allow():
            raise CircuitBreakerOpen(
                _connection_circuit_breaker.name,
                _connection_circuit_breaker.recovery_timeout,
            )

        logger.info("Attempting MCP server connection",
                    command=command[0] if command else "unknown",
                    max_retries=max_retries)

        last_error = None
        for attempt in range(max_retries):
            try:
                logger.debug("Connection attempt started",
                            attempt=attempt + 1,
                            max_retries=max_retries)

                self.process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                self.reader = self.process.stdout
                self.writer = self.process.stdin

                # Initialize
                await self._initialize()

                # Load tools
                await self._load_tools()

                # Record success in circuit breaker
                _connection_circuit_breaker.record_success()
                logger.info("MCP server connection established",
                           tools_loaded=len(self.tools))
                return  # Success
            except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning("Connection attempt failed",
                              attempt=attempt + 1,
                              max_retries=max_retries,
                              error=str(e))

                # Clean up failed process if it exists
                if self.process:
                    try:
                        self.process.terminate()
                        await self.process.wait()
                    except Exception:
                        pass
                    self.process = None
                    self.reader = None
                    self.writer = None

                if attempt < max_retries - 1:
                    delay = calculate_backoff_with_jitter(
                        attempt, base_delay, jitter_factor=jitter_factor
                    )
                    logger.debug("Retry scheduled",
                                attempt=attempt + 1,
                                delay_seconds=round(delay, 2))
                    await asyncio.sleep(delay)

        # Record failure in circuit breaker
        _connection_circuit_breaker.record_failure()
        logger.error("MCP server connection failed after all retries",
                    max_retries=max_retries,
                    error=str(last_error))

        raise ConnectionError(
            f"Failed to connect to MCP server after {max_retries} attempts: {last_error}"
        )

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }

        self.writer.write((json.dumps(request) + "\n").encode())
        await self.writer.drain()

        # Read until we see the response whose id matches this
        # request. Servers may interleave notifications (no id) or
        # unrelated messages; skip them instead of desynchronizing.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"No response for request {request['id']}"
                )
            line = await asyncio.wait_for(
                self.reader.readline(), timeout=remaining
            )
            response = json.loads(line.decode())

            if response.get("id") != request["id"]:
                continue  # Notification or unrelated message

            if "error" in response:
                raise MCPError(response["error"]["message"])

            return response.get("result", {})

    async def _initialize(self):
        """Initialize the MCP connection."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "book-client", "version": "1.0.0"}
        })
        return result

    async def _load_tools(self):
        """Load available tools from server with size limit enforcement."""
        result = await self._send_request("tools/list")
        for tool_data in result.get("tools", []):
            # Enforce max size limit with LRU-style eviction
            if len(self.tools) >= self._max_tools:
                # Remove oldest entry (first inserted)
                oldest_key = next(iter(self.tools))
                del self.tools[oldest_key]
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data["description"],
                input_schema=tool_data["inputSchema"]
            )
            self.tools[tool.name] = tool

    async def call_tool(self, name: str, arguments: dict,
                         max_retries: int = 3, base_delay: float = 1.0,
                         jitter_factor: float = 0.5) -> Any:
        """Call a tool on the MCP server with retry logic.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            max_retries: Maximum number of retry attempts (default: 3).
            base_delay: Base delay for exponential backoff in seconds (default: 1.0).
            jitter_factor: Random variation for backoff (0.5 = ±50% of delay).

        Returns:
            The tool execution result.

        Raises:
            ValueError: If the tool name is not found.
            MCPError: If all retry attempts fail.
        """
        if name not in self.tools:
            logger.warning("Tool call failed - unknown tool", tool_name=name)
            raise ValueError(f"Unknown tool: {name}")

        logger.debug("Tool call started", tool_name=name)
        start_time = time.perf_counter()

        last_error = None
        for attempt in range(max_retries):
            try:
                result = await self._send_request("tools/call", {
                    "name": name,
                    "arguments": arguments
                })

                # Parse the result content
                content = result.get("content", [])
                if content and content[0]["type"] == "text":
                    parsed_result = json.loads(content[0]["text"])
                else:
                    parsed_result = result

                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Tool call completed",
                           tool_name=name,
                           latency_ms=round(latency_ms, 2),
                           attempts=attempt + 1)
                return parsed_result
            except asyncio.TimeoutError as e:
                # A timed-out call may still have executed on the
                # server; retrying would silently re-execute tools
                # that may not be idempotent, so fail fast instead.
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("Tool call timed out - not retrying",
                            tool_name=name,
                            attempt=attempt + 1,
                            latency_ms=round(latency_ms, 2))
                raise MCPError(
                    f"Tool call '{name}' timed out; not retried "
                    f"because the tool may not be idempotent"
                ) from e
            except (MCPError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("Tool call attempt failed",
                              tool_name=name,
                              attempt=attempt + 1,
                              max_retries=max_retries,
                              error=str(e))
                if attempt < max_retries - 1:
                    delay = calculate_backoff_with_jitter(
                        attempt, base_delay, jitter_factor=jitter_factor
                    )
                    logger.debug("Tool call retry scheduled",
                                tool_name=name,
                                attempt=attempt + 1,
                                delay_seconds=round(delay, 2))
                    await asyncio.sleep(delay)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Tool call failed after all retries",
                    tool_name=name,
                    max_retries=max_retries,
                    latency_ms=round(latency_ms, 2),
                    error=str(last_error))
        raise MCPError(f"Tool call failed after {max_retries} attempts: {last_error}")

    def get_tools_for_llm(self) -> list[dict]:
        """Get tools in format suitable for LLM function calling."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
            for tool in self.tools.values()
        ]

    async def close(self, timeout: float = 5.0):
        """Close the MCP connection and terminate the subprocess gracefully.

        Args:
            timeout: Maximum time to wait for graceful termination before force-killing.
        """
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass  # Already closed or errored
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # Process did not terminate gracefully, force kill
                self.process.kill()
                await self.process.wait()
            except Exception:
                pass  # Already terminated


async def demo():
    """Demonstrate MCP client usage."""
    client = MCPClient()

    # Connect to the server (assuming it's running)
    # await client.connect_stdio(["python", "mcp_server.py"])

    # For demo, we'll simulate with direct server usage
    logger.info("MCP Client ready")
    logger.info("To connect: await client.connect_stdio(['python', 'mcp_server.py'])")

if __name__ == "__main__":
    asyncio.run(demo())
