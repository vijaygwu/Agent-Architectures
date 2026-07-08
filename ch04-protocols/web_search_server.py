"""
Chapter 4: Agent Communication Protocols
MCP Server Implementation
=========================================

A complete MCP server providing web search and URL fetching tools.

Usage:
    python web_search_server.py

Production Notes:
- Graceful shutdown is supported via shutdown_event parameter
- Resource bounds prevent unbounded memory growth
"""

import asyncio
import ipaddress
import json
import os
import signal
import socket
import time
from dataclasses import dataclass, asdict
from typing import Any

import aiohttp

# Timeout for external API calls (configurable via environment variable)
EXTERNAL_API_TIMEOUT = float(os.environ.get("SEARCH_API_TIMEOUT", "30.0"))


# =============================================================================
# Structured Logging
# =============================================================================

try:
    from common.metrics import StructuredLogger
    logger = StructuredLogger("web_search_server")
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("web_search_server")


# =============================================================================
# Circuit Breaker
# =============================================================================

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
except ImportError:
    # Inline fallback implementation
    class CircuitBreakerOpen(Exception):
        """Raised when circuit breaker is open."""
        pass

    class CircuitBreaker:
        """Simple circuit breaker for external service calls."""

        def __init__(self, name: str, failure_threshold: int = 5,
                     recovery_timeout: float = 60.0):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failures = 0
            self._last_failure_time: float | None = None
            self._state = "closed"  # closed, open, half-open

        def allow(self) -> bool:
            """Check if a request is allowed through the circuit breaker."""
            if self._state == "closed":
                return True
            elif self._state == "open":
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = "half-open"
                        logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                        return True
                return False
            else:  # half-open
                return True

        def record_success(self):
            """Record a successful call."""
            if self._state == "half-open":
                self._state = "closed"
                self._failures = 0
                logger.info(f"Circuit breaker '{self.name}' closed after successful call")
            elif self._state == "closed":
                self._failures = 0

        def record_failure(self):
            """Record a failed call."""
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self._failures} failures"
                )

        def reset(self):
            """Reset the circuit breaker to closed state."""
            self._failures = 0
            self._last_failure_time = None
            self._state = "closed"


# Circuit breaker for web search provider
_search_circuit_breaker = CircuitBreaker(
    name="web_search_provider",
    failure_threshold=5,
    recovery_timeout=60.0
)

# Circuit breaker for URL fetch operations
_fetch_circuit_breaker = CircuitBreaker(
    name="url_fetch_provider",
    failure_threshold=5,
    recovery_timeout=60.0
)


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter for controlling request throughput."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed per window.
            window_seconds: Time window in seconds (default: 60.0 for 100 req/min).
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []

    def allow(self) -> bool:
        """Check if a request is allowed under the rate limit.

        Returns:
            True if the request is allowed, False otherwise.
        """
        now = time.time()
        # Remove expired timestamps
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

    def reset(self):
        """Reset the rate limiter."""
        self.requests.clear()


# Global rate limiter for tool calls (100 requests per minute)
_tool_rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)


# =============================================================================
# Request Validation
# =============================================================================

def validate_jsonrpc_request(data: dict) -> tuple[bool, str]:
    """Validate JSON-RPC request structure.

    Args:
        data: The parsed JSON data to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check required fields
    if "jsonrpc" not in data:
        return False, "Missing required field: jsonrpc"
    if data.get("jsonrpc") != "2.0":
        return False, "Invalid jsonrpc version, expected '2.0'"
    if "id" not in data:
        return False, "Missing required field: id"
    if "method" not in data:
        return False, "Missing required field: method"

    # Validate field types
    if not isinstance(data["id"], (int, str)):
        return False, "Field 'id' must be a string or integer"
    if not isinstance(data["method"], str):
        return False, "Field 'method' must be a string"
    if "params" in data and not isinstance(data["params"], dict):
        return False, "Field 'params' must be an object"

    return True, ""


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handles graceful shutdown for async servers.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.setup_signals()

        async def server_loop():
            while not shutdown.shutdown_event.is_set():
                # process requests with timeout
                try:
                    await asyncio.wait_for(process(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
    """

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self._loop = None

    def setup_signals(self, loop: asyncio.AbstractEventLoop = None):
        """Register signal handlers for graceful shutdown."""
        self._loop = loop or asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, signum):
        """Handle shutdown signal by setting the event."""
        self.shutdown_event.set()

    def cleanup_signals(self):
        """Remove signal handlers (call during cleanup)."""
        if self._loop:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._loop.remove_signal_handler(sig)
                except (ValueError, RuntimeError):
                    pass


# =============================================================================
# JSON-RPC Types (simplified from MCP spec)
# =============================================================================

@dataclass
class Resource:
    uri: str
    name: str
    mimeType: str = "text/plain"

@dataclass
class JsonRpcRequest:
    jsonrpc: str
    id: int | str
    method: str
    params: dict | None = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: int | str = None
    result: Any = None
    error: dict = None


@dataclass
class Tool:
    name: str
    description: str
    inputSchema: dict


# =============================================================================
# Tool Implementations
# =============================================================================

async def web_search(query: str, num_results: int = 5, timeout_seconds: float = 30.0) -> dict:
    """Search the web using DuckDuckGo."""
    # Sanitize query for logging (truncate and remove sensitive patterns)
    sanitized_query = query[:100] if len(query) > 100 else query
    logger.info(f"Search request received", extra={
        "query_length": len(query),
        "query_preview": sanitized_query[:50],
        "num_results": num_results
    })

    # Check circuit breaker before making external call
    if not _search_circuit_breaker.allow():
        logger.warning("Search request rejected - circuit breaker open", extra={
            "circuit_breaker": "web_search_provider"
        })
        return {"success": False, "error": "Search provider circuit breaker open", "query": query}

    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    start_time = time.time()

    async def _do_search():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    # Extract related topics (with defensive checks)
                    for topic in data.get("RelatedTopics", [])[:num_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": topic.get("Text", "")[:100],
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", "")
                            })

                    return {"success": True, "results": results, "query": query}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

    try:
        result = await asyncio.wait_for(_do_search(), timeout=EXTERNAL_API_TIMEOUT)
        latency_ms = (time.time() - start_time) * 1000
        _search_circuit_breaker.record_success()
        logger.info("Search API call completed", extra={
            "latency_ms": round(latency_ms, 2),
            "success": result.get("success", False),
            "result_count": len(result.get("results", []))
        })
        return result
    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        _search_circuit_breaker.record_failure()
        logger.error("Search API timeout", extra={
            "latency_ms": round(latency_ms, 2),
            "timeout_threshold": EXTERNAL_API_TIMEOUT,
            "query_preview": sanitized_query[:50]
        })
        return {"success": False, "error": "External API timeout", "query": query}
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        _search_circuit_breaker.record_failure()
        logger.error("Search API error", extra={
            "latency_ms": round(latency_ms, 2),
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        raise


async def fetch_url(url: str, max_length: int = 10000) -> dict:
    """Fetch content from a URL."""
    # Sanitize URL for logging (remove query params that might contain sensitive data)
    from urllib.parse import urlparse
    parsed = urlparse(url)
    sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    logger.info("URL fetch request received", extra={
        "url_host": parsed.netloc,
        "url_path": parsed.path[:50],
        "max_length": max_length
    })

    # SSRF guard: allow only http(s) URLs on public addresses.
    if parsed.scheme not in ("http", "https"):
        return {"success": False,
                "error": f"Unsupported URL scheme: {parsed.scheme}",
                "url": url}
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname, None
        )
    except (socket.gaierror, TypeError) as e:
        return {"success": False,
                "error": f"Could not resolve host: {e}", "url": url}
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (addr.is_private or addr.is_loopback
                or addr.is_link_local or addr.is_reserved):
            return {"success": False,
                    "error": "URL resolves to a private address",
                    "url": url}

    # Check circuit breaker before making external call
    if not _fetch_circuit_breaker.allow():
        logger.warning("Fetch request rejected - circuit breaker open", extra={
            "circuit_breaker": "url_fetch_provider",
            "url_host": parsed.netloc
        })
        return {"success": False, "error": "URL fetch circuit breaker open", "url": url}

    start_time = time.time()

    async def _do_fetch():
        async with aiohttp.ClientSession() as session:
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                # Do not follow redirects: a redirect to a private or
                # link-local address would bypass the SSRF guard above
                async with session.get(
                    url, timeout=timeout, allow_redirects=False
                ) as response:
                    if 300 <= response.status < 400:
                        return {"success": False,
                                "error": "Redirects are not followed",
                                "url": url}
                    if response.status == 200:
                        # Stream at most max_length bytes rather than
                        # buffering the whole body in memory.
                        raw = await response.content.read(max_length + 1)
                        try:
                            content = raw.decode(
                                response.charset or "utf-8",
                                errors="replace"
                            )
                        except LookupError:
                            content = raw.decode(
                                "utf-8", errors="replace"
                            )
                        return {
                            "success": True,
                            "url": url,
                            "content": content[:max_length],
                            "truncated": len(raw) > max_length,
                            "content_type": response.headers.get(
                                "Content-Type", "unknown")
                        }
                    else:
                        err = f"HTTP {response.status}"
                        return {"success": False, "error": err, "url": url}
            except aiohttp.ClientError as e:
                return {"success": False, "error": f"Client error: {e}", "url": url}
            except TimeoutError:
                return {"success": False, "error": "Request timed out", "url": url}

    try:
        result = await asyncio.wait_for(_do_fetch(), timeout=EXTERNAL_API_TIMEOUT)
        latency_ms = (time.time() - start_time) * 1000
        if result.get("success"):
            _fetch_circuit_breaker.record_success()
        else:
            _fetch_circuit_breaker.record_failure()
        logger.info("URL fetch completed", extra={
            "latency_ms": round(latency_ms, 2),
            "success": result.get("success", False),
            "url_host": parsed.netloc,
            "truncated": result.get("truncated", False)
        })
        return result
    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        _fetch_circuit_breaker.record_failure()
        logger.error("URL fetch timeout", extra={
            "latency_ms": round(latency_ms, 2),
            "timeout_threshold": EXTERNAL_API_TIMEOUT,
            "url_host": parsed.netloc
        })
        return {"success": False, "error": "External API timeout", "url": url}
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        _fetch_circuit_breaker.record_failure()
        logger.error("URL fetch error", extra={
            "latency_ms": round(latency_ms, 2),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "url_host": parsed.netloc
        })
        raise


# =============================================================================
# MCP Server
# =============================================================================

class MCPServer:
    """MCP Server with bounded resource usage."""

    # Maximum cached results to prevent unbounded memory growth
    MAX_CACHED_RESULTS = 1000

    def __init__(self):
        self._result_cache: dict[str, dict] = {}  # Bounded cache for results
        self.tools = {
            "web_search": Tool(
                name="web_search",
                description="Search the web for information using DuckDuckGo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            "fetch_url": Tool(
                name="fetch_url",
                description="Fetch the content of a URL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum content length",
                            "default": 10000
                        }
                    },
                    "required": ["url"]
                }
            )
        }

        self.tool_handlers = {
            "web_search": web_search,
            "fetch_url": fetch_url
        }

    async def handle_request(
            self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Route JSON-RPC requests to appropriate handlers."""
        method = request.method
        params = request.params or {}

        try:
            if method == "initialize":
                return self._handle_initialize(request.id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request.id)
            elif method == "tools/call":
                return await self._handle_tools_call(request.id, params)
            elif method == "resources/list":
                return self._handle_resources_list(request.id)
            elif method in ("health", "ping"):
                # Health check endpoint for monitoring
                return self._handle_health(request.id)
            else:
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown method: {method}"}
                )
        except (ValueError, TypeError, KeyError) as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )

    def _handle_health(self, req_id: int) -> JsonRpcResponse:
        """Handle health check requests."""
        return JsonRpcResponse(
            id=req_id,
            result={
                "status": "healthy",
                "server": "book-mcp-server",
                "cache_size": len(self._result_cache)
            }
        )

    def _handle_initialize(self, req_id: int, params: dict) -> JsonRpcResponse:
        """Handle MCP initialization."""
        return JsonRpcResponse(
            id=req_id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "book-mcp-server",
                    "version": "1.0.0"
                }
            }
        )

    def _handle_tools_list(self, req_id: int) -> JsonRpcResponse:
        """List available tools."""
        return JsonRpcResponse(
            id=req_id,
            result={
                "tools": [asdict(tool) for tool in self.tools.values()]
            }
        )

    async def _handle_tools_call(self, req_id: int, params: dict) -> JsonRpcResponse:
        """Execute a tool call with rate limiting.

        Rate limited to 100 requests per minute.
        """
        # Check rate limit before processing
        if not _tool_rate_limiter.allow():
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32002, "message": "Rate limit exceeded. Try again later."}
            )

        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tool_handlers:
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"}
            )

        handler = self.tool_handlers[tool_name]
        result = await handler(**arguments)

        return JsonRpcResponse(
            id=req_id,
            result={
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)}
                ]
            }
        )

    def _handle_resources_list(self, req_id: int) -> JsonRpcResponse:
        """List available resources."""
        return JsonRpcResponse(
            id=req_id,
            result={"resources": []}
        )


async def run_stdio_server(shutdown_event: asyncio.Event = None):
    """Run MCP server over stdio with graceful shutdown support.

    Args:
        shutdown_event: Optional event to signal graceful shutdown.
                       When set, the server will complete current request
                       and exit the loop cleanly.
    """
    import sys
    server = MCPServer()
    loop = asyncio.get_running_loop()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    while shutdown_event is None or not shutdown_event.is_set():
        try:
            # Use timeout to periodically check shutdown event
            line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        except asyncio.TimeoutError:
            # No input, check shutdown and continue
            continue

        if not line:
            break

        try:
            data = json.loads(line.decode())

            # Validate JSON-RPC request structure
            is_valid, error_msg = validate_jsonrpc_request(data)
            if not is_valid:
                error_response = JsonRpcResponse(
                    id=data.get("id"),
                    error={"code": -32600, "message": f"Invalid request: {error_msg}"}
                )
                writer.write((json.dumps(asdict(error_response)) + "\n").encode())
                await writer.drain()
                continue

            request = JsonRpcRequest(**data)
            response = await server.handle_request(request)

            response_json = json.dumps(asdict(response)) + "\n"
            writer.write(response_json.encode())
            await writer.drain()
        except json.JSONDecodeError:
            error_response = JsonRpcResponse(
                error={"code": -32700, "message": "Parse error"}
            )
            writer.write((json.dumps(asdict(error_response)) + "\n").encode())
            await writer.drain()
        except Exception as e:
            # Keep serving on unexpected failures (e.g. upstream
            # aiohttp errors re-raised by tool handlers): return a
            # JSON-RPC internal error instead of crashing the loop.
            logger.error("Unhandled server error", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            error_response = JsonRpcResponse(
                error={"code": -32603, "message": "Internal error"}
            )
            writer.write((json.dumps(asdict(error_response)) + "\n").encode())
            await writer.drain()


async def run_with_graceful_shutdown():
    """Run the stdio server with signal-based graceful shutdown."""
    shutdown = GracefulShutdown()
    loop = asyncio.get_running_loop()
    shutdown.setup_signals(loop)

    try:
        await run_stdio_server(shutdown.shutdown_event)
    finally:
        shutdown.cleanup_signals()


async def demo():
    """Demonstrate MCP server functionality."""
    server = MCPServer()

    # Initialize
    init_request = JsonRpcRequest(
        jsonrpc="2.0",
        id=1,
        method="initialize",
        params={"protocolVersion": "2024-11-05"}
    )
    init_response = await server.handle_request(init_request)
    print(f"Initialize: {init_response.result}")

    # List tools
    list_request = JsonRpcRequest(jsonrpc="2.0", id=2, method="tools/list")
    list_response = await server.handle_request(list_request)
    print(f"\nAvailable tools: {[t['name'] for t in list_response.result['tools']]}")

    # Call web_search
    search_request = JsonRpcRequest(
        jsonrpc="2.0",
        id=3,
        method="tools/call",
        params={
            "name": "web_search",
            "arguments": {"query": "Python programming", "num_results": 3}
        }
    )
    search_response = await server.handle_request(search_request)
    print(f"\nSearch results:\n{search_response.result['content'][0]['text']}")

if __name__ == "__main__":
    # Run demo if executed directly
    asyncio.run(demo())
