"""
Chapter 4: Agent Communication Protocols
MCP Server Implementation
=========================================

A complete MCP server providing web search and URL fetching tools.

Usage:
    python mcp_server.py

Production Notes:
- Add authentication middleware for tool access control
- Implement request rate limiting per client
- Use structured logging with correlation IDs for debugging
- Consider adding tool result caching for expensive operations
"""

import asyncio
import ipaddress
import json
import logging
import os
import signal
import socket
import sys
import time
import urllib.parse
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any
import aiohttp


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredLogger:
    """JSON-formatted structured logger for production observability.

    Outputs logs in JSON format for easy parsing by log aggregation systems
    (ELK, Datadog, CloudWatch, etc.).
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # Add JSON handler to stderr
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(self._JsonFormatter())
        self.logger.addHandler(handler)

    class _JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Add extra fields if present
            if hasattr(record, "extra"):
                log_data.update(record.extra)
            return json.dumps(log_data)

    def _log(self, level: int, message: str, **extra):
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        record.extra = extra
        self.logger.handle(record)

    def info(self, message: str, **extra):
        self._log(logging.INFO, message, **extra)

    def error(self, message: str, **extra):
        self._log(logging.ERROR, message, **extra)

    def warning(self, message: str, **extra):
        self._log(logging.WARNING, message, **extra)

    def debug(self, message: str, **extra):
        self._log(logging.DEBUG, message, **extra)


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class LatencyHistogram:
    """Simple histogram for tracking latency distributions."""
    buckets: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    counts: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    total_count: int = 0
    total_sum: float = 0.0

    def observe(self, value: float):
        """Record a latency observation."""
        self.total_count += 1
        self.total_sum += value
        for bucket in self.buckets:
            if value <= bucket:
                self.counts[bucket] += 1
                break
        else:
            self.counts[float("inf")] += 1

    def to_dict(self) -> dict:
        return {
            "count": self.total_count,
            "sum": self.total_sum,
            "avg": self.total_sum / self.total_count if self.total_count > 0 else 0,
            "buckets": {str(k): v for k, v in self.counts.items()}
        }


class ServerMetrics:
    """Metrics collection for MCP server observability.

    Tracks:
    - request_count: Total number of requests received
    - error_count: Total number of errors
    - latency_histogram: Distribution of request latencies
    - tool_calls: Count of each tool invocation
    """

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latency_histogram = LatencyHistogram()
        self.tool_calls: dict[str, int] = defaultdict(int)
        self._start_time = time.time()

    def record_request(self, method: str, latency: float, error: bool = False):
        """Record a request with its latency."""
        self.request_count += 1
        self.latency_histogram.observe(latency)
        if error:
            self.error_count += 1

    def record_tool_call(self, tool_name: str):
        """Record a tool invocation."""
        self.tool_calls[tool_name] += 1

    def to_dict(self) -> dict:
        """Export metrics as a dictionary."""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "latency": self.latency_histogram.to_dict(),
            "tool_calls": dict(self.tool_calls)
        }


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handles graceful shutdown for async servers."""

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
        logger.info("Shutdown signal received", signal=signum)
        self.shutdown_event.set()

    def cleanup_signals(self):
        """Remove signal handlers (call during cleanup)."""
        if self._loop:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._loop.remove_signal_handler(sig)
                except (ValueError, RuntimeError):
                    pass


# Initialize structured logger and metrics
logger = StructuredLogger("mcp-server-book")
metrics = ServerMetrics()


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


# =============================================================================
# Authentication
# =============================================================================

class TokenAuthenticator:
    """Simple token-based authentication.

    Configure via environment variable MCP_API_TOKEN or disable with MCP_AUTH_DISABLED=true.
    """

    def __init__(self):
        self.enabled = os.environ.get("MCP_AUTH_DISABLED", "").lower() != "true"
        self.valid_token = os.environ.get("MCP_API_TOKEN", "")
        if self.enabled and not self.valid_token:
            logging.getLogger("mcp-server-book").warning(
                "Authentication enabled but MCP_API_TOKEN is not "
                "set; all requests will be allowed"
            )

    def authenticate(self, token: str | None) -> bool:
        """Check if the provided token is valid.

        Args:
            token: The authentication token to verify.

        Returns:
            True if authentication passes (disabled or valid token), False otherwise.
        """
        if not self.enabled:
            return True
        if not self.valid_token:
            # No token configured, allow all (but log warning)
            return True
        return token == self.valid_token


# Global instances
_rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)
_authenticator = TokenAuthenticator()


@dataclass
class Tool:
    name: str
    description: str
    inputSchema: dict

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


async def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web using DuckDuckGo."""
    timeout = aiohttp.ClientTimeout(total=30.0)
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

async def fetch_url(url: str, max_length: int = 10000) -> dict:
    """Fetch content from a URL (http/https only, streamed reads)."""
    # SSRF guard: allow only http(s) URLs on public addresses.
    parsed = urllib.parse.urlparse(url)
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

    async with aiohttp.ClientSession() as session:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    # Stream at most max_length bytes instead of
                    # buffering an arbitrarily large body in memory.
                    raw = await response.content.read(max_length + 1)
                    try:
                        content = raw.decode(
                            response.charset or "utf-8",
                            errors="replace"
                        )
                    except LookupError:
                        content = raw.decode("utf-8", errors="replace")
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


class MCPServer:
    """MCP Server with structured logging and metrics collection."""

    def __init__(self):
        self.request_id = None  # Current request correlation ID
        logger.info("MCP server initializing")
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
        logger.info("MCP server initialized", tools=list(self.tools.keys()))

    async def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Route JSON-RPC requests to appropriate handlers with metrics."""
        method = request.method
        params = request.params or {}
        self.request_id = str(uuid.uuid4())[:8]  # Correlation ID
        start_time = time.time()
        error_occurred = False

        logger.info(
            "Request received",
            request_id=self.request_id,
            method=method,
            has_params=params is not None
        )

        try:
            if method == "initialize":
                return self._handle_initialize(request.id, params)
            elif method == "tools/list":
                return self._handle_tools_list(request.id)
            elif method == "tools/call":
                # Pass the auth token through so TokenAuthenticator
                # actually sees it (carried in params._meta, with a
                # top-level auth_token field accepted as well).
                meta = params.get("_meta") or {}
                auth_token = meta.get("auth_token") or params.get(
                    "auth_token"
                )
                return await self._handle_tools_call(
                    request.id, params, auth_token=auth_token
                )
            elif method == "resources/list":
                return self._handle_resources_list(request.id)
            elif method == "metrics":
                # Expose metrics endpoint
                return self._handle_metrics(request.id)
            else:
                error_occurred = True
                logger.warning(
                    "Unknown method requested",
                    request_id=self.request_id,
                    method=method
                )
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown method: {method}"}
                )
        except (ValueError, TypeError, KeyError, asyncio.TimeoutError) as e:
            error_occurred = True
            logger.error(
                "Request failed",
                request_id=self.request_id,
                method=method,
                error=str(e),
                error_type=type(e).__name__
            )
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )
        finally:
            latency = time.time() - start_time
            metrics.record_request(method, latency, error=error_occurred)
            logger.info(
                "Request completed",
                request_id=self.request_id,
                method=method,
                latency_ms=round(latency * 1000, 2),
                error=error_occurred
            )

    def _handle_metrics(self, req_id: int) -> JsonRpcResponse:
        """Return server metrics."""
        return JsonRpcResponse(
            id=req_id,
            result=metrics.to_dict()
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

    async def _handle_tools_call(self, req_id: int, params: dict,
                                   auth_token: str | None = None) -> JsonRpcResponse:
        """Execute a tool call with rate limiting, authentication, and logging.

        Args:
            req_id: The request ID.
            params: Parameters containing tool name and arguments.
            auth_token: Optional authentication token from request metadata.

        Returns:
            JSON-RPC response with tool result or error.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(
            "Tool call started",
            request_id=self.request_id,
            tool=tool_name,
            arguments=arguments
        )

        # Check authentication
        if not _authenticator.authenticate(auth_token):
            logger.warning(
                "Authentication failed",
                request_id=self.request_id,
                tool=tool_name
            )
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32001, "message": "Authentication failed"}
            )

        # Check rate limit
        if not _rate_limiter.allow():
            logger.warning(
                "Rate limit exceeded",
                request_id=self.request_id,
                tool=tool_name
            )
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32002, "message": "Rate limit exceeded. Try again later."}
            )

        if tool_name not in self.tool_handlers:
            logger.warning(
                "Unknown tool requested",
                request_id=self.request_id,
                tool=tool_name
            )
            return JsonRpcResponse(
                id=req_id,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"}
            )

        # Validate arguments against the tool's declared inputSchema
        # before dispatch: unknown keys are rejected instead of being
        # forwarded as arbitrary kwargs into the handler.
        schema = self.tools[tool_name].inputSchema
        allowed_args = set(schema.get("properties", {}))
        unknown_args = set(arguments) - allowed_args
        missing_args = set(schema.get("required", [])) - set(arguments)
        if unknown_args or missing_args:
            return JsonRpcResponse(
                id=req_id,
                error={
                    "code": -32602,
                    "message": (
                        f"Invalid arguments for {tool_name}: "
                        f"unknown={sorted(unknown_args)}, "
                        f"missing={sorted(missing_args)}"
                    )
                }
            )

        metrics.record_tool_call(tool_name)
        tool_start = time.time()

        handler = self.tool_handlers[tool_name]
        try:
            result = await handler(**arguments)
        except Exception as e:
            logger.error(
                "Tool execution failed",
                request_id=self.request_id,
                tool=tool_name,
                error=str(e),
                error_type=type(e).__name__
            )
            return JsonRpcResponse(
                id=req_id,
                error={
                    "code": -32603,
                    "message": (
                        f"Tool execution failed: {type(e).__name__}"
                    )
                }
            )

        tool_latency = time.time() - tool_start
        logger.info(
            "Tool call completed",
            request_id=self.request_id,
            tool=tool_name,
            latency_ms=round(tool_latency * 1000, 2)
        )

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
    """Run MCP server over stdio with graceful shutdown and logging.

    Args:
        shutdown_event: Optional event to signal graceful shutdown
    """
    logger.info("Starting MCP server", transport="stdio")
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
            continue

        if not line:
            logger.info("Input stream closed, shutting down")
            break

        try:
            data = json.loads(line.decode())
            request = JsonRpcRequest(**data)
            response = await server.handle_request(request)

            response_json = json.dumps(asdict(response)) + "\n"
            writer.write(response_json.encode())
            await writer.drain()
        except json.JSONDecodeError as e:
            logger.error("JSON parse error", error=str(e))
            metrics.record_request("parse_error", 0, error=True)
            error_response = JsonRpcResponse(
                error={"code": -32700, "message": "Parse error"}
            )
            writer.write((json.dumps(asdict(error_response)) + "\n").encode())
            await writer.drain()

    # Log final metrics on shutdown
    logger.info("Server shutting down", final_metrics=metrics.to_dict())


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
