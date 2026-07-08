"""
Chapter 4: MCP Server Implementation
=====================================
A complete Model Context Protocol server for enterprise tool integration.

This example implements an MCP server that exposes procurement tools
to AI applications following the official MCP specification.

Reference: https://modelcontextprotocol.io/specification/2024-11-05
"""

__all__ = [
    "RateLimiter",
    "RateLimitExceeded",
    "DistributedRateLimiter",
    "GracefulShutdown",
    "Vendor",
    "PurchaseOrder",
    "ProcurementDatabase",
]

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Sequence
from dataclasses import dataclass, asdict

# Optional Redis support for distributed rate limiting
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

RATE_LIMIT_REDIS_URL = os.environ.get("RATE_LIMIT_REDIS_URL")


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


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class DistributedRateLimiter:
    """Rate limiter with Redis backend for multi-instance deployments.

    Uses sliding window log algorithm for accurate rate limiting.
    """

    def __init__(self, redis_url: str, key_prefix: str,
                 max_requests: int, window_seconds: float):
        """Initialize distributed rate limiter.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for Redis keys.
            max_requests: Maximum number of requests allowed per window.
            window_seconds: Time window in seconds.
        """
        self.redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def allow(self, client_id: str = "global") -> bool:
        """Check if a request is allowed under the rate limit.

        Args:
            client_id: Identifier for the client (default: "global" for shared limit).

        Returns:
            True if the request is allowed, False otherwise.
        """
        key = f"{self.key_prefix}:{client_id}"
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current entries in window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry to clean up keys
        pipe.expire(key, int(self.window_seconds) + 1)

        results = pipe.execute()
        current_count = results[1]
        return current_count < self.max_requests

    def reset(self, client_id: str = "global"):
        """Reset the rate limiter for a specific client.

        Args:
            client_id: Identifier for the client to reset.
        """
        key = f"{self.key_prefix}:{client_id}"
        self.redis.delete(key)


# Initialize rate limiter (distributed if Redis available and configured)
if REDIS_AVAILABLE and RATE_LIMIT_REDIS_URL:
    _tool_rate_limiter = DistributedRateLimiter(
        redis_url=RATE_LIMIT_REDIS_URL,
        key_prefix="mcp_server",
        max_requests=100,
        window_seconds=60.0
    )
else:
    _tool_rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)


# =============================================================================
# Circuit Breaker for Tool Execution (External Service Calls)
# =============================================================================

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
    _RESILIENCE_AVAILABLE = True
except ImportError:
    _RESILIENCE_AVAILABLE = False

    # Inline fallback circuit breaker implementation
    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        def __init__(self, message: str):
            super().__init__(message)

    class CircuitBreaker:
        """Simple circuit breaker fallback for when common.resilience is unavailable."""

        def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._is_open = False

        def allow(self) -> bool:
            """Check if requests are allowed through the circuit breaker."""
            if not self._is_open:
                return True
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._is_open = False
                self._failure_count = 0
                return True
            return False

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


# Circuit breaker for tool execution (external service calls)
_tool_execution_circuit_breaker = CircuitBreaker(
    name="mcp_tool_execution",
    failure_threshold=5,
    recovery_timeout=60.0
)


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """Handles graceful shutdown for async servers.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.setup_signals()

        while not shutdown.shutdown_event.is_set():
            # process requests
    """

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self._loop = None

    def setup_signals(self, loop: asyncio.AbstractEventLoop = None):
        """Register signal handlers for graceful shutdown.

        Args:
            loop: Event loop to use for thread-safe signal handling
        """
        self._loop = loop or asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, signum):
        """Handle shutdown signal by setting the event."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_event.set()

    def cleanup_signals(self):
        """Remove signal handlers (call during cleanup)."""
        if self._loop:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._loop.remove_signal_handler(sig)
                except (ValueError, RuntimeError):
                    pass

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        Prompt,
        PromptMessage,
        PromptArgument,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    stdio_server = None
    Tool = TextContent = Resource = Prompt = PromptMessage = PromptArgument = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procurement-mcp-server")


# =============================================================================
# Domain Models
# =============================================================================

@dataclass
class Vendor:
    id: str
    name: str
    category: str
    risk_score: float
    contact_email: str
    payment_terms: int  # days
    active: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PurchaseOrder:
    id: str
    vendor_id: str
    items: list[dict]
    total_amount: float
    currency: str
    status: str  # draft, pending_approval, approved, rejected, completed
    created_by: str
    created_at: str
    approver: str | None = None
    approved_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Mock Database (Replace with real database in production)
# =============================================================================

class ProcurementDatabase:
    """Simulated procurement database for demonstration"""

    def __init__(self):
        self.vendors: dict[str, Vendor] = {
            "V001": Vendor(
                id="V001",
                name="Acme Supplies Co",
                category="office_supplies",
                risk_score=0.15,
                contact_email="sales@acme.com",
                payment_terms=30,
                active=True
            ),
            "V002": Vendor(
                id="V002",
                name="TechParts International",
                category="electronics",
                risk_score=0.25,
                contact_email="orders@techparts.com",
                payment_terms=45,
                active=True
            ),
            "V003": Vendor(
                id="V003",
                name="Global Logistics Ltd",
                category="shipping",
                risk_score=0.10,
                contact_email="shipping@globallog.com",
                payment_terms=15,
                active=True
            ),
        }

        self.purchase_orders: dict[str, PurchaseOrder] = {}
        self._max_purchase_orders = 100000
        self.po_counter = 1000

    async def get_vendor(self, vendor_id: str) -> Vendor | None:
        return self.vendors.get(vendor_id)

    async def search_vendors(
        self,
        category: str | None = None,
        max_risk_score: float | None = None
    ) -> list[Vendor]:
        results = list(self.vendors.values())

        if category:
            results = [v for v in results if v.category == category]

        if max_risk_score is not None:
            results = [v for v in results if v.risk_score <= max_risk_score]

        return results

    async def create_purchase_order(
        self,
        vendor_id: str,
        items: list[dict],
        created_by: str,
        approver: str
    ) -> PurchaseOrder:
        self.po_counter += 1
        po_id = f"PO{self.po_counter}"

        total = sum(item.get("quantity", 1) * item.get("unit_price", 0) for item in items)

        po = PurchaseOrder(
            id=po_id,
            vendor_id=vendor_id,
            items=items,
            total_amount=total,
            currency="USD",
            status="pending_approval",
            created_by=created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            approver=approver
        )

        # Evict completed orders if at capacity
        if len(self.purchase_orders) >= self._max_purchase_orders:
            completed = [pid for pid, p in self.purchase_orders.items()
                        if p.status in ("completed", "rejected")]
            for pid in completed[:1000]:
                del self.purchase_orders[pid]
        self.purchase_orders[po_id] = po
        return po

    async def get_purchase_order(self, po_id: str) -> PurchaseOrder | None:
        return self.purchase_orders.get(po_id)

    async def update_po_status(self, po_id: str, status: str) -> PurchaseOrder | None:
        po = self.purchase_orders.get(po_id)
        if po:
            po.status = status
            if status == "approved":
                po.approved_at = datetime.now(timezone.utc).isoformat()
        return po


# =============================================================================
# MCP Server Implementation
# =============================================================================

# Initialize database
db = ProcurementDatabase()

# Create MCP server (only if MCP is available)
if MCP_AVAILABLE:
    server = Server("procurement-mcp-server")
else:
    server = None
    # Define no-op decorator for when MCP is unavailable
    def _noop_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Use conditional decorator
_list_tools = server.list_tools if MCP_AVAILABLE else _noop_decorator
_call_tool = server.call_tool if MCP_AVAILABLE else _noop_decorator
_list_resources = server.list_resources if MCP_AVAILABLE else _noop_decorator
_read_resource = server.read_resource if MCP_AVAILABLE else _noop_decorator
_list_prompts = server.list_prompts if MCP_AVAILABLE else _noop_decorator
_get_prompt = server.get_prompt if MCP_AVAILABLE else _noop_decorator


@_list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools.

    Tools are model-controlled - the LLM decides when to invoke them
    based on the user's request and tool descriptions.
    """
    return [
        Tool(
            name="lookup_vendor",
            description=(
                "Look up detailed information about a specific vendor by their ID. "
                "Returns vendor details including name, category, risk score, "
                "contact information, and payment terms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor_id": {
                        "type": "string",
                        "description": "The unique vendor identifier (e.g., V001)"
                    }
                },
                "required": ["vendor_id"]
            }
        ),
        Tool(
            name="search_vendors",
            description=(
                "Search for vendors matching specified criteria. "
                "Can filter by category and maximum acceptable risk score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Vendor category to filter by",
                        "enum": ["office_supplies", "electronics", "shipping", "raw_materials"]
                    },
                    "max_risk_score": {
                        "type": "number",
                        "description": "Maximum acceptable risk score (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        ),
        Tool(
            name="create_purchase_order",
            description=(
                "Create a new purchase order for a vendor. "
                "The PO will be created in 'pending_approval' status and "
                "routed to the specified approver. Returns the created PO details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor_id": {
                        "type": "string",
                        "description": "The vendor to create the PO for"
                    },
                    "items": {
                        "type": "array",
                        "description": "List of items to order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                                "unit_price": {"type": "number", "minimum": 0}
                            },
                            "required": ["description", "quantity", "unit_price"]
                        }
                    },
                    "approver": {
                        "type": "string",
                        "description": "Email of the person who should approve this PO"
                    }
                },
                "required": ["vendor_id", "items", "approver"]
            }
        ),
        Tool(
            name="get_purchase_order",
            description="Retrieve details of an existing purchase order by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "po_id": {
                        "type": "string",
                        "description": "The purchase order ID (e.g., PO1001)"
                    }
                },
                "required": ["po_id"]
            }
        ),
        Tool(
            name="approve_purchase_order",
            description=(
                "Approve a pending purchase order. "
                "Only POs in 'pending_approval' status can be approved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "po_id": {
                        "type": "string",
                        "description": "The purchase order ID to approve"
                    }
                },
                "required": ["po_id"]
            }
        )
    ]


@_call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """
    Handle tool invocations.

    Each tool call is logged for audit purposes and returns structured
    JSON responses that the LLM can interpret.

    Also handles special health check methods for monitoring.
    Rate limited to 100 requests per minute.
    """
    # Handle health check / ping (internal method, not a tool) - not rate limited
    if name in ("health", "ping"):
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "healthy",
                "server": "procurement-mcp-server",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        )]

    # Check rate limit before processing tool calls. The distributed
    # limiter performs a synchronous Redis round trip, so run it in
    # a worker thread to avoid blocking the event loop.
    if not await asyncio.to_thread(_tool_rate_limiter.allow):
        logger.warning(f"Rate limit exceeded for tool call: {name}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "isError": True
            })
        )]

    logger.info(f"Tool called: {name} with arguments: {arguments}")

    # Check circuit breaker before tool execution (protects against cascading failures)
    if not _tool_execution_circuit_breaker.allow():
        logger.warning(f"Circuit breaker open for tool execution: {name}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": "Tool execution circuit breaker open. Service temporarily unavailable.",
                "isError": True
            })
        )]

    result = None
    try:
        if name == "lookup_vendor":
            vendor = await db.get_vendor(arguments["vendor_id"])
            if vendor:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "vendor": vendor.to_dict()
                    }, indent=2)
                )]
            else:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {arguments['vendor_id']} not found"
                    })
                )]

        elif name == "search_vendors":
            vendors = await db.search_vendors(
                category=arguments.get("category"),
                max_risk_score=arguments.get("max_risk_score")
            )
            result = [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(vendors),
                    "vendors": [v.to_dict() for v in vendors]
                }, indent=2)
            )]

        elif name == "create_purchase_order":
            # Validate vendor exists
            vendor = await db.get_vendor(arguments["vendor_id"])
            if not vendor:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {arguments['vendor_id']} not found"
                    })
                )]
            elif not vendor.active:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {vendor.name} is not active"
                    })
                )]
            else:
                po = await db.create_purchase_order(
                    vendor_id=arguments["vendor_id"],
                    items=arguments["items"],
                    created_by="mcp-agent",  # In production, get from auth context
                    approver=arguments["approver"]
                )

                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Purchase order {po.id} created successfully",
                        "purchase_order": po.to_dict()
                    }, indent=2)
                )]

        elif name == "get_purchase_order":
            po = await db.get_purchase_order(arguments["po_id"])
            if po:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "purchase_order": po.to_dict()
                    }, indent=2)
                )]
            else:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Purchase order {arguments['po_id']} not found"
                    })
                )]

        elif name == "approve_purchase_order":
            po = await db.get_purchase_order(arguments["po_id"])
            if not po:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Purchase order {arguments['po_id']} not found"
                    })
                )]
            elif po.status != "pending_approval":
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Cannot approve PO in status '{po.status}'"
                    })
                )]
            else:
                updated_po = await db.update_po_status(arguments["po_id"], "approved")
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": f"Purchase order {po.id} approved",
                        "purchase_order": updated_po.to_dict()
                    }, indent=2)
                )]

        else:
            result = [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}"
                })
            )]

        # Record success for circuit breaker (service responded, even if business logic failed)
        _tool_execution_circuit_breaker.record_success()
        return result

    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.exception(f"Error in tool {name}")
        _tool_execution_circuit_breaker.record_failure()
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "isError": True
            })
        )]


@_list_resources()
async def list_resources() -> list[Resource]:
    """
    List available resources.

    Resources are application-controlled - the host application decides
    when to include them in context.
    """
    return [
        Resource(
            uri="procurement://vendors/catalog",
            name="Vendor Catalog",
            description="Complete catalog of approved vendors",
            mimeType="application/json"
        ),
        Resource(
            uri="procurement://policies/approval-matrix",
            name="Approval Matrix",
            description="PO approval thresholds and routing rules",
            mimeType="application/json"
        )
    ]


@_read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content by URI"""

    if uri == "procurement://vendors/catalog":
        vendors = await db.search_vendors()
        return json.dumps({
            "catalog_version": "2024.1",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "vendors": [v.to_dict() for v in vendors]
        }, indent=2)

    elif uri == "procurement://policies/approval-matrix":
        return json.dumps({
            "approval_matrix": [
                {"max_amount": 1000, "approver_level": "team_lead"},
                {"max_amount": 10000, "approver_level": "manager"},
                {"max_amount": 50000, "approver_level": "director"},
                {"max_amount": float("inf"), "approver_level": "vp"}
            ],
            "special_categories": {
                "electronics": {"additional_approval": "it_security"},
                "software": {"additional_approval": "it_architecture"}
            }
        }, indent=2)

    raise ValueError(f"Unknown resource: {uri}")


@_list_prompts()
async def list_prompts() -> list[Prompt]:
    """
    List available prompt templates.

    Prompts are user-controlled templates that help users interact
    with the procurement system effectively.
    """
    return [
        Prompt(
            name="find_vendor",
            description="Find a vendor for a specific procurement need",
            arguments=[
                PromptArgument(
                    name="need",
                    description="What you need to procure",
                    required=True
                ),
                PromptArgument(
                    name="budget",
                    description="Approximate budget",
                    required=False
                )
            ]
        ),
        Prompt(
            name="create_po_workflow",
            description="Step-by-step workflow for creating a purchase order",
            arguments=[
                PromptArgument(
                    name="vendor_id",
                    description="The vendor to order from",
                    required=True
                )
            ]
        )
    ]


@_get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> list[PromptMessage]:
    """Get a prompt template with filled arguments"""

    if name == "find_vendor":
        need = arguments.get("need", "general supplies") if arguments else "general supplies"
        budget = arguments.get("budget", "not specified") if arguments else "not specified"

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""I need to find a vendor for the following procurement need:

Need: {need}
Budget: {budget}

Please:
1. Search for vendors that might fulfill this need
2. Compare their risk scores and payment terms
3. Recommend the best option with justification"""
                )
            )
        ]

    elif name == "create_po_workflow":
        vendor_id = arguments.get("vendor_id") if arguments else None
        if not vendor_id:
            raise ValueError("vendor_id is required")

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Help me create a purchase order for vendor {vendor_id}.

Please guide me through:
1. First, look up the vendor details to confirm they're active
2. Help me specify the items I want to order
3. Create the purchase order with appropriate approval routing
4. Summarize the created PO and next steps"""
                )
            )
        ]

    raise ValueError(f"Unknown prompt: {name}")


# =============================================================================
# Server Entry Point
# =============================================================================

async def main():
    """Run the MCP server using stdio transport with graceful shutdown."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP library required. Install with: pip install mcp")

    # Setup graceful shutdown
    shutdown = GracefulShutdown()
    loop = asyncio.get_running_loop()
    shutdown.setup_signals(loop)

    logger.info("Starting Procurement MCP Server")

    try:
        async with stdio_server() as (read_stream, write_stream):
            # Create a task for the server
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options()
                )
            )

            # Wait for either server completion or shutdown signal
            shutdown_task = asyncio.create_task(shutdown.shutdown_event.wait())

            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            logger.info("Server shutdown complete")
    finally:
        shutdown.cleanup_signals()


if __name__ == "__main__":
    asyncio.run(main())
