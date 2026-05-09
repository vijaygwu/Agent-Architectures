"""
Chapter 4: Agent Communication Protocols
A2A (Agent-to-Agent) Implementation
====================================

Complete implementation of the A2A protocol for agent discovery and communication.

Usage:
    python a2a_client.py

Production Notes:
- Structured logging for all requests/responses
- Timing metrics for API calls
- Error context preserved in logs
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
from enum import Enum
import aiohttp

# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
except ImportError:
    # Inline fallback implementation
    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        pass

    class CircuitBreaker:
        """Simple circuit breaker implementation.

        States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Circuit is open, requests fail fast
        - HALF_OPEN: Testing if service has recovered
        """

        def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failure_count = 0
            self._last_failure_time: float | None = None
            self._state = "CLOSED"

        @property
        def state(self) -> str:
            if self._state == "OPEN":
                # Check if recovery timeout has passed
                if self._last_failure_time and (time.time() - self._last_failure_time) >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
            return self._state

        def record_success(self):
            """Record a successful call."""
            self._failure_count = 0
            self._state = "CLOSED"

        def record_failure(self):
            """Record a failed call."""
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"

        def allow_request(self) -> bool:
            """Check if a request should be allowed."""
            current_state = self.state
            if current_state == "CLOSED":
                return True
            elif current_state == "HALF_OPEN":
                return True  # Allow one request to test recovery
            else:  # OPEN
                return False

        async def __aenter__(self):
            if not self.allow_request():
                raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is open")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.record_success()
            else:
                self.record_failure()
            return False


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredLogger:
    """JSON-formatted structured logger for production observability."""

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
# Client Metrics
# =============================================================================

class ClientMetrics:
    """Metrics collection for A2A client operations.

    Tracks:
    - request_count: Total API requests made
    - error_count: Total errors encountered
    - latency_by_operation: Latency stats per operation type
    """

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latency_by_operation: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()

    def record_request(self, operation: str, latency: float, error: bool = False):
        """Record a request with its latency."""
        self.request_count += 1
        self.latency_by_operation[operation].append(latency)
        if error:
            self.error_count += 1

    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation."""
        latencies = self.latency_by_operation.get(operation, [])
        if not latencies:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        return {
            "count": len(latencies),
            "avg_ms": round(sum(latencies) / len(latencies) * 1000, 2),
            "min_ms": round(min(latencies) * 1000, 2),
            "max_ms": round(max(latencies) * 1000, 2)
        }

    def to_dict(self) -> dict:
        """Export all metrics."""
        return {
            "uptime_seconds": time.time() - self._start_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "operations": {op: self.get_stats(op) for op in self.latency_by_operation}
        }


# Initialize logger and metrics
logger = StructuredLogger("a2a-client")
client_metrics = ClientMetrics()


# Default timeout for HTTP operations (configurable via environment)
DEFAULT_TIMEOUT = float(os.environ.get("A2A_CLIENT_TIMEOUT", "30.0"))

# Simple retry helper for HTTP operations
async def _retry_http(coro_func, max_attempts=3, base_delay=1.0):
    """Retry HTTP operation with exponential backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except (aiohttp.ClientError, TimeoutError) as e:
            last_error = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_error


class A2AError(Exception):
    """Exception raised for A2A protocol errors."""
    pass


# A2A Types

class TaskState(str, Enum):
    PENDING = "pending"
    WORKING = "working"  # A2A spec terminology
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Skill:
    id: str
    name: str
    description: str
    inputModes: list[str] = field(default_factory=lambda: ["text"])
    outputModes: list[str] = field(default_factory=lambda: ["text"])

@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    skills: list[Skill] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": [asdict(s) for s in self.skills]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        skills = [Skill(**s) for s in data.get("skills", [])]
        return cls(
            name=data["name"],
            description=data["description"],
            url=data["url"],
            version=data.get("version", "1.0.0"),
            capabilities=data.get("capabilities", {}),
            skills=skills
        )

@dataclass
class Task:
    id: str
    skill: str
    input: dict
    state: TaskState = TaskState.PENDING
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "skill": self.skill,
            "input": self.input,
            "state": self.state.value,
            "metadata": self.metadata,
            "createdAt": self.created_at
        }

@dataclass
class Artifact:
    id: str
    task_id: str
    type: str
    content: Any
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "taskId": self.task_id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "createdAt": self.created_at
        }

# A2A Client

class A2AClient:
    """Client for discovering and communicating with A2A agents.

    Includes structured logging and timing metrics for all operations.
    Features circuit breaker pattern and connection pooling for resilience.
    """

    def __init__(self, max_known_agents: int = 1000, timeout: float | None = None,
                 max_connections: int = 100, max_connections_per_host: int = 20,
                 keepalive_timeout: float = 30.0):
        self.known_agents: dict[str, AgentCard] = {}
        self._max_known_agents = max_known_agents
        self.timeout = timeout
        self.session: aiohttp.ClientSession | None = None
        self.request_id = None  # Correlation ID for current operation
        # Connection pooling configuration
        self._max_connections = max_connections
        self._max_connections_per_host = max_connections_per_host
        self._keepalive_timeout = keepalive_timeout
        # Circuit breakers for agent endpoints
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        logger.info(
            "A2A client initialized",
            max_known_agents=max_known_agents,
            timeout=timeout or DEFAULT_TIMEOUT,
            max_connections=max_connections,
            max_connections_per_host=max_connections_per_host,
            keepalive_timeout=keepalive_timeout
        )

    def _get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create a circuit breaker for the given endpoint."""
        if endpoint not in self._circuit_breakers:
            self._circuit_breakers[endpoint] = CircuitBreaker(
                name=f"a2a_{endpoint}",
                failure_threshold=5,
                recovery_timeout=60.0
            )
        return self._circuit_breakers[endpoint]

    async def __aenter__(self):
        effective_timeout = self.timeout or DEFAULT_TIMEOUT
        timeout = aiohttp.ClientTimeout(total=effective_timeout)
        # Configure connection pooling for better performance and resource management
        connector = aiohttp.TCPConnector(
            limit=self._max_connections,  # max total connections
            limit_per_host=self._max_connections_per_host,  # max per host
            keepalive_timeout=self._keepalive_timeout
        )
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        logger.info(
            "A2A client session started",
            timeout=effective_timeout,
            max_connections=self._max_connections,
            max_connections_per_host=self._max_connections_per_host
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
            logger.info("A2A client session closed", metrics=client_metrics.to_dict())

    def _new_request_id(self) -> str:
        """Generate a new correlation ID for request tracking."""
        self.request_id = str(uuid.uuid4())[:8]
        return self.request_id

    async def discover_agent(self, url: str, timeout: float | None = None) -> AgentCard:
        """Fetch an agent's card from its well-known URL."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with A2AClient()' context manager.")

        request_id = self._new_request_id()
        start_time = time.time()
        card_url = f"{url.rstrip('/')}/.well-known/agent.json"
        effective_timeout = timeout or self.timeout or DEFAULT_TIMEOUT

        logger.info(
            "Discovering agent",
            request_id=request_id,
            url=url,
            card_url=card_url,
            timeout=effective_timeout
        )

        circuit_breaker = self._get_circuit_breaker(url)
        try:
            async with circuit_breaker:
                request_timeout = aiohttp.ClientTimeout(total=effective_timeout)
                async with self.session.get(card_url, timeout=request_timeout) as response:
                    latency = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        card = AgentCard.from_dict(data)
                        # Evict oldest if at capacity
                        if len(self.known_agents) >= self._max_known_agents:
                            oldest_key = next(iter(self.known_agents))
                            del self.known_agents[oldest_key]
                        self.known_agents[url] = card

                        client_metrics.record_request("discover_agent", latency)
                        logger.info(
                            "Agent discovered",
                            request_id=request_id,
                            agent_name=card.name,
                            skills=len(card.skills),
                            latency_ms=round(latency * 1000, 2)
                        )
                        return card
                    else:
                        client_metrics.record_request("discover_agent", latency, error=True)
                        logger.error(
                            "Agent discovery failed",
                            request_id=request_id,
                            url=url,
                            status=response.status,
                            latency_ms=round(latency * 1000, 2)
                        )
                        raise A2AError(
                            f"Failed to discover agent at {url}: HTTP {response.status}")
        except CircuitBreakerOpen as e:
            latency = time.time() - start_time
            client_metrics.record_request("discover_agent", latency, error=True)
            logger.warning(
                "Circuit breaker open for agent",
                request_id=request_id,
                url=url,
                circuit_breaker=circuit_breaker.name,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Circuit breaker open for agent at {url}") from e
        except aiohttp.ClientError as e:
            latency = time.time() - start_time
            client_metrics.record_request("discover_agent", latency, error=True)
            logger.error(
                "Agent discovery network error",
                request_id=request_id,
                url=url,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Network error discovering agent at {url}: {e}") from e

    async def send_task(self, agent_url: str, skill: str, input_data: dict,
                        metadata: dict = None, timeout: float | None = None) -> Task:
        """Send a task to an agent with logging and metrics."""
        request_id = self._new_request_id()
        start_time = time.time()
        effective_timeout = timeout or self.timeout or DEFAULT_TIMEOUT

        task = Task(
            id=str(uuid.uuid4()),
            skill=skill,
            input=input_data,
            metadata=metadata or {}
        )

        endpoint = f"{agent_url.rstrip('/')}/tasks"

        logger.info(
            "Sending task",
            request_id=request_id,
            task_id=task.id,
            skill=skill,
            agent_url=agent_url,
            timeout=effective_timeout
        )

        circuit_breaker = self._get_circuit_breaker(agent_url)
        try:
            async with circuit_breaker:
                request_timeout = aiohttp.ClientTimeout(total=effective_timeout)
                async with self.session.post(endpoint, json=task.to_dict(), timeout=request_timeout) as response:
                    latency = time.time() - start_time
                    if response.status in (200, 201, 202):
                        data = await response.json()
                        task.state = TaskState(data.get("state", "pending"))

                        client_metrics.record_request("send_task", latency)
                        logger.info(
                            "Task sent successfully",
                            request_id=request_id,
                            task_id=task.id,
                            state=task.state.value,
                            latency_ms=round(latency * 1000, 2)
                        )
                        return task
                    else:
                        error = await response.text()
                        client_metrics.record_request("send_task", latency, error=True)
                        logger.error(
                            "Task send failed",
                            request_id=request_id,
                            task_id=task.id,
                            status=response.status,
                            error=error,
                            latency_ms=round(latency * 1000, 2)
                        )
                        raise A2AError(f"Failed to send task: {error}")
        except CircuitBreakerOpen as e:
            latency = time.time() - start_time
            client_metrics.record_request("send_task", latency, error=True)
            logger.warning(
                "Circuit breaker open for task send",
                request_id=request_id,
                task_id=task.id,
                agent_url=agent_url,
                circuit_breaker=circuit_breaker.name,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Circuit breaker open for agent at {agent_url}") from e
        except aiohttp.ClientError as e:
            latency = time.time() - start_time
            client_metrics.record_request("send_task", latency, error=True)
            logger.error(
                "Task send network error",
                request_id=request_id,
                task_id=task.id,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Network error sending task: {e}") from e

    async def get_task_status(self, agent_url: str, task_id: str, timeout: float | None = None) -> Task:
        """Get the status of a task with logging and metrics."""
        start_time = time.time()
        endpoint = f"{agent_url.rstrip('/')}/tasks/{task_id}"
        effective_timeout = timeout or self.timeout or DEFAULT_TIMEOUT

        circuit_breaker = self._get_circuit_breaker(agent_url)
        try:
            async with circuit_breaker:
                request_timeout = aiohttp.ClientTimeout(total=effective_timeout)
                async with self.session.get(endpoint, timeout=request_timeout) as response:
                    latency = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        task = Task(
                            id=data["id"],
                            skill=data["skill"],
                            input=data["input"],
                            state=TaskState(data["state"]),
                            metadata=data.get("metadata", {})
                        )

                        client_metrics.record_request("get_task_status", latency)
                        logger.debug(
                            "Task status retrieved",
                            task_id=task_id,
                            state=task.state.value,
                            latency_ms=round(latency * 1000, 2)
                        )
                        return task
                    else:
                        client_metrics.record_request("get_task_status", latency, error=True)
                        logger.error(
                            "Task status retrieval failed",
                            task_id=task_id,
                            status=response.status,
                            latency_ms=round(latency * 1000, 2)
                        )
                        raise A2AError(f"Task not found: {task_id}")
        except CircuitBreakerOpen as e:
            latency = time.time() - start_time
            client_metrics.record_request("get_task_status", latency, error=True)
            logger.warning(
                "Circuit breaker open for task status",
                task_id=task_id,
                agent_url=agent_url,
                circuit_breaker=circuit_breaker.name,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Circuit breaker open for agent at {agent_url}") from e
        except aiohttp.ClientError as e:
            latency = time.time() - start_time
            client_metrics.record_request("get_task_status", latency, error=True)
            logger.error(
                "Task status network error",
                task_id=task_id,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency * 1000, 2)
            )
            raise A2AError(f"Network error getting task status: {e}") from e

    async def get_task_artifacts(self, agent_url: str,
                                  task_id: str, timeout: float | None = None) -> list[Artifact]:
        """Get artifacts produced by a task."""
        endpoint = f"{agent_url.rstrip('/')}/tasks/{task_id}/artifacts"
        effective_timeout = timeout or self.timeout or DEFAULT_TIMEOUT
        request_timeout = aiohttp.ClientTimeout(total=effective_timeout)

        circuit_breaker = self._get_circuit_breaker(agent_url)
        try:
            async with circuit_breaker:
                async with self.session.get(endpoint, timeout=request_timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            Artifact(
                                id=a["id"],
                                task_id=a["taskId"],
                                type=a["type"],
                                content=a["content"],
                                metadata=a.get("metadata", {})
                            )
                            for a in data.get("artifacts", [])
                        ]
                    else:
                        return []
        except CircuitBreakerOpen:
            logger.warning(
                "Circuit breaker open for artifacts retrieval",
                task_id=task_id,
                agent_url=agent_url,
                circuit_breaker=circuit_breaker.name
            )
            return []

    async def wait_for_completion(self, agent_url: str, task_id: str,
                                   timeout: float = 300,
                                   poll_interval: float = 1,
                                   shutdown_event: asyncio.Event | None = None) -> Task:
        """Wait for a task to complete.

        Args:
            agent_url: URL of the agent handling the task.
            task_id: ID of the task to wait for.
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between status checks in seconds.
            shutdown_event: Optional event to signal graceful shutdown. When set,
                           the method will return early with the current task state.

        Returns:
            The task in its final (or current, if shutdown requested) state.

        Raises:
            TimeoutError: If the task does not complete within the timeout.
        """
        start = asyncio.get_running_loop().time()

        while True:
            # Check for graceful shutdown request
            if shutdown_event is not None and shutdown_event.is_set():
                return await self.get_task_status(agent_url, task_id)

            task = await self.get_task_status(agent_url, task_id)

            terminal_states = (
                TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)
            if task.state in terminal_states:
                return task

            elapsed = asyncio.get_running_loop().time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} did not complete within {timeout}s") from None

            await asyncio.sleep(poll_interval)

# A2A Server

class A2AServer:
    """Server that exposes an agent via the A2A protocol."""

    def __init__(self, card: AgentCard, task_handler, max_tasks: int = 10000):
        self.card = card
        self.task_handler = task_handler
        self.tasks: dict[str, Task] = {}
        self.artifacts: dict[str, list[Artifact]] = {}
        self._max_tasks = max_tasks

    async def handle_agent_card(self, request) -> dict:
        """Handle GET /.well-known/agent.json"""
        return self.card.to_dict()

    async def handle_create_task(self, request) -> dict:
        """Handle POST /tasks"""
        data = await request.json()

        task = Task(
            id=data.get("id", str(uuid.uuid4())),
            skill=data["skill"],
            input=data["input"],
            metadata=data.get("metadata", {})
        )

        # Validate skill exists
        skill_ids = [s.id for s in self.card.skills]
        if task.skill not in skill_ids:
            raise ValueError(f"Unknown skill: {task.skill}")

        # Evict completed tasks if at capacity
        if len(self.tasks) >= self._max_tasks:
            completed = [tid for tid, t in self.tasks.items()
                        if t.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)]
            for tid in completed[:100]:
                del self.tasks[tid]
                self.artifacts.pop(tid, None)
        self.tasks[task.id] = task
        # Bound artifacts dict like tasks - clean up old entries
        if len(self.artifacts) >= self._max_tasks:
            completed = [tid for tid, t in self.tasks.items()
                        if t.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)]
            for tid in completed[:100]:
                self.artifacts.pop(tid, None)
        self.artifacts[task.id] = []

        # Start processing asynchronously
        asyncio.create_task(self._process_task(task))

        return task.to_dict()

    async def handle_get_task(self, request, task_id: str) -> dict:
        """Handle GET /tasks/{task_id}"""
        if task_id not in self.tasks:
            raise KeyError(f"Task not found: {task_id}")
        return self.tasks[task_id].to_dict()

    async def handle_get_artifacts(self, request, task_id: str) -> dict:
        """Handle GET /tasks/{task_id}/artifacts"""
        artifacts = self.artifacts.get(task_id, [])
        return {"artifacts": [a.to_dict() for a in artifacts]}

    async def _process_task(self, task: Task):
        """Process a task using the handler."""
        task.state = TaskState.WORKING

        try:
            result = await self.task_handler(task)

            # Create artifact from result
            artifact = Artifact(
                id=str(uuid.uuid4()),
                task_id=task.id,
                type="result",
                content=result
            )
            self.artifacts[task.id].append(artifact)
            task.state = TaskState.COMPLETED

        except Exception as e:
            task.state = TaskState.FAILED
            task.metadata["error"] = str(e)


async def example_task_handler(task: Task) -> dict:
    """Example handler that processes research tasks."""
    query = task.input.get("query", "")

    # Simulate research
    await asyncio.sleep(1)

    return {
        "findings": [
            f"Finding 1 about {query}",
            f"Finding 2 about {query}",
        ],
        "confidence": 0.85
    }

async def demo():
    """Demonstrate A2A protocol usage."""

    # Create an agent card
    card = AgentCard(
        name="Research Assistant",
        description="Finds and analyzes information",
        url="http://localhost:8080",
        skills=[
            Skill(
                id="web_research",
                name="Web Research",
                description="Search and analyze web content"
            )
        ],
        capabilities={"streaming": False}
    )

    print("Agent Card:")
    print(json.dumps(card.to_dict(), indent=2))

    # Create server
    server = A2AServer(card, example_task_handler)

    # Simulate a task
    task = Task(
        id="test_task_1",
        skill="web_research",
        input={"query": "AI trends 2024"}
    )

    server.tasks[task.id] = task
    server.artifacts[task.id] = []

    # Process the task
    await server._process_task(task)

    print(f"\nTask state: {task.state.value}")
    print(f"Artifacts: {[a.to_dict() for a in server.artifacts[task.id]]}")

if __name__ == "__main__":
    asyncio.run(demo())
