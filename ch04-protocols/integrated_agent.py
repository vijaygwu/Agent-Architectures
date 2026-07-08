"""
Chapter 4: Agent Communication Protocols
Integrated Agent Implementation
================================

An agent that uses MCP for tools and A2A for inter-agent communication.

Usage:
    python integrated_agent.py
"""

__all__ = [
    "retry_with_exponential_backoff",
    "IntegratedAgent",
]

import asyncio
import logging
import os
from typing import Any
from mcp_client import MCPClient
from a2a_client_book import A2AClient, AgentCard, TaskState

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
except ImportError:
    # Inline fallback
    class CircuitBreakerOpen(Exception):
        pass

    class CircuitBreaker:
        def __init__(self, **kwargs):
            import time as _time
            self._time = _time
            self._failures = 0
            self._threshold = kwargs.get('failure_threshold', 5)
            self._recovery_timeout = kwargs.get('recovery_timeout', 30.0)
            self._last_failure_time = 0.0

        def allow(self):
            if self._failures < self._threshold:
                return True
            # Half-open: allow a test call after the recovery timeout
            elapsed = self._time.time() - self._last_failure_time
            return elapsed >= self._recovery_timeout

        def record_success(self):
            self._failures = 0

        def record_failure(self):
            self._failures += 1
            self._last_failure_time = self._time.time()


# Module-level circuit breaker for LLM/tool calls
_llm_circuit_breaker = CircuitBreaker(
    name="integrated_agent_llm",
    failure_threshold=3,
    recovery_timeout=60.0
)

# Set up logging
logger = logging.getLogger("integrated_agent")

# Environment-configurable timeout for task delegation (in seconds)
DELEGATE_TIMEOUT = float(os.environ.get("DELEGATE_TASK_TIMEOUT", "300"))

# Environment-configurable timeout for LLM/tool calls (in seconds)
LLM_CALL_TIMEOUT = float(os.environ.get("LLM_CALL_TIMEOUT", "60"))

# Environment-configurable timeout for MCP server connection (in seconds)
MCP_CONNECT_TIMEOUT = float(os.environ.get("MCP_CONNECT_TIMEOUT", "30"))

# Environment-configurable timeout for agent discovery (in seconds)
AGENT_DISCOVERY_TIMEOUT = float(os.environ.get("AGENT_DISCOVERY_TIMEOUT", "30"))


# =============================================================================
# Retry with Exponential Backoff for Task Delegation
# =============================================================================
# Task delegation to other agents can fail due to:
# - Network issues (transient)
# - Target agent temporarily overloaded (transient)
# - Target agent restarting (transient)
#
# Retry with exponential backoff handles these transient failures by:
# 1. Waiting progressively longer between retries
# 2. Adding jitter to prevent thundering herd
# 3. Giving up after a configurable number of attempts
#
# This pattern is essential for robust inter-agent communication.
# =============================================================================

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (asyncio.TimeoutError, ConnectionError, RuntimeError)
):
    """
    Execute an async function with retry and exponential backoff.

    This is a production-ready retry pattern that handles transient failures
    gracefully. The delay between retries grows exponentially:
    - Attempt 1: immediate
    - Attempt 2: base_delay seconds (1s default)
    - Attempt 3: base_delay * exponential_base seconds (2s default)
    - Attempt 4: base_delay * exponential_base^2 seconds (4s default)
    - ...up to max_delay

    Args:
        func: Async function to execute (should be a zero-arg callable)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        retryable_exceptions: Tuple of exceptions that should trigger retry

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail

    Example:
        result = await retry_with_exponential_backoff(
            lambda: client.send_task(url, skill, data),
            max_retries=3,
            retryable_exceptions=(asyncio.TimeoutError, ConnectionError)
        )
    """
    import random

    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)

                # Add jitter (0.5x to 1.5x) to prevent thundering herd
                jitter = delay * (0.5 + random.random())

                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} failed with {type(e).__name__}: {e}. "
                    f"Retrying in {jitter:.2f}s"
                )
                await asyncio.sleep(jitter)
            else:
                logger.error(
                    f"All {max_retries} retry attempts exhausted. Last error: {e}"
                )

    raise last_exception


class IntegratedAgent:
    """An agent that uses MCP for tools and A2A for inter-agent communication.

    This class demonstrates protocol integration by combining:
    - MCP (Model Context Protocol) for tool access
    - A2A (Agent-to-Agent) for inter-agent communication

    Usage:
        async with IntegratedAgent(card) as agent:
            await agent.add_tool_server(...)
    """

    def __init__(self, agent_card: AgentCard):
        """Initialize the integrated agent.

        Args:
            agent_card: The A2A agent card describing this agent's capabilities.
        """
        self.card = agent_card
        self.mcp_clients: dict[str, MCPClient] = {}
        self._max_mcp_clients = 100
        self.a2a_client = A2AClient()
        self._initialized = False

    async def __aenter__(self):
        """Initialize async resources."""
        await self.a2a_client.__aenter__()
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        await self.a2a_client.__aexit__(exc_type, exc_val, exc_tb)
        for client in self.mcp_clients.values():
            await client.close()
        self._initialized = False

    async def add_tool_server(
        self, name: str, command: list[str], timeout: float | None = None
    ):
        """Connect to an MCP tool server.

        Args:
            name: A unique name to identify this tool server.
            command: The command and arguments to start the MCP server process.
            timeout: Maximum time to wait for connection (default: MCP_CONNECT_TIMEOUT).

        Raises:
            TimeoutError: If connection times out.
            ValueError: If maximum MCP clients reached.
        """
        if len(self.mcp_clients) >= self._max_mcp_clients:
            raise ValueError(f"Maximum MCP clients ({self._max_mcp_clients}) reached")
        client = MCPClient()
        connect_timeout = timeout if timeout is not None else MCP_CONNECT_TIMEOUT
        try:
            async with asyncio.timeout(connect_timeout):
                await client.connect_stdio(command)
        except TimeoutError:
            logger.warning(
                f"MCP server connection timed out after {connect_timeout}s for '{name}'"
            )
            raise TimeoutError(
                f"MCP server connection timed out after {connect_timeout}s"
            )
        self.mcp_clients[name] = client

    async def discover_agent(
        self, url: str, timeout: float | None = None
    ) -> AgentCard:
        """Discover another agent via A2A.

        Args:
            url: The base URL of the agent to discover.
            timeout: Maximum time to wait for discovery (default: AGENT_DISCOVERY_TIMEOUT).

        Returns:
            The discovered agent's AgentCard.

        Raises:
            TimeoutError: If discovery times out.
        """
        discovery_timeout = timeout if timeout is not None else AGENT_DISCOVERY_TIMEOUT
        try:
            async with asyncio.timeout(discovery_timeout):
                return await self.a2a_client.discover_agent(url)
        except TimeoutError:
            logger.warning(
                f"Agent discovery timed out after {discovery_timeout}s for '{url}'"
            )
            raise TimeoutError(
                f"Agent discovery timed out after {discovery_timeout}s"
            )

    async def call_tool(
        self, server: str, tool: str, args: dict, timeout: float | None = None
    ) -> Any:
        """Call a tool via MCP.

        Args:
            server: The name of the MCP server (as registered with add_tool_server).
            tool: The name of the tool to call.
            args: Arguments to pass to the tool.
            timeout: Maximum time to wait for tool execution (default: LLM_CALL_TIMEOUT).

        Returns:
            The result from the tool execution.

        Raises:
            ValueError: If the server name is not registered.
            CircuitBreakerOpen: If the circuit breaker is open due to too many failures.
            TimeoutError: If the tool call times out.
        """
        client = self.mcp_clients.get(server)
        if not client:
            raise ValueError(f"Unknown tool server: {server}")

        # Check circuit breaker before making the call
        if not _llm_circuit_breaker.allow():
            raise CircuitBreakerOpen("integrated_agent_llm", 60.0)

        call_timeout = timeout if timeout is not None else LLM_CALL_TIMEOUT
        try:
            async with asyncio.timeout(call_timeout):
                result = await client.call_tool(tool, args)
            _llm_circuit_breaker.record_success()
            return result
        except TimeoutError:
            _llm_circuit_breaker.record_failure()
            logger.warning(
                f"Tool call timed out after {call_timeout}s: {server}/{tool}"
            )
            raise TimeoutError(
                f"Tool call '{tool}' on server '{server}' timed out after {call_timeout}s"
            )
        except Exception as e:
            _llm_circuit_breaker.record_failure()
            raise

    async def delegate_task(
        self,
        agent_url: str,
        skill: str,
        input_data: dict,
        timeout: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """Delegate a task to another agent via A2A with retry logic.

        This method includes retry with exponential backoff to handle transient
        failures in inter-agent communication. Retries are appropriate for:
        - Network connectivity issues
        - Target agent temporarily overloaded
        - Transient service errors

        The exponential backoff prevents overwhelming a recovering service
        and spreads out retry attempts from multiple callers.

        Args:
            agent_url: The URL of the target agent.
            skill: The skill/capability to invoke on the target agent.
            input_data: Input data for the task.
            timeout: Maximum time to wait for completion (default: 300 seconds).
            max_retries: Maximum retry attempts for transient failures (default: 3).
            retry_delay: Base delay between retries in seconds (default: 1.0).

        Returns:
            The result from the delegated task.

        Raises:
            RuntimeError: If the task fails after all retries.
            asyncio.TimeoutError: If the task times out after all retries.
        """
        # Submit the task exactly once: re-sending on retry would
        # create a new task (and duplicate side effects) on the
        # target agent for non-idempotent skills.
        task = await self.a2a_client.send_task(agent_url, skill, input_data)

        async def _await_completion():
            """Inner function for retry wrapper (polling only)."""
            completed = await asyncio.wait_for(
                self.a2a_client.wait_for_completion(agent_url, task.id),
                timeout=timeout or DELEGATE_TIMEOUT
            )

            if completed.state == TaskState.COMPLETED:
                artifacts = await self.a2a_client.get_task_artifacts(agent_url, task.id)
                return artifacts[0].content if artifacts else None
            else:
                # Task failed on the target agent. Not retryable:
                # resubmission could duplicate side effects.
                error_msg = completed.metadata.get('error', 'Unknown error')
                raise RuntimeError(f"Task failed: {error_msg}")

        # Retry only the polling/wait step for the SAME task id.
        # Retryable exceptions:
        # - asyncio.TimeoutError: Network/service temporarily slow
        # - ConnectionError: Network connectivity issues
        # RuntimeError (task failure) is deliberately NOT retried.
        return await retry_with_exponential_backoff(
            _await_completion,
            max_retries=max_retries,
            base_delay=retry_delay,
            max_delay=30.0,
            exponential_base=2.0,
            retryable_exceptions=(asyncio.TimeoutError, ConnectionError)
        )

    def get_all_tools(self) -> list[dict]:
        """Get all available tools for LLM function calling.

        Returns:
            A list of tool definitions in a format suitable for LLM function calling,
            with each tool annotated with its source server name.
        """
        tools = []
        for name, client in self.mcp_clients.items():
            for tool in client.get_tools_for_llm():
                tool["server"] = name
                tools.append(tool)
        return tools
