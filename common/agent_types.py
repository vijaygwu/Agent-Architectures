"""
Common types used across Book 1 code examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Protocol, runtime_checkable

# Default model - configurable via environment variable
DEFAULT_MODEL: Final[str] = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4")


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for agent tools."""

    name: str
    description: str

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agents."""

    name: str

    def run(self, task: str) -> Any:
        """Run the agent on a task."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Return health status for monitoring."""
        ...


class AgentRole(Enum):
    """Standard agent roles in multi-agent systems."""
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    VALIDATOR = "validator"
    GUARDIAN = "guardian"
    SPECIALIST = "specialist"


class MessageType(Enum):
    """Types of messages exchanged between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HANDOFF = "handoff"
    STATUS = "status"
    ERROR = "error"


class TaskStatus(Enum):
    """Status of a task in the system."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    role: AgentRole
    status: str
    current_task: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    status: TaskStatus
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """Definition of a tool available to an agent."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Any] = None


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    name: str
    role: AgentRole
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[Tool] = field(default_factory=list)
    system_prompt: str = ""


class BaseAgent:
    """Base class for all agents providing common functionality."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.memory: Dict[str, Any] = {}
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in config.tools}
        self._init_time: datetime = datetime.now(timezone.utc)
        self._last_activity: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return self.config.name

    def run(self, task: str) -> Any:
        """Run the agent on a task. Subclasses should override this method.

        Args:
            task: The task to execute

        Returns:
            The result of executing the task

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement run()")

    def health_check(self) -> dict[str, Any]:
        """Return health status for monitoring.

        Returns:
            dict with keys:
            - status: "healthy" | "degraded" | "unhealthy"
            - memory_items: number of items in memory
            - tools_available: list of available tool names
            - last_activity: ISO timestamp of last action (if tracked)
            - uptime_seconds: time since agent initialization (if tracked)
        """
        import sys

        status = "healthy"
        memory_items = len(self.memory) if hasattr(self, 'memory') and self.memory else 0
        tools_available = list(self.tools.keys()) if hasattr(self, 'tools') and self.tools else []

        # Check for potential issues
        if hasattr(self, 'memory') and isinstance(self.memory, dict):
            if len(self.memory) > 10000:  # Arbitrary threshold
                status = "degraded"

        # Calculate uptime
        uptime_seconds = None
        if hasattr(self, '_init_time') and self._init_time:
            uptime_seconds = (datetime.now(timezone.utc) - self._init_time).total_seconds()

        # Get last activity timestamp
        last_activity = None
        if hasattr(self, '_last_activity') and self._last_activity:
            last_activity = self._last_activity.isoformat()

        return {
            "status": status,
            "memory_items": memory_items,
            "tools_available": tools_available,
            "tools_count": len(tools_available),
            "python_version": sys.version_info[:2],
            "last_activity": last_activity,
            "uptime_seconds": uptime_seconds,
        }

    def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity = datetime.now(timezone.utc)
