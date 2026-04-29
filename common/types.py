"""
Common types used across Book 1 code examples.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# Default model - update this when new models are released
DEFAULT_MODEL = "claude-sonnet-4-20250514"


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
