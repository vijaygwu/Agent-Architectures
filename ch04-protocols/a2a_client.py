"""
Chapter 4: Agent-to-Agent (A2A) Protocol Implementation
========================================================
Implementation of Google's A2A protocol for agent interoperability.

This module provides:
1. Agent Card definition and validation
2. A2A client for agent discovery and communication
3. Task delegation patterns

Reference: https://github.com/google/A2A
"""

import asyncio
import json
import uuid
import httpx
from datetime import datetime
from typing import Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# A2A Protocol Data Types
# =============================================================================

class TaskState(str, Enum):
    """Task lifecycle states per A2A specification"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message roles in A2A conversations"""
    USER = "user"
    AGENT = "agent"


@dataclass
class AgentCapability:
    """
    Describes a specific capability an agent provides.

    Capabilities are used for:
    1. Agent discovery - finding agents that can handle specific tasks
    2. Capability negotiation - determining interaction modalities
    3. Task routing - directing requests to appropriate agents
    """
    name: str
    description: str
    input_schema: dict | None = None
    output_schema: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AgentAuthentication:
    """Authentication configuration for agent access"""
    type: Literal["none", "api_key", "oauth2", "mtls"]
    scopes: list[str] = field(default_factory=list)
    token_url: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AgentCard:
    """
    Agent Card - the A2A discovery document.

    Agent Cards describe an agent's capabilities, endpoint, and
    authentication requirements. They enable other agents to:
    1. Discover what the agent can do
    2. Understand how to authenticate
    3. Know what interaction modes are supported

    This is the A2A equivalent of an API specification.
    """
    name: str
    description: str
    version: str
    endpoint: str
    capabilities: list[AgentCapability]
    authentication: AgentAuthentication

    # Optional metadata
    owner: str | None = None
    documentation_url: str | None = None
    supported_modes: list[str] = field(default_factory=lambda: ["text"])
    max_concurrent_tasks: int = 10

    def to_dict(self) -> dict:
        """Serialize to A2A-compatible JSON"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "endpoint": self.endpoint,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "authentication": self.authentication.to_dict(),
            "owner": self.owner,
            "documentation_url": self.documentation_url,
            "supported_modes": self.supported_modes,
            "max_concurrent_tasks": self.max_concurrent_tasks
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        """Deserialize from JSON"""
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            endpoint=data["endpoint"],
            capabilities=[
                AgentCapability(**c) for c in data["capabilities"]
            ],
            authentication=AgentAuthentication(**data["authentication"]),
            owner=data.get("owner"),
            documentation_url=data.get("documentation_url"),
            supported_modes=data.get("supported_modes", ["text"]),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10)
        )

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(c.name == capability_name for c in self.capabilities)


@dataclass
class A2AMessage:
    """Message in an A2A conversation"""
    id: str
    role: MessageRole
    content: str
    timestamp: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class A2ATask:
    """
    Task delegated between agents via A2A.

    Tasks represent units of work that one agent delegates to another.
    They include full conversation history and metadata for tracking.
    """
    id: str
    source_agent: str
    target_agent: str
    capability: str
    state: TaskState
    messages: list[A2AMessage]
    created_at: str
    updated_at: str
    result: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "capability": self.capability,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error
        }


# =============================================================================
# A2A Client Implementation
# =============================================================================

class A2AClient:
    """
    Client for A2A protocol communication.

    Handles:
    1. Agent discovery via Agent Cards
    2. Task creation and delegation
    3. Response handling (sync, streaming, async)
    4. Authentication
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        timeout: float = 30.0
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.timeout = timeout
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self._known_agents: dict[str, AgentCard] = {}
        self._active_tasks: dict[str, A2ATask] = {}

    async def close(self):
        """Close HTTP client"""
        await self._http_client.aclose()

    # -------------------------------------------------------------------------
    # Agent Discovery
    # -------------------------------------------------------------------------

    async def discover_agent(self, endpoint: str) -> AgentCard:
        """
        Discover an agent by fetching its Agent Card.

        The Agent Card is typically served at /.well-known/agent.json
        or at the agent's root endpoint.
        """
        # Try well-known location first
        well_known_url = f"{endpoint.rstrip('/')}/.well-known/agent.json"

        try:
            response = await self._http_client.get(well_known_url)
            if response.status_code == 200:
                card = AgentCard.from_dict(response.json())
                self._known_agents[card.name] = card
                return card
        except httpx.RequestError:
            pass

        # Fall back to root endpoint
        response = await self._http_client.get(endpoint)
        response.raise_for_status()
        card = AgentCard.from_dict(response.json())
        self._known_agents[card.name] = card
        return card

    async def discover_from_registry(
        self,
        registry_url: str,
        capability: str | None = None
    ) -> list[AgentCard]:
        """
        Discover agents from a central registry.

        The registry acts as "DNS for agents", enabling discovery
        of agents by capability.
        """
        params = {}
        if capability:
            params["capability"] = capability

        response = await self._http_client.get(
            f"{registry_url}/agents",
            params=params
        )
        response.raise_for_status()

        agents = []
        for agent_data in response.json()["agents"]:
            card = AgentCard.from_dict(agent_data)
            self._known_agents[card.name] = card
            agents.append(card)

        return agents

    def get_known_agent(self, name: str) -> AgentCard | None:
        """Get a previously discovered agent by name"""
        return self._known_agents.get(name)

    # -------------------------------------------------------------------------
    # Task Delegation
    # -------------------------------------------------------------------------

    async def create_task(
        self,
        target_agent: str | AgentCard,
        capability: str,
        initial_message: str,
        metadata: dict | None = None
    ) -> A2ATask:
        """
        Create and send a new task to another agent.

        This is the primary method for agent-to-agent delegation.
        """
        # Resolve agent card if name provided
        if isinstance(target_agent, str):
            card = self._known_agents.get(target_agent)
            if not card:
                raise ValueError(f"Unknown agent: {target_agent}. Discover it first.")
        else:
            card = target_agent

        # Verify capability
        if not card.has_capability(capability):
            raise ValueError(
                f"Agent {card.name} does not have capability: {capability}"
            )

        # Create task
        task_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        initial_msg = A2AMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=initial_message,
            timestamp=now,
            metadata=metadata or {}
        )

        task = A2ATask(
            id=task_id,
            source_agent=self.agent_id,
            target_agent=card.name,
            capability=capability,
            state=TaskState.PENDING,
            messages=[initial_msg],
            created_at=now,
            updated_at=now
        )

        # Send to target agent
        response = await self._http_client.post(
            f"{card.endpoint}/tasks",
            json=task.to_dict(),
            headers=await self._get_auth_headers(card)
        )
        response.raise_for_status()

        # Update task with response
        response_data = response.json()
        task.state = TaskState(response_data.get("state", "pending"))

        self._active_tasks[task_id] = task
        return task

    async def send_message(
        self,
        task_id: str,
        message: str,
        metadata: dict | None = None
    ) -> A2ATask:
        """
        Send a follow-up message in an existing task conversation.
        """
        task = self._active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        card = self._known_agents.get(task.target_agent)
        if not card:
            raise ValueError(f"Unknown agent: {task.target_agent}")

        msg = A2AMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=message,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )

        response = await self._http_client.post(
            f"{card.endpoint}/tasks/{task_id}/messages",
            json=msg.to_dict(),
            headers=await self._get_auth_headers(card)
        )
        response.raise_for_status()

        task.messages.append(msg)
        task.updated_at = datetime.utcnow().isoformat()

        # Add agent response
        response_data = response.json()
        if "message" in response_data:
            agent_msg = A2AMessage(
                id=response_data["message"]["id"],
                role=MessageRole.AGENT,
                content=response_data["message"]["content"],
                timestamp=response_data["message"]["timestamp"],
                metadata=response_data["message"].get("metadata", {})
            )
            task.messages.append(agent_msg)

        task.state = TaskState(response_data.get("state", task.state.value))

        return task

    async def get_task_status(self, task_id: str) -> A2ATask:
        """Poll for task status (for async tasks)"""
        task = self._active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        card = self._known_agents.get(task.target_agent)
        if not card:
            raise ValueError(f"Unknown agent: {task.target_agent}")

        response = await self._http_client.get(
            f"{card.endpoint}/tasks/{task_id}",
            headers=await self._get_auth_headers(card)
        )
        response.raise_for_status()

        data = response.json()
        task.state = TaskState(data["state"])
        task.result = data.get("result")
        task.error = data.get("error")
        task.updated_at = data.get("updated_at", datetime.utcnow().isoformat())

        return task

    async def cancel_task(self, task_id: str) -> A2ATask:
        """Cancel an in-progress task"""
        task = self._active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")

        card = self._known_agents.get(task.target_agent)
        if not card:
            raise ValueError(f"Unknown agent: {task.target_agent}")

        response = await self._http_client.post(
            f"{card.endpoint}/tasks/{task_id}/cancel",
            headers=await self._get_auth_headers(card)
        )
        response.raise_for_status()

        task.state = TaskState.CANCELLED
        task.updated_at = datetime.utcnow().isoformat()

        return task

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def _get_auth_headers(self, card: AgentCard) -> dict[str, str]:
        """Get authentication headers for agent communication"""
        auth = card.authentication

        if auth.type == "none":
            return {}

        elif auth.type == "api_key":
            # In production, retrieve from secure storage
            api_key = self._get_api_key(card.name)
            return {"Authorization": f"Bearer {api_key}"}

        elif auth.type == "oauth2":
            token = await self._get_oauth_token(card)
            return {"Authorization": f"Bearer {token}"}

        elif auth.type == "mtls":
            # mTLS is handled at the transport level
            return {}

        return {}

    def _get_api_key(self, agent_name: str) -> str:
        """Retrieve API key from secure storage"""
        # In production, use a secrets manager
        import os
        key = os.environ.get(f"A2A_API_KEY_{agent_name.upper()}")
        if not key:
            raise ValueError(f"No API key configured for agent: {agent_name}")
        return key

    async def _get_oauth_token(self, card: AgentCard) -> str:
        """Obtain OAuth2 token for agent authentication"""
        # Implement OAuth2 client credentials flow
        # This is a simplified example
        if not card.authentication.token_url:
            raise ValueError(f"No token URL for agent: {card.name}")

        response = await self._http_client.post(
            card.authentication.token_url,
            data={
                "grant_type": "client_credentials",
                "scope": " ".join(card.authentication.scopes)
            }
        )
        response.raise_for_status()
        return response.json()["access_token"]


# =============================================================================
# A2A Server Implementation (FastAPI)
# =============================================================================

def create_a2a_server(
    agent_card: AgentCard,
    handler: "A2ATaskHandler"
):
    """
    Create a FastAPI application that serves A2A protocol.

    This factory function creates the standard A2A endpoints:
    - GET / - Returns Agent Card
    - GET /.well-known/agent.json - Returns Agent Card
    - POST /tasks - Create new task
    - GET /tasks/{id} - Get task status
    - POST /tasks/{id}/messages - Send message
    - POST /tasks/{id}/cancel - Cancel task
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title=agent_card.name, version=agent_card.version)

    @app.get("/")
    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        """Return the Agent Card for discovery"""
        return JSONResponse(content=agent_card.to_dict())

    @app.post("/tasks")
    async def create_task(task_data: dict):
        """Handle incoming task delegation"""
        try:
            task = await handler.handle_task(task_data)
            return JSONResponse(content=task.to_dict())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get task status"""
        task = handler.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return JSONResponse(content=task.to_dict())

    @app.post("/tasks/{task_id}/messages")
    async def send_message(task_id: str, message_data: dict):
        """Handle follow-up message in task conversation"""
        try:
            task = await handler.handle_message(task_id, message_data)
            return JSONResponse(content=task.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        """Cancel an in-progress task"""
        try:
            task = await handler.cancel_task(task_id)
            return JSONResponse(content=task.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app


class A2ATaskHandler:
    """
    Base class for handling A2A tasks.

    Subclass this to implement your agent's task handling logic.
    """

    def __init__(self):
        self.tasks: dict[str, A2ATask] = {}

    async def handle_task(self, task_data: dict) -> A2ATask:
        """Handle incoming task - override in subclass"""
        raise NotImplementedError

    async def handle_message(self, task_id: str, message_data: dict) -> A2ATask:
        """Handle follow-up message - override in subclass"""
        raise NotImplementedError

    async def cancel_task(self, task_id: str) -> A2ATask:
        """Cancel task - override in subclass"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        task.state = TaskState.CANCELLED
        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        """Get task by ID"""
        return self.tasks.get(task_id)


# =============================================================================
# Example: Inventory Agent Implementation
# =============================================================================

class InventoryAgentHandler(A2ATaskHandler):
    """
    Example A2A handler for an inventory management agent.

    This demonstrates how to implement A2A protocol for a specific
    domain capability.
    """

    def __init__(self):
        super().__init__()
        # Mock inventory data
        self.inventory = {
            "SKU001": {"name": "Widget A", "quantity": 150, "reorder_point": 50},
            "SKU002": {"name": "Widget B", "quantity": 30, "reorder_point": 100},
            "SKU003": {"name": "Gadget X", "quantity": 500, "reorder_point": 200},
        }

    async def handle_task(self, task_data: dict) -> A2ATask:
        """Process incoming inventory task"""
        task_id = task_data["id"]
        capability = task_data["capability"]
        messages = task_data["messages"]

        # Parse the initial request
        initial_message = messages[0]["content"]

        # Create task record
        task = A2ATask(
            id=task_id,
            source_agent=task_data["source_agent"],
            target_agent=task_data["target_agent"],
            capability=capability,
            state=TaskState.IN_PROGRESS,
            messages=[A2AMessage(**m) for m in messages],
            created_at=task_data["created_at"],
            updated_at=datetime.utcnow().isoformat()
        )

        # Handle based on capability
        if capability == "check_stock":
            result = await self._check_stock(initial_message)
        elif capability == "reserve_inventory":
            result = await self._reserve_inventory(initial_message)
        elif capability == "reorder_alert":
            result = await self._check_reorder()
        else:
            task.state = TaskState.FAILED
            task.error = f"Unknown capability: {capability}"
            self.tasks[task_id] = task
            return task

        # Add response message
        response_msg = A2AMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.AGENT,
            content=json.dumps(result),
            timestamp=datetime.utcnow().isoformat()
        )
        task.messages.append(response_msg)
        task.state = TaskState.COMPLETED
        task.result = result

        self.tasks[task_id] = task
        return task

    async def _check_stock(self, request: str) -> dict:
        """Check stock levels for requested items"""
        # Simple parsing - in production use proper NLU
        results = {}
        for sku, data in self.inventory.items():
            results[sku] = {
                "name": data["name"],
                "available": data["quantity"],
                "low_stock": data["quantity"] < data["reorder_point"]
            }
        return {"stock_levels": results}

    async def _reserve_inventory(self, request: str) -> dict:
        """Reserve inventory for an order"""
        # Parse reservation request and process
        return {"reserved": True, "reservation_id": str(uuid.uuid4())}

    async def _check_reorder(self) -> dict:
        """Check for items needing reorder"""
        alerts = []
        for sku, data in self.inventory.items():
            if data["quantity"] < data["reorder_point"]:
                alerts.append({
                    "sku": sku,
                    "name": data["name"],
                    "current_quantity": data["quantity"],
                    "reorder_point": data["reorder_point"],
                    "suggested_order": data["reorder_point"] * 2
                })
        return {"reorder_alerts": alerts}

    async def handle_message(self, task_id: str, message_data: dict) -> A2ATask:
        """Handle follow-up messages"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Add the incoming message
        msg = A2AMessage(
            id=message_data["id"],
            role=MessageRole.USER,
            content=message_data["content"],
            timestamp=message_data["timestamp"],
            metadata=message_data.get("metadata", {})
        )
        task.messages.append(msg)

        # Generate response based on conversation context
        response_content = f"Received your follow-up. Task {task_id} context updated."

        response_msg = A2AMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.AGENT,
            content=response_content,
            timestamp=datetime.utcnow().isoformat()
        )
        task.messages.append(response_msg)
        task.updated_at = datetime.utcnow().isoformat()

        return task


# Create the inventory agent's Agent Card
INVENTORY_AGENT_CARD = AgentCard(
    name="inventory-agent",
    description="Manages warehouse inventory, stock levels, and reorder alerts",
    version="1.2.0",
    endpoint="https://agents.company.com/inventory",
    capabilities=[
        AgentCapability(
            name="check_stock",
            description="Check current stock levels for items",
            input_schema={
                "type": "object",
                "properties": {
                    "skus": {"type": "array", "items": {"type": "string"}}
                }
            }
        ),
        AgentCapability(
            name="reserve_inventory",
            description="Reserve items for an order",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        ),
        AgentCapability(
            name="reorder_alert",
            description="Get alerts for items below reorder point"
        )
    ],
    authentication=AgentAuthentication(
        type="oauth2",
        scopes=["inventory:read", "inventory:write"],
        token_url="https://auth.company.com/oauth/token"
    ),
    owner="supply-chain-team",
    supported_modes=["text", "structured_json"]
)


# =============================================================================
# Usage Example
# =============================================================================

async def example_usage():
    """Demonstrate A2A protocol usage"""

    # Create A2A client for the procurement agent
    client = A2AClient(
        agent_id="procurement-agent-001",
        agent_name="procurement-agent"
    )

    try:
        # 1. Discover the inventory agent
        print("Discovering inventory agent...")
        # In production, discover from actual endpoint
        # card = await client.discover_agent("https://agents.company.com/inventory")

        # For demo, use the predefined card
        client._known_agents["inventory-agent"] = INVENTORY_AGENT_CARD

        # 2. Create a task to check stock
        print("\nCreating stock check task...")
        task = await client.create_task(
            target_agent="inventory-agent",
            capability="check_stock",
            initial_message="Check stock levels for all widgets",
            metadata={"priority": "high", "requester": "po-workflow"}
        )
        print(f"Task created: {task.id}")
        print(f"State: {task.state}")

        # 3. Poll for completion (for async tasks)
        # In practice, you might use webhooks or SSE
        # task = await client.get_task_status(task.id)

        print("\nTask completed!")
        print(f"Result: {task.result}")

    finally:
        await client.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
