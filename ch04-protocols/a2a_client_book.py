"""
Chapter 4: Agent Communication Protocols
A2A (Agent-to-Agent) Implementation
====================================

Complete implementation of the A2A protocol for agent discovery and communication.

Usage:
    python a2a_client.py
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
from enum import Enum
import aiohttp

# Default timeout for HTTP operations
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30.0)

# Simple retry helper for HTTP operations
async def _retry_http(coro_func, max_attempts=3, base_delay=1.0):
    """Retry HTTP operation with exponential backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
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
    """Client for discovering and communicating with A2A agents."""

    def __init__(self):
        self.known_agents: dict[str, AgentCard] = {}
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30.0)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def discover_agent(self, url: str) -> AgentCard:
        """Fetch an agent's card from its well-known URL."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with A2AClient()' context manager.")
        card_url = f"{url.rstrip('/')}/.well-known/agent.json"

        async with self.session.get(card_url) as response:
            if response.status == 200:
                data = await response.json()
                card = AgentCard.from_dict(data)
                self.known_agents[url] = card
                return card
            else:
                raise A2AError(
                    f"Failed to discover agent at {url}: HTTP {response.status}")

    async def send_task(self, agent_url: str, skill: str, input_data: dict,
                        metadata: dict = None) -> Task:
        """Send a task to an agent."""
        task = Task(
            id=str(uuid.uuid4()),
            skill=skill,
            input=input_data,
            metadata=metadata or {}
        )

        endpoint = f"{agent_url.rstrip('/')}/tasks"

        async with self.session.post(endpoint, json=task.to_dict()) as response:
            if response.status in (200, 201, 202):
                data = await response.json()
                task.state = TaskState(data.get("state", "pending"))
                return task
            else:
                error = await response.text()
                raise A2AError(f"Failed to send task: {error}")

    async def get_task_status(self, agent_url: str, task_id: str) -> Task:
        """Get the status of a task."""
        endpoint = f"{agent_url.rstrip('/')}/tasks/{task_id}"

        async with self.session.get(endpoint) as response:
            if response.status == 200:
                data = await response.json()
                return Task(
                    id=data["id"],
                    skill=data["skill"],
                    input=data["input"],
                    state=TaskState(data["state"]),
                    metadata=data.get("metadata", {})
                )
            else:
                raise A2AError(f"Task not found: {task_id}")

    async def get_task_artifacts(self, agent_url: str,
                                  task_id: str) -> list[Artifact]:
        """Get artifacts produced by a task."""
        endpoint = f"{agent_url.rstrip('/')}/tasks/{task_id}/artifacts"

        async with self.session.get(endpoint) as response:
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

    async def wait_for_completion(self, agent_url: str, task_id: str,
                                   timeout: float = 300,
                                   poll_interval: float = 1) -> Task:
        """Wait for a task to complete."""
        start = asyncio.get_running_loop().time()

        while True:
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

    def __init__(self, card: AgentCard, task_handler):
        self.card = card
        self.task_handler = task_handler
        self.tasks: dict[str, Task] = {}
        self.artifacts: dict[str, list[Artifact]] = {}

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

        self.tasks[task.id] = task
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
