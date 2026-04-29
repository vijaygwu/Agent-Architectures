"""
Chapter 7: Swarm Pattern Implementation
======================================

Implements emergent coordination through pheromone-based communication,
stigmergic coordination, and self-organizing agent behaviors.

Based on OpenAI Swarm concepts and ant colony optimization principles.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict
import json
import hashlib


class PheromoneType(Enum):
    """Types of pheromones for indirect agent communication."""
    SUCCESS = "success"          # Task completed successfully
    FAILURE = "failure"          # Task failed, avoid this path
    EXPLORATION = "exploration"  # New territory being explored
    RESOURCE = "resource"        # Resource discovered
    DANGER = "danger"            # Hazard or constraint detected
    ASSISTANCE = "assistance"    # Help needed


@dataclass
class Pheromone:
    """A pheromone marker left by an agent."""
    type: PheromoneType
    location: str  # Task or resource identifier
    intensity: float  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""

    def decay(self, rate: float = 0.1) -> float:
        """Apply time-based decay to pheromone intensity."""
        age = time.time() - self.timestamp
        decay_factor = max(0, 1 - (rate * age))
        self.intensity *= decay_factor
        return self.intensity

    def reinforce(self, amount: float = 0.2):
        """Reinforce pheromone when path is used again."""
        self.intensity = min(1.0, self.intensity + amount)
        self.timestamp = time.time()


class PheromoneTrail:
    """
    Shared environment for pheromone-based communication.
    Implements stigmergic coordination - agents communicate
    indirectly through modifications to the environment.
    """

    def __init__(self, decay_rate: float = 0.05):
        self.trails: dict[str, list[Pheromone]] = defaultdict(list)
        self.decay_rate = decay_rate
        self._lock = asyncio.Lock()

    async def deposit(self, pheromone: Pheromone):
        """Deposit a pheromone at a location."""
        async with self._lock:
            existing = self._find_matching(pheromone)
            if existing:
                existing.reinforce(pheromone.intensity * 0.5)
                existing.metadata.update(pheromone.metadata)
            else:
                self.trails[pheromone.location].append(pheromone)

    def _find_matching(self, pheromone: Pheromone) -> Optional[Pheromone]:
        """Find existing pheromone of same type at location."""
        for p in self.trails[pheromone.location]:
            if p.type == pheromone.type:
                return p
        return None

    async def sense(self, location: str,
                    pheromone_type: Optional[PheromoneType] = None) -> list[Pheromone]:
        """Sense pheromones at a location."""
        async with self._lock:
            await self._apply_decay()
            pheromones = self.trails.get(location, [])
            if pheromone_type:
                return [p for p in pheromones if p.type == pheromone_type]
            return list(pheromones)

    async def sense_nearby(self, location: str, radius: int = 2) -> dict[str, list[Pheromone]]:
        """Sense pheromones in nearby locations (for exploration)."""
        nearby = {}
        for loc, pheromones in self.trails.items():
            if self._location_distance(location, loc) <= radius:
                nearby[loc] = pheromones
        return nearby

    def _location_distance(self, loc1: str, loc2: str) -> int:
        """Simple distance heuristic based on location similarity."""
        parts1 = loc1.split("/")
        parts2 = loc2.split("/")
        common = sum(1 for a, b in zip(parts1, parts2) if a == b)
        return max(len(parts1), len(parts2)) - common

    async def _apply_decay(self):
        """Apply decay to all pheromones and remove depleted ones."""
        for location in list(self.trails.keys()):
            self.trails[location] = [
                p for p in self.trails[location]
                if p.decay(self.decay_rate) > 0.01
            ]
            if not self.trails[location]:
                del self.trails[location]

    async def get_strongest_path(self,
                                  pheromone_type: PheromoneType) -> list[tuple[str, float]]:
        """Get locations sorted by pheromone intensity."""
        paths = []
        for location, pheromones in self.trails.items():
            for p in pheromones:
                if p.type == pheromone_type:
                    paths.append((location, p.intensity))
        return sorted(paths, key=lambda x: x[1], reverse=True)


@dataclass
class SwarmTask:
    """A task to be processed by the swarm."""
    id: str
    type: str
    payload: dict
    priority: float = 0.5
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Any = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)

    def to_location(self) -> str:
        """Convert task to a location identifier for pheromone trails."""
        return f"task/{self.type}/{self.id}"


class TaskPool:
    """
    Shared pool of tasks for swarm agents.
    Implements task stealing and load balancing.
    """

    def __init__(self):
        self.tasks: dict[str, SwarmTask] = {}
        self.completed: list[SwarmTask] = []
        self._lock = asyncio.Lock()

    async def add_task(self, task: SwarmTask):
        """Add a task to the pool."""
        async with self._lock:
            self.tasks[task.id] = task

    async def claim_task(self, agent_id: str,
                         task_types: Optional[list[str]] = None) -> Optional[SwarmTask]:
        """Attempt to claim an available task."""
        async with self._lock:
            for task in sorted(self.tasks.values(),
                             key=lambda t: t.priority, reverse=True):
                if task.status == "pending":
                    if task_types is None or task.type in task_types:
                        task.status = "claimed"
                        task.assigned_agent = agent_id
                        return task
            return None

    async def complete_task(self, task_id: str, result: Any, success: bool = True):
        """Mark a task as completed."""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.result = result
                task.status = "completed" if success else "failed"
                if success or task.attempts >= task.max_attempts:
                    self.completed.append(task)
                    del self.tasks[task_id]
                else:
                    task.status = "pending"
                    task.attempts += 1
                    task.assigned_agent = None

    async def release_task(self, task_id: str):
        """Release a claimed task back to the pool."""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = "pending"
                task.assigned_agent = None

    async def get_stats(self) -> dict:
        """Get pool statistics."""
        async with self._lock:
            return {
                "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "claimed": sum(1 for t in self.tasks.values() if t.status == "claimed"),
                "completed": len(self.completed),
                "total": len(self.tasks) + len(self.completed)
            }


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    task_types: list[str]
    efficiency: dict[str, float] = field(default_factory=dict)  # task_type -> efficiency
    max_concurrent: int = 1


class SwarmAgent:
    """
    An individual agent in the swarm.
    Follows simple local rules that produce emergent coordination.
    """

    def __init__(self,
                 agent_id: str,
                 capability: AgentCapability,
                 task_pool: TaskPool,
                 pheromone_trail: PheromoneTrail,
                 task_handler: Callable[[SwarmTask], Any]):
        self.id = agent_id
        self.capability = capability
        self.task_pool = task_pool
        self.pheromone_trail = pheromone_trail
        self.task_handler = task_handler
        self.current_tasks: list[SwarmTask] = []
        self.completed_count = 0
        self.failed_count = 0
        self._running = False
        self._exploration_rate = 0.2  # Probability of exploring vs exploiting

    async def start(self):
        """Start the agent's work loop."""
        self._running = True
        while self._running:
            await self._work_cycle()
            await asyncio.sleep(0.1)  # Small delay between cycles

    async def stop(self):
        """Stop the agent."""
        self._running = False

    async def _work_cycle(self):
        """Execute one work cycle: sense, decide, act."""
        # Check if can take more tasks
        if len(self.current_tasks) >= self.capability.max_concurrent:
            return

        # Sense environment
        task_decision = await self._sense_and_decide()

        if task_decision:
            # Try to claim the decided task
            task = await self.task_pool.claim_task(
                self.id,
                self.capability.task_types
            )
            if task:
                await self._execute_task(task)

    async def _sense_and_decide(self) -> bool:
        """
        Sense pheromones and decide whether to take a task.
        Implements exploration vs exploitation trade-off.
        """
        # Check for danger pheromones
        for task_type in self.capability.task_types:
            location = f"task/{task_type}"
            danger = await self.pheromone_trail.sense(
                location, PheromoneType.DANGER
            )
            if danger and danger[0].intensity > 0.7:
                continue  # Avoid this task type temporarily

            # Check success pheromones (exploitation)
            success = await self.pheromone_trail.sense(
                location, PheromoneType.SUCCESS
            )

            # Decide based on exploration vs exploitation
            if random.random() < self._exploration_rate:
                # Explore: try even without strong success signals
                return True
            elif success and success[0].intensity > 0.3:
                # Exploit: follow successful paths
                return True

        # Default: try if capable
        return True

    async def _execute_task(self, task: SwarmTask):
        """Execute a claimed task."""
        self.current_tasks.append(task)
        location = task.to_location()

        # Deposit exploration pheromone
        await self.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.EXPLORATION,
            location=location,
            intensity=0.5,
            agent_id=self.id
        ))

        try:
            # Execute the task
            result = await self._run_with_timeout(task)

            # Success: deposit success pheromone
            await self.pheromone_trail.deposit(Pheromone(
                type=PheromoneType.SUCCESS,
                location=location,
                intensity=0.8,
                metadata={"result_summary": str(result)[:100]},
                agent_id=self.id
            ))

            await self.task_pool.complete_task(task.id, result, success=True)
            self.completed_count += 1

            # Adjust exploration rate based on success
            self._exploration_rate = max(0.1, self._exploration_rate - 0.02)

        except Exception as e:
            # Failure: deposit failure/danger pheromone
            await self.pheromone_trail.deposit(Pheromone(
                type=PheromoneType.FAILURE,
                location=location,
                intensity=0.6,
                metadata={"error": str(e)},
                agent_id=self.id
            ))

            if task.attempts >= task.max_attempts - 1:
                # Repeated failures: mark as danger
                await self.pheromone_trail.deposit(Pheromone(
                    type=PheromoneType.DANGER,
                    location=location,
                    intensity=0.9,
                    metadata={"reason": "repeated_failures"},
                    agent_id=self.id
                ))

            await self.task_pool.complete_task(task.id, str(e), success=False)
            self.failed_count += 1

            # Increase exploration on failure
            self._exploration_rate = min(0.5, self._exploration_rate + 0.05)

        finally:
            self.current_tasks.remove(task)

    async def _run_with_timeout(self, task: SwarmTask, timeout: float = 30.0) -> Any:
        """Run task handler with timeout."""
        return await asyncio.wait_for(
            asyncio.create_task(self._async_handler(task)),
            timeout=timeout
        )

    async def _async_handler(self, task: SwarmTask) -> Any:
        """Wrap handler for async execution."""
        if asyncio.iscoroutinefunction(self.task_handler):
            return await self.task_handler(task)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.task_handler, task)

    async def request_assistance(self, task: SwarmTask, reason: str):
        """Broadcast assistance request via pheromones."""
        await self.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.ASSISTANCE,
            location=task.to_location(),
            intensity=0.9,
            metadata={
                "requesting_agent": self.id,
                "reason": reason,
                "task_payload": task.payload
            },
            agent_id=self.id
        ))


class Swarm:
    """
    The swarm coordinator.
    Manages agent lifecycle and provides global observation.
    """

    def __init__(self,
                 decay_rate: float = 0.05,
                 enable_queen: bool = True):
        self.task_pool = TaskPool()
        self.pheromone_trail = PheromoneTrail(decay_rate)
        self.agents: dict[str, SwarmAgent] = {}
        self.enable_queen = enable_queen
        self._running = False
        self._agent_tasks: list[asyncio.Task] = []

    def add_agent(self,
                  agent_id: str,
                  capability: AgentCapability,
                  task_handler: Callable[[SwarmTask], Any]) -> SwarmAgent:
        """Add an agent to the swarm."""
        agent = SwarmAgent(
            agent_id=agent_id,
            capability=capability,
            task_pool=self.task_pool,
            pheromone_trail=self.pheromone_trail,
            task_handler=task_handler
        )
        self.agents[agent_id] = agent
        return agent

    async def submit_task(self, task: SwarmTask):
        """Submit a task to the swarm."""
        await self.task_pool.add_task(task)

    async def submit_tasks(self, tasks: list[SwarmTask]):
        """Submit multiple tasks to the swarm."""
        for task in tasks:
            await self.task_pool.add_task(task)

    async def start(self):
        """Start all agents in the swarm."""
        self._running = True

        # Start all agents
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start())
            self._agent_tasks.append(task)

        # Start queen agent if enabled
        if self.enable_queen:
            queen_task = asyncio.create_task(self._queen_loop())
            self._agent_tasks.append(queen_task)

    async def stop(self):
        """Stop all agents in the swarm."""
        self._running = False
        for agent in self.agents.values():
            await agent.stop()

        for task in self._agent_tasks:
            task.cancel()

        await asyncio.gather(*self._agent_tasks, return_exceptions=True)

    async def _queen_loop(self):
        """
        Queen agent: monitors swarm health and adjusts parameters.
        Implements high-level coordination without direct control.
        """
        while self._running:
            await asyncio.sleep(5)  # Check every 5 seconds

            stats = await self.task_pool.get_stats()

            # Check for stalled tasks
            pending = stats["pending"]
            claimed = stats["claimed"]

            if pending > len(self.agents) * 2:
                # Too many pending tasks: signal resource need
                await self.pheromone_trail.deposit(Pheromone(
                    type=PheromoneType.RESOURCE,
                    location="swarm/capacity",
                    intensity=0.9,
                    metadata={"pending_count": pending}
                ))

            # Check for assistance requests
            assistance_trails = await self.pheromone_trail.get_strongest_path(
                PheromoneType.ASSISTANCE
            )

            for location, intensity in assistance_trails[:3]:
                if intensity > 0.7:
                    # Boost priority of tasks needing assistance
                    task_id = location.split("/")[-1]
                    if task_id in self.task_pool.tasks:
                        self.task_pool.tasks[task_id].priority = min(
                            1.0,
                            self.task_pool.tasks[task_id].priority + 0.2
                        )

    async def wait_for_completion(self, timeout: Optional[float] = None) -> dict:
        """Wait for all tasks to complete."""
        start_time = time.time()

        while True:
            stats = await self.task_pool.get_stats()
            if stats["pending"] == 0 and stats["claimed"] == 0:
                break

            if timeout and (time.time() - start_time) > timeout:
                break

            await asyncio.sleep(0.5)

        return {
            "stats": await self.task_pool.get_stats(),
            "completed_tasks": self.task_pool.completed,
            "agent_stats": {
                agent_id: {
                    "completed": agent.completed_count,
                    "failed": agent.failed_count
                }
                for agent_id, agent in self.agents.items()
            }
        }

    async def get_swarm_state(self) -> dict:
        """Get current swarm state for observability."""
        return {
            "agents": {
                agent_id: {
                    "current_tasks": len(agent.current_tasks),
                    "completed": agent.completed_count,
                    "failed": agent.failed_count,
                    "exploration_rate": agent._exploration_rate
                }
                for agent_id, agent in self.agents.items()
            },
            "task_pool": await self.task_pool.get_stats(),
            "pheromone_summary": {
                ptype.value: len(await self.pheromone_trail.get_strongest_path(ptype))
                for ptype in PheromoneType
            }
        }


# =============================================================================
# Specialized Swarm Patterns
# =============================================================================

class HandoffSwarm(Swarm):
    """
    Swarm with explicit handoff capabilities between agents.
    Based on OpenAI Swarm's handoff pattern.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handoff_registry: dict[str, list[str]] = {}  # agent_id -> can_handoff_to

    def register_handoff(self, from_agent: str, to_agents: list[str]):
        """Register valid handoff paths."""
        self.handoff_registry[from_agent] = to_agents

    async def handoff(self,
                      task: SwarmTask,
                      from_agent: str,
                      to_agent: str,
                      context: dict) -> bool:
        """
        Handoff a task from one agent to another.
        Returns True if handoff was successful.
        """
        # Validate handoff
        if from_agent not in self.handoff_registry:
            return False
        if to_agent not in self.handoff_registry[from_agent]:
            return False
        if to_agent not in self.agents:
            return False

        # Update task with context
        task.payload["handoff_context"] = context
        task.payload["handoff_from"] = from_agent
        task.assigned_agent = None
        task.status = "pending"

        # Deposit pheromone trail for handoff
        await self.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.RESOURCE,
            location=f"handoff/{to_agent}",
            intensity=0.9,
            metadata={"task_id": task.id, "from": from_agent}
        ))

        return True


class SpecializationSwarm(Swarm):
    """
    Swarm where agents dynamically specialize based on success.
    Implements emergent division of labor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specialization_scores: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # agent_id -> task_type -> score

    def add_agent(self, agent_id: str, capability: AgentCapability,
                  task_handler: Callable[[SwarmTask], Any]) -> SwarmAgent:
        """Add agent with specialization tracking."""
        agent = super().add_agent(agent_id, capability, task_handler)

        # Wrap the original task handler to track specialization
        original_handler = task_handler

        async def tracking_handler(task: SwarmTask):
            try:
                result = await self._run_handler(original_handler, task)
                # Increase specialization on success
                self.specialization_scores[agent_id][task.type] += 0.1
                return result
            except Exception as e:
                # Decrease specialization on failure
                self.specialization_scores[agent_id][task.type] -= 0.05
                raise e

        agent.task_handler = tracking_handler
        return agent

    async def _run_handler(self, handler, task):
        if asyncio.iscoroutinefunction(handler):
            return await handler(task)
        return handler(task)

    def get_best_agent_for_task(self, task_type: str) -> Optional[str]:
        """Get the agent most specialized for a task type."""
        best_agent = None
        best_score = -float('inf')

        for agent_id, scores in self.specialization_scores.items():
            if agent_id in self.agents:
                score = scores.get(task_type, 0)
                if score > best_score:
                    best_score = score
                    best_agent = agent_id

        return best_agent


# =============================================================================
# Example Usage and Task Handlers
# =============================================================================

async def example_research_handler(task: SwarmTask) -> dict:
    """Example handler for research tasks."""
    await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate work
    return {
        "query": task.payload.get("query", ""),
        "results": [f"Result {i} for {task.payload.get('query', '')}"
                   for i in range(3)]
    }


async def example_analysis_handler(task: SwarmTask) -> dict:
    """Example handler for analysis tasks."""
    await asyncio.sleep(random.uniform(1.0, 3.0))  # Simulate work
    return {
        "analysis": f"Analysis of {task.payload.get('data', 'unknown')}",
        "confidence": random.uniform(0.7, 0.99)
    }


async def example_writing_handler(task: SwarmTask) -> dict:
    """Example handler for writing tasks."""
    await asyncio.sleep(random.uniform(0.5, 1.5))  # Simulate work
    return {
        "content": f"Written content for topic: {task.payload.get('topic', 'unknown')}",
        "word_count": random.randint(100, 500)
    }


async def main():
    """Demonstration of swarm pattern."""
    print("=" * 60)
    print("Swarm Pattern Demonstration")
    print("=" * 60)

    # Create swarm
    swarm = SpecializationSwarm(decay_rate=0.1)

    # Add specialized agents
    swarm.add_agent(
        "researcher-1",
        AgentCapability(task_types=["research"], max_concurrent=2),
        example_research_handler
    )
    swarm.add_agent(
        "researcher-2",
        AgentCapability(task_types=["research"], max_concurrent=2),
        example_research_handler
    )
    swarm.add_agent(
        "analyst-1",
        AgentCapability(task_types=["analysis", "research"], max_concurrent=1),
        example_analysis_handler
    )
    swarm.add_agent(
        "writer-1",
        AgentCapability(task_types=["writing"], max_concurrent=3),
        example_writing_handler
    )

    # Submit tasks
    tasks = [
        SwarmTask(id=f"research-{i}", type="research",
                  payload={"query": f"Topic {i}"}, priority=0.5 + i*0.1)
        for i in range(5)
    ] + [
        SwarmTask(id=f"analysis-{i}", type="analysis",
                  payload={"data": f"Dataset {i}"}, priority=0.6)
        for i in range(3)
    ] + [
        SwarmTask(id=f"writing-{i}", type="writing",
                  payload={"topic": f"Subject {i}"}, priority=0.4)
        for i in range(4)
    ]

    await swarm.submit_tasks(tasks)

    print(f"\nSubmitted {len(tasks)} tasks to swarm with {len(swarm.agents)} agents")
    print("\nStarting swarm...")

    await swarm.start()

    # Wait for completion
    results = await swarm.wait_for_completion(timeout=30.0)

    await swarm.stop()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nTask Pool Stats: {results['stats']}")
    print(f"\nAgent Performance:")
    for agent_id, stats in results['agent_stats'].items():
        print(f"  {agent_id}: completed={stats['completed']}, failed={stats['failed']}")

    print(f"\nSpecialization Scores:")
    for agent_id, scores in swarm.specialization_scores.items():
        print(f"  {agent_id}: {dict(scores)}")

    print(f"\nCompleted {len(results['completed_tasks'])} tasks")


if __name__ == "__main__":
    asyncio.run(main())
