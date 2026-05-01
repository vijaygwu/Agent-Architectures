"""
Chapter 7: Swarm Pattern Implementation
======================================

Implements emergent coordination through pheromone-based communication,
stigmergic coordination, and self-organizing agent behaviors.

Based on OpenAI Swarm concepts and ant colony optimization principles.
"""

import asyncio
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import cycle
from typing import Any, Callable, Optional
from collections import defaultdict
import json
import hashlib

# Assume embedding utilities available (e.g., from openai or sentence_transformers)
# def get_embedding(text: str) -> list[float]: ...
# def cosine_similarity(a: list[float], b: list[float]) -> float: ...

# Placeholder implementations for embedding utilities
def get_embedding(text: str) -> list[float]:
    """Placeholder for embedding function."""
    # Simple hash-based pseudo-embedding for demonstration
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PheromoneType(Enum):
    """Types of pheromones for indirect agent communication."""
    VALUE = "value"           # Something useful here
    EXPLORED = "explored"     # Already checked this
    DANGER = "danger"         # Problem or dead end
    QUESTION = "question"     # Needs investigation
    SYNTHESIS = "synthesis"   # Multiple sources combined here
    RESOURCE = "resource"     # Resource availability signal


@dataclass
class Pheromone:
    """A pheromone marker left by an agent."""
    type: PheromoneType
    location: str             # Where in the problem space
    intensity: float          # 0.0 to 1.0
    data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

    def decay(self, rate: float = 0.1) -> "Pheromone":
        """Pheromones weaken over time."""
        age_seconds = (datetime.utcnow() - self.created_at).total_seconds()
        decay_factor = max(0, 1 - (rate * age_seconds / 3600))
        return Pheromone(
            type=self.type,
            location=self.location,
            intensity=self.intensity * decay_factor,
            data=self.data,
            created_at=self.created_at,
            created_by=self.created_by
        )

    def reinforce(self, amount: float = 0.2):
        """Reinforce pheromone when path is used again."""
        self.intensity = min(1.0, self.intensity + amount)
        self.created_at = datetime.utcnow()


class PheromoneField:
    """
    Shared environment for pheromone-based communication.
    Implements stigmergic coordination - agents communicate
    indirectly through modifications to the environment.
    """

    def __init__(self):
        self.pheromones: list[Pheromone] = []

    def deposit(self, pheromone: Pheromone):
        """Add a pheromone to the field."""
        self.pheromones.append(pheromone)

    def sense(self, location: str, radius: float = 0.5) -> list[Pheromone]:
        """Detect pheromones near a location."""
        nearby = []
        for p in self.pheromones:
            if self.distance(location, p.location) <= radius:
                decayed = p.decay()
                if decayed.intensity > 0.01:  # Threshold
                    nearby.append(decayed)
        return nearby

    def sense_type(self, ptype: PheromoneType, top_k: int = 10) -> list[Pheromone]:
        """Find strongest pheromones of a type."""
        matching = [p.decay() for p in self.pheromones if p.type == ptype]
        matching.sort(key=lambda p: p.intensity, reverse=True)
        return matching[:top_k]

    def distance(self, loc1: str, loc2: str) -> float:
        """Semantic distance between locations (embedding-based)."""
        emb1 = get_embedding(loc1)
        emb2 = get_embedding(loc2)
        return 1 - cosine_similarity(emb1, emb2)

    def prune(self, threshold: float = 0.01):
        """Remove pheromones below threshold intensity."""
        self.pheromones = [p for p in self.pheromones if p.decay().intensity > threshold]

    def all(self) -> list[Pheromone]:
        """Return all pheromones."""
        return self.pheromones


# Alias for backward compatibility
PheromoneTrail = PheromoneField


class DecayStrategies:
    """Different strategies for pheromone decay."""

    @staticmethod
    def linear_decay(initial: float, rate: float, age_seconds: float) -> float:
        """Simple linear decay."""
        return max(0, initial - (rate * age_seconds))

    @staticmethod
    def exponential_decay(initial: float, half_life: float,
                           age_seconds: float) -> float:
        """Exponential decay with configurable half-life."""
        return initial * math.exp(-math.log(2) * age_seconds / half_life)

    @staticmethod
    def threshold_decay(initial: float, threshold_seconds: float,
                         age_seconds: float) -> float:
        """No decay until threshold, then linear decay to zero."""
        if age_seconds < threshold_seconds:
            return initial
        else:
            overage = age_seconds - threshold_seconds
            return max(0, initial * (1 - overage / threshold_seconds))


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
            danger = [p for p in self.pheromone_trail.sense(location)
                     if p.type == PheromoneType.DANGER]
            if danger and danger[0].intensity > 0.7:
                continue  # Avoid this task type temporarily

            # Check success pheromones (exploitation)
            success = [p for p in self.pheromone_trail.sense(location)
                      if p.type == PheromoneType.VALUE]

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
        self.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.EXPLORED,
            location=location,
            intensity=0.5,
            created_by=self.id
        ))

        try:
            # Execute the task
            result = await self._run_with_timeout(task)

            # Success: deposit success pheromone
            self.pheromone_trail.deposit(Pheromone(
                type=PheromoneType.VALUE,
                location=location,
                intensity=0.8,
                data={"result_summary": str(result)[:100]},
                created_by=self.id
            ))

            await self.task_pool.complete_task(task.id, result, success=True)
            self.completed_count += 1

            # Adjust exploration rate based on success
            self._exploration_rate = max(0.1, self._exploration_rate - 0.02)

        except Exception as e:
            # Failure: deposit failure/danger pheromone
            self.pheromone_trail.deposit(Pheromone(
                type=PheromoneType.DANGER,
                location=location,
                intensity=0.6,
                data={"error": str(e)},
                created_by=self.id
            ))

            if task.attempts >= task.max_attempts - 1:
                # Repeated failures: mark as danger
                self.pheromone_trail.deposit(Pheromone(
                    type=PheromoneType.DANGER,
                    location=location,
                    intensity=0.9,
                    data={"reason": "repeated_failures"},
                    created_by=self.id
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
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.task_handler, task)

    async def request_assistance(self, task: SwarmTask, reason: str):
        """Broadcast assistance request via pheromones."""
        self.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.QUESTION,
            location=task.to_location(),
            intensity=0.9,
            data={
                "requesting_agent": self.id,
                "reason": reason,
                "task_payload": task.payload
            },
            created_by=self.id
        ))

    async def get_task_context(self, task: SwarmTask) -> dict:
        """Get context about a task for handoff."""
        return {
            "task_id": task.id,
            "task_type": task.type,
            "payload": task.payload,
            "attempts": task.attempts
        }


class Swarm:
    """
    The swarm coordinator.
    Manages agent lifecycle and provides global observation.
    """

    def __init__(self,
                 decay_rate: float = 0.05,
                 enable_queen: bool = True):
        self.task_pool = TaskPool()
        self.pheromone_trail = PheromoneField()
        self.decay_rate = decay_rate
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
                self.pheromone_trail.deposit(Pheromone(
                    type=PheromoneType.VALUE,
                    location="swarm/capacity",
                    intensity=0.9,
                    data={"pending_count": pending}
                ))

            # Check for assistance requests
            assistance_pheromones = self.pheromone_trail.sense_type(
                PheromoneType.QUESTION
            )

            for p in assistance_pheromones[:3]:
                if p.intensity > 0.7:
                    # Boost priority of tasks needing assistance
                    task_id = p.location.split("/")[-1]
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
                ptype.value: len(self.pheromone_trail.sense_type(ptype))
                for ptype in PheromoneType
            }
        }


# =============================================================================
# Specialized Swarm Patterns
# =============================================================================

class HeterogeneousSwarm(Swarm):
    """Swarm with different agent types working together."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_types = {}

    def add_agent_type(self, type_name: str,
                        capability: AgentCapability, count: int):
        """Add a type of agent to the swarm."""
        self.agent_types[type_name] = capability

        for i in range(count):
            agent_id = f"{type_name}_{i}"
            self.add_agent(agent_id, capability, self._create_handler(capability))

    def _create_handler(self, capability: AgentCapability) -> Callable:
        """Create a default handler for the capability."""
        async def handler(task: SwarmTask):
            await asyncio.sleep(random.uniform(0.5, 2.0))
            return {"task_id": task.id, "type": task.type}
        return handler

    def rebalance(self, type_counts: dict[str, int]):
        """Adjust the number of each agent type."""
        for type_name, target_count in type_counts.items():
            current = sum(1 for a in self.agents.values()
                         if a.id.startswith(type_name))

            if current < target_count:
                # Add agents
                for i in range(target_count - current):
                    self.add_agent(
                        f"{type_name}_{current + i}",
                        self.agent_types[type_name],
                        self._create_handler(self.agent_types[type_name])
                    )
            elif current > target_count:
                # Remove agents (let them terminate naturally)
                to_remove = [a for a in self.agents.values()
                            if a.id.startswith(type_name)][:current - target_count]
                for agent in to_remove:
                    asyncio.create_task(agent.stop())


class HierarchicalSwarm(Swarm):
    """Swarm with scouts, workers, and supervisors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scouts = []    # Explore new areas
        self.workers = []   # Exploit known areas
        self.supervisors = []  # Coordinate and report

    async def _queen_loop(self):
        """Supervisor coordination (not command)."""
        while self._running:
            await asyncio.sleep(10)

            # Analyze pheromone patterns
            patterns = await self._analyze_patterns()

            # Adjust swarm composition
            if patterns["unexplored_ratio"] > 0.5:
                # Lots unexplored - need more scouts
                await self._promote_to_scout()
            elif patterns["high_value_unexploited"]:
                # Good areas not being worked - need more workers
                await self._promote_to_worker()

            # Generate summary for observability
            await self._emit_summary(patterns)

    async def _analyze_patterns(self) -> dict:
        """Analyze pheromone patterns for decision making."""
        value_pheromones = self.pheromone_trail.sense_type(PheromoneType.VALUE)
        explored_pheromones = self.pheromone_trail.sense_type(PheromoneType.EXPLORED)

        total = len(value_pheromones) + len(explored_pheromones)
        unexplored_ratio = 1.0 - (len(explored_pheromones) / max(1, total))

        high_value = [p for p in value_pheromones if p.intensity > 0.7]
        high_value_unexploited = len(high_value) > len(self.workers)

        return {
            "unexplored_ratio": unexplored_ratio,
            "high_value_unexploited": high_value_unexploited,
            "total_pheromones": total
        }

    async def _promote_to_scout(self):
        """Promote a worker to scout role."""
        if self.workers:
            agent = self.workers.pop()
            self.scouts.append(agent)

    async def _promote_to_worker(self):
        """Promote a scout to worker role."""
        if self.scouts:
            agent = self.scouts.pop()
            self.workers.append(agent)

    async def _emit_summary(self, patterns: dict):
        """Emit summary for observability."""
        pass  # Implement logging/metrics as needed


class HandoffSwarm(Swarm):
    """Swarm where agents hand off tasks to specialists."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handoff_rules = {}  # from_type -> [to_types]

    def register_handoff_rule(self, from_type: str, to_types: list[str]):
        """Define which agent types can hand off to which others."""
        self.handoff_rules[from_type] = to_types

    async def request_handoff(
        self,
        from_agent: SwarmAgent,
        task: SwarmTask,
        reason: str
    ) -> bool:
        """Request handoff to a more suitable agent."""

        # Determine best recipient
        agent_type = from_agent.id.split("_")[0]
        possible_recipients = self.handoff_rules.get(agent_type, [])

        if not possible_recipients:
            return False

        # Find available agent of preferred type
        for recipient_type in possible_recipients:
            for agent in self.agents.values():
                if (agent.id.startswith(recipient_type) and
                    len(agent.current_tasks) < agent.capability.max_concurrent):

                    # Execute handoff
                    task.payload["handoff_context"] = {
                        "from_agent": from_agent.id,
                        "reason": reason,
                        "prior_work": await from_agent.get_task_context(task)
                    }

                    # Leave pheromone trail
                    self.pheromone_trail.deposit(Pheromone(
                        type=PheromoneType.RESOURCE,
                        location=f"handoff/{agent.id}",
                        intensity=0.8,
                        data={"task_id": task.id}
                    ))

                    # Release and reclaim
                    await self.task_pool.release_task(task.id)
                    return True

        return False


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
# Pheromone Strategies (from Chapter 7)
# =============================================================================

class PheromoneStrategies:
    """Pheromone-based decision strategies for swarm agents."""

    @staticmethod
    def choose_direction(pheromones: list[Pheromone]) -> str:
        """Choose direction based on pheromone attraction."""
        value_pheromones = [p for p in pheromones if p.type == PheromoneType.VALUE]

        if not value_pheromones:
            return ""  # Caller should use random_direction()

        # Probabilistic selection weighted by intensity
        total = sum(p.intensity for p in value_pheromones)
        r = random.random() * total

        cumulative = 0
        for p in value_pheromones:
            cumulative += p.intensity
            if cumulative >= r:
                return p.location

        return value_pheromones[-1].location

    @staticmethod
    def filter_options(options: list[str],
                       pheromones: list[Pheromone]) -> list[str]:
        """Remove options with strong negative signals."""
        danger_locations = {
            p.location for p in pheromones
            if p.type == PheromoneType.DANGER and p.intensity > 0.5
        }

        return [o for o in options if o not in danger_locations]

    @staticmethod
    def reinforce_path(environment, agent_id: str, path: list[str], success_level: float):
        """Strengthen pheromones along a successful path."""
        for location in path:
            nearby = environment.pheromones.sense(location, radius=0.1)
            existing = next((p for p in nearby if p.type == PheromoneType.VALUE), None)
            if existing:
                # Increase intensity
                existing.intensity = min(1.0, existing.intensity + success_level * 0.2)
            else:
                # Create new pheromone
                environment.pheromones.deposit(Pheromone(
                    type=PheromoneType.VALUE,
                    location=location,
                    intensity=success_level * 0.5,
                    created_by=agent_id
                ))


async def evaporation_loop(swarm: Swarm, rate: float = 0.1):
    """Background task that decays all pheromones."""
    while swarm._running:
        for pheromone in swarm.pheromone_trail.all():
            pheromone.intensity *= (1 - rate)

        # Remove faded pheromones
        swarm.pheromone_trail.prune(threshold=0.01)

        await asyncio.sleep(10)  # Every 10 seconds


# =============================================================================
# Exploration Strategy (from Chapter 7)
# =============================================================================

class ExplorationStrategy:
    """Manages the exploration/exploitation tradeoff."""

    def __init__(
        self,
        base_exploration_rate: float = 0.2,
        success_adjustment: float = 0.02,
        failure_adjustment: float = 0.05
    ):
        self.base_rate = base_exploration_rate
        self.success_adjustment = success_adjustment
        self.failure_adjustment = failure_adjustment
        self.current_rate = base_exploration_rate

    def should_explore(self) -> bool:
        """Decide whether to explore or exploit."""
        return random.random() < self.current_rate

    def record_success(self):
        """Decrease exploration after success (exploitation is working)."""
        self.current_rate = max(0.05, self.current_rate - self.success_adjustment)

    def record_failure(self):
        """Increase exploration after failure (need new approaches)."""
        self.current_rate = min(0.5, self.current_rate + self.failure_adjustment)

    def choose_action(self, pheromone_signals: list[Pheromone],
                       options: list[str]) -> str:
        """Choose an action based on exploration/exploitation."""

        if self.should_explore():
            # Explore: choose randomly, potentially ignoring pheromones
            return random.choice(options)
        else:
            # Exploit: follow strongest positive signal
            if pheromone_signals:
                best = max(pheromone_signals, key=lambda p: p.intensity)
                return best.location
            else:
                return random.choice(options)


# =============================================================================
# Observability and Control (from Chapter 7)
# =============================================================================

@dataclass
class AgentState:
    """State of a single agent."""
    current_location: str
    energy: float
    current_task: Optional[str]
    exploration_rate: float


@dataclass
class SwarmSnapshot:
    """Snapshot of swarm state at a point in time."""
    timestamp: float
    agent_states: dict[str, AgentState]
    pheromone_field: dict
    task_distribution: dict
    convergence_metrics: dict


class SwarmObserver:
    """Observes and reports on swarm state."""

    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    async def get_state_snapshot(self) -> SwarmSnapshot:
        """Capture current swarm state."""
        return SwarmSnapshot(
            timestamp=time.time(),
            agent_states={
                agent_id: AgentState(
                    current_location=getattr(agent, 'current_location', ''),
                    energy=getattr(agent, 'energy', 1.0),
                    current_task=(agent.current_tasks[0].id
                                  if agent.current_tasks else None),
                    exploration_rate=agent._exploration_rate
                )
                for agent_id, agent in self.swarm.agents.items()
            },
            pheromone_field=await self._snapshot_pheromones(),
            task_distribution=await self._snapshot_tasks(),
            convergence_metrics=await self._calculate_convergence()
        )

    async def _snapshot_pheromones(self) -> dict:
        """Snapshot pheromone field state."""
        return {
            ptype.value: [
                {"location": p.location, "intensity": p.intensity}
                for p in self.swarm.pheromone_trail.sense_type(ptype)
            ]
            for ptype in PheromoneType
        }

    async def _snapshot_tasks(self) -> dict:
        """Snapshot task distribution."""
        return await self.swarm.task_pool.get_stats()

    async def _calculate_convergence(self) -> dict:
        """Measure how converged the swarm is."""
        value_pheromones = self.swarm.pheromone_trail.sense_type(PheromoneType.VALUE)

        if not value_pheromones:
            return {"converged": False, "concentration": 0.0}

        # Calculate concentration (Gini coefficient of intensities)
        intensities = [p.intensity for p in value_pheromones]
        total = sum(intensities)

        if total == 0:
            return {"converged": False, "concentration": 0.0}

        normalized = [i/total for i in sorted(intensities)]
        n = len(normalized)
        cumulative = sum((i+1) * v for i, v in enumerate(normalized))
        gini = (2 * cumulative) / (n * sum(normalized)) - (n + 1) / n

        return {
            "converged": gini > 0.7,
            "concentration": gini,
            "top_location": value_pheromones[0].location if value_pheromones else None,
            "top_intensity": value_pheromones[0].intensity if value_pheromones else 0
        }


class SwarmController:
    """Provides control mechanisms for swarm behavior."""

    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    async def boost_area(self, location: str, intensity: float = 0.9):
        """Artificially boost interest in an area."""
        self.swarm.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.VALUE,
            location=location,
            intensity=intensity,
            data={"source": "controller_boost"}
        ))

    async def block_area(self, location: str):
        """Prevent agents from exploring an area."""
        self.swarm.pheromone_trail.deposit(Pheromone(
            type=PheromoneType.DANGER,
            location=location,
            intensity=1.0,
            data={"source": "controller_block"}
        ))

    async def inject_task(self, task: SwarmTask, priority_boost: float = 0.3):
        """Inject a high-priority task."""
        task.priority = min(1.0, task.priority + priority_boost)
        await self.swarm.submit_task(task)

        # Attract agents to this task
        await self.boost_area(task.to_location(), 0.8)

    async def reset_exploration(self):
        """Reset all agents to exploration mode."""
        for agent in self.swarm.agents.values():
            agent._exploration_rate = 0.5

    async def force_convergence(self, target_location: str):
        """Force swarm to converge on a location."""
        # Block all other high-value areas
        value_pheromones = self.swarm.pheromone_trail.sense_type(PheromoneType.VALUE)

        for p in value_pheromones:
            if p.location != target_location:
                await self.block_area(p.location)

        # Boost target
        await self.boost_area(target_location, 1.0)


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
