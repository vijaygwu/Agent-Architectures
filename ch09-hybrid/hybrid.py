"""
Chapter 9: Hybrid Architecture Implementation
=============================================

Combines multiple coordination patterns (Orchestrator, Council, Swarm, Guardian)
into a unified architecture that adapts to different task requirements.

Demonstrates pattern composition and dynamic routing.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import time
import json


class CoordinationPattern(Enum):
    """Available coordination patterns."""
    ORCHESTRATOR = "orchestrator"   # Centralized control, sequential
    COUNCIL = "council"             # Deliberative, consensus-based
    SWARM = "swarm"                 # Emergent, decentralized
    GUARDIAN = "guardian"           # Safety-first, monitored
    PIPELINE = "pipeline"           # Linear processing chain
    HYBRID = "hybrid"               # Dynamic pattern selection


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    SIMPLE = "simple"       # Single agent, no coordination needed
    MODERATE = "moderate"   # Requires some coordination
    COMPLEX = "complex"     # Requires multi-agent deliberation
    CRITICAL = "critical"   # Requires safety oversight


@dataclass
class HybridTask:
    """A task with metadata for hybrid routing."""
    id: str
    type: str
    payload: dict
    complexity: TaskComplexity = TaskComplexity.MODERATE
    requires_consensus: bool = False
    safety_sensitive: bool = False
    time_sensitive: bool = False
    preferred_pattern: Optional[CoordinationPattern] = None
    result: Any = None
    pattern_used: Optional[CoordinationPattern] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PatternMetrics:
    """Metrics for pattern performance tracking."""
    pattern: CoordinationPattern
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    avg_quality_score: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / (self.success_count + self.failure_count) \
            if (self.success_count + self.failure_count) > 0 else 0.0


class PatternInterface(ABC):
    """Interface that all coordination patterns must implement."""

    @abstractmethod
    async def execute(self, task: HybridTask) -> Any:
        """Execute a task using this pattern."""
        pass

    @abstractmethod
    async def can_handle(self, task: HybridTask) -> bool:
        """Check if this pattern can handle the given task."""
        pass

    @abstractmethod
    def get_pattern_type(self) -> CoordinationPattern:
        """Return the pattern type."""
        pass


class OrchestratorPattern(PatternInterface):
    """
    Orchestrator pattern: centralized coordination with a controller
    that delegates to worker agents.
    """

    def __init__(self, workers: dict[str, Callable]):
        self.workers = workers
        self.controller_state = {}

    async def execute(self, task: HybridTask) -> Any:
        # Plan execution
        plan = await self._create_plan(task)

        results = []
        for step in plan:
            worker_id = step["worker"]
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                step_result = await self._execute_worker(worker, step, task)
                results.append(step_result)

        # Aggregate results
        return await self._aggregate(results, task)

    async def _create_plan(self, task: HybridTask) -> list[dict]:
        """Create execution plan based on task type."""
        task_type = task.type

        default_plan = [
            {"worker": "analyzer", "action": "analyze", "input": task.payload},
            {"worker": "processor", "action": "process", "depends_on": [0]},
            {"worker": "validator", "action": "validate", "depends_on": [1]}
        ]

        return task.metadata.get("plan", default_plan)

    async def _execute_worker(self, worker: Callable, step: dict, task: HybridTask) -> Any:
        if asyncio.iscoroutinefunction(worker):
            return await worker(step, task)
        return worker(step, task)

    async def _aggregate(self, results: list, task: HybridTask) -> dict:
        return {
            "pattern": "orchestrator",
            "steps_completed": len(results),
            "results": results,
            "task_id": task.id
        }

    async def can_handle(self, task: HybridTask) -> bool:
        return task.complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]

    def get_pattern_type(self) -> CoordinationPattern:
        return CoordinationPattern.ORCHESTRATOR


class CouncilPattern(PatternInterface):
    """
    Council pattern: multiple agents deliberate and reach consensus
    through voting or structured debate.
    """

    def __init__(self, council_members: list[Callable], quorum: int = 3):
        self.council_members = council_members
        self.quorum = quorum
        self.voting_threshold = 0.6  # 60% agreement required

    async def execute(self, task: HybridTask) -> Any:
        # Phase 1: Generate proposals
        proposals = await self._gather_proposals(task)

        # Phase 2: Critique and refine
        critiques = await self._gather_critiques(proposals, task)

        # Phase 3: Vote
        votes = await self._vote(proposals, critiques, task)

        # Phase 4: Reach consensus
        consensus = await self._reach_consensus(proposals, votes, task)

        return {
            "pattern": "council",
            "proposals_count": len(proposals),
            "consensus_reached": consensus.get("reached", False),
            "final_decision": consensus.get("decision"),
            "vote_summary": votes,
            "task_id": task.id
        }

    async def _gather_proposals(self, task: HybridTask) -> list[dict]:
        proposals = []
        for i, member in enumerate(self.council_members):
            try:
                if asyncio.iscoroutinefunction(member):
                    proposal = await member({"action": "propose", "task": task.payload})
                else:
                    proposal = member({"action": "propose", "task": task.payload})
                proposals.append({"member": i, "proposal": proposal})
            except Exception as e:
                proposals.append({"member": i, "error": str(e)})
        return proposals

    async def _gather_critiques(self, proposals: list, task: HybridTask) -> list[dict]:
        critiques = []
        for proposal in proposals:
            if "error" in proposal:
                continue
            for i, member in enumerate(self.council_members):
                if i != proposal["member"]:
                    try:
                        if asyncio.iscoroutinefunction(member):
                            critique = await member({
                                "action": "critique",
                                "proposal": proposal["proposal"],
                                "task": task.payload
                            })
                        else:
                            critique = member({
                                "action": "critique",
                                "proposal": proposal["proposal"],
                                "task": task.payload
                            })
                        critiques.append({
                            "critic": i,
                            "target": proposal["member"],
                            "critique": critique
                        })
                    except Exception:
                        pass
        return critiques

    async def _vote(self, proposals: list, critiques: list, task: HybridTask) -> dict:
        votes = {i: 0 for i in range(len(proposals))}
        for i, member in enumerate(self.council_members):
            try:
                if asyncio.iscoroutinefunction(member):
                    vote = await member({
                        "action": "vote",
                        "proposals": proposals,
                        "critiques": critiques
                    })
                else:
                    vote = member({
                        "action": "vote",
                        "proposals": proposals,
                        "critiques": critiques
                    })
                if isinstance(vote, int) and vote < len(proposals):
                    votes[vote] += 1
            except Exception:
                pass
        return votes

    async def _reach_consensus(self, proposals: list, votes: dict,
                                task: HybridTask) -> dict:
        total_votes = sum(votes.values())
        if total_votes == 0:
            return {"reached": False, "decision": None}

        best_proposal_idx = max(votes.keys(), key=lambda k: votes[k])
        vote_share = votes[best_proposal_idx] / total_votes

        if vote_share >= self.voting_threshold:
            return {
                "reached": True,
                "decision": proposals[best_proposal_idx].get("proposal"),
                "vote_share": vote_share
            }
        return {"reached": False, "decision": None, "vote_share": vote_share}

    async def can_handle(self, task: HybridTask) -> bool:
        return task.requires_consensus or task.complexity == TaskComplexity.COMPLEX

    def get_pattern_type(self) -> CoordinationPattern:
        return CoordinationPattern.COUNCIL


class SwarmPattern(PatternInterface):
    """
    Swarm pattern: emergent coordination through indirect communication
    and simple local rules.
    """

    def __init__(self, agent_pool: list[Callable], pool_size: int = 5):
        self.agent_pool = agent_pool
        self.pool_size = pool_size
        self.pheromone_trails: dict[str, float] = {}

    async def execute(self, task: HybridTask) -> Any:
        # Split task into subtasks
        subtasks = await self._decompose(task)

        # Let agents self-organize
        results = await self._swarm_process(subtasks, task)

        # Aggregate emergent results
        return {
            "pattern": "swarm",
            "subtasks_processed": len(results),
            "results": results,
            "pheromone_state": dict(list(self.pheromone_trails.items())[:10]),
            "task_id": task.id
        }

    async def _decompose(self, task: HybridTask) -> list[dict]:
        payload = task.payload
        if isinstance(payload.get("items"), list):
            return [{"item": item, "index": i}
                   for i, item in enumerate(payload["items"])]
        return [{"payload": payload, "index": 0}]

    async def _swarm_process(self, subtasks: list, task: HybridTask) -> list:
        async def process_subtask(subtask: dict, agent_idx: int) -> dict:
            agent = self.agent_pool[agent_idx % len(self.agent_pool)]

            # Check pheromone trail
            trail_key = f"{task.type}:{subtask.get('index', 0)}"
            trail_strength = self.pheromone_trails.get(trail_key, 0.5)

            try:
                if asyncio.iscoroutinefunction(agent):
                    result = await agent(subtask, task)
                else:
                    result = agent(subtask, task)

                # Reinforce positive trail
                self.pheromone_trails[trail_key] = min(1.0, trail_strength + 0.1)
                return {"success": True, "result": result, "index": subtask.get("index")}

            except Exception as e:
                # Weaken trail on failure
                self.pheromone_trails[trail_key] = max(0.0, trail_strength - 0.2)
                return {"success": False, "error": str(e), "index": subtask.get("index")}

        tasks = [
            process_subtask(subtask, i)
            for i, subtask in enumerate(subtasks)
        ]
        return await asyncio.gather(*tasks)

    async def can_handle(self, task: HybridTask) -> bool:
        return not task.requires_consensus and not task.safety_sensitive

    def get_pattern_type(self) -> CoordinationPattern:
        return CoordinationPattern.SWARM


class GuardianPattern(PatternInterface):
    """
    Guardian pattern: wraps another pattern with safety monitoring
    and intervention capabilities.
    """

    def __init__(self,
                 inner_pattern: PatternInterface,
                 safety_checks: list[Callable],
                 intervention_threshold: float = 0.7):
        self.inner_pattern = inner_pattern
        self.safety_checks = safety_checks
        self.intervention_threshold = intervention_threshold
        self.intervention_log: list[dict] = []

    async def execute(self, task: HybridTask) -> Any:
        # Pre-execution safety check
        pre_check = await self._pre_check(task)
        if not pre_check["safe"]:
            return {
                "pattern": "guardian",
                "blocked": True,
                "reason": pre_check["reason"],
                "task_id": task.id
            }

        # Execute inner pattern with monitoring
        start_time = time.time()
        try:
            result = await self._monitored_execution(task)

            # Post-execution validation
            post_check = await self._post_check(result, task)
            if not post_check["valid"]:
                # Attempt remediation
                result = await self._remediate(result, post_check, task)

            return {
                "pattern": "guardian",
                "inner_pattern": self.inner_pattern.get_pattern_type().value,
                "result": result,
                "safety_checks_passed": True,
                "execution_time": time.time() - start_time,
                "interventions": len(self.intervention_log),
                "task_id": task.id
            }

        except Exception as e:
            self.intervention_log.append({
                "type": "exception",
                "task_id": task.id,
                "error": str(e),
                "timestamp": time.time()
            })
            return {
                "pattern": "guardian",
                "error": str(e),
                "blocked": True,
                "task_id": task.id
            }

    async def _pre_check(self, task: HybridTask) -> dict:
        for check in self.safety_checks:
            try:
                if asyncio.iscoroutinefunction(check):
                    result = await check({"phase": "pre", "task": task.payload})
                else:
                    result = check({"phase": "pre", "task": task.payload})

                if isinstance(result, dict) and not result.get("safe", True):
                    return result
                elif result is False:
                    return {"safe": False, "reason": "Safety check failed"}
            except Exception as e:
                return {"safe": False, "reason": f"Safety check error: {e}"}
        return {"safe": True}

    async def _monitored_execution(self, task: HybridTask) -> Any:
        return await self.inner_pattern.execute(task)

    async def _post_check(self, result: Any, task: HybridTask) -> dict:
        for check in self.safety_checks:
            try:
                if asyncio.iscoroutinefunction(check):
                    check_result = await check({
                        "phase": "post",
                        "result": result,
                        "task": task.payload
                    })
                else:
                    check_result = check({
                        "phase": "post",
                        "result": result,
                        "task": task.payload
                    })

                if isinstance(check_result, dict) and not check_result.get("valid", True):
                    return check_result
            except Exception as e:
                return {"valid": False, "reason": f"Validation error: {e}"}
        return {"valid": True}

    async def _remediate(self, result: Any, check_result: dict,
                         task: HybridTask) -> Any:
        self.intervention_log.append({
            "type": "remediation",
            "task_id": task.id,
            "reason": check_result.get("reason"),
            "timestamp": time.time()
        })
        # Simple remediation: return sanitized result
        if isinstance(result, dict):
            result["remediated"] = True
            result["remediation_reason"] = check_result.get("reason")
        return result

    async def can_handle(self, task: HybridTask) -> bool:
        return task.safety_sensitive

    def get_pattern_type(self) -> CoordinationPattern:
        return CoordinationPattern.GUARDIAN


class PatternRouter:
    """
    Routes tasks to the appropriate coordination pattern based on
    task characteristics and historical performance.
    """

    def __init__(self):
        self.patterns: dict[CoordinationPattern, PatternInterface] = {}
        self.metrics: dict[CoordinationPattern, PatternMetrics] = {}
        self.routing_rules: list[Callable[[HybridTask], Optional[CoordinationPattern]]] = []

    def register_pattern(self, pattern: PatternInterface):
        """Register a coordination pattern."""
        pattern_type = pattern.get_pattern_type()
        self.patterns[pattern_type] = pattern
        self.metrics[pattern_type] = PatternMetrics(pattern=pattern_type)

    def add_routing_rule(self, rule: Callable[[HybridTask], Optional[CoordinationPattern]]):
        """Add a custom routing rule."""
        self.routing_rules.append(rule)

    async def route(self, task: HybridTask) -> CoordinationPattern:
        """Determine the best pattern for a task."""
        # Check explicit preference
        if task.preferred_pattern and task.preferred_pattern in self.patterns:
            return task.preferred_pattern

        # Apply custom routing rules
        for rule in self.routing_rules:
            pattern = rule(task)
            if pattern and pattern in self.patterns:
                return pattern

        # Default routing logic
        if task.safety_sensitive:
            return CoordinationPattern.GUARDIAN
        elif task.requires_consensus:
            return CoordinationPattern.COUNCIL
        elif task.complexity == TaskComplexity.SIMPLE:
            return CoordinationPattern.SWARM
        elif task.complexity == TaskComplexity.COMPLEX:
            return CoordinationPattern.COUNCIL
        else:
            return CoordinationPattern.ORCHESTRATOR

    def update_metrics(self, pattern: CoordinationPattern,
                       success: bool, execution_time: float,
                       quality_score: float = 0.0):
        """Update pattern performance metrics."""
        if pattern in self.metrics:
            metrics = self.metrics[pattern]
            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1
            metrics.total_time += execution_time

            # Update rolling average quality score
            total = metrics.success_count + metrics.failure_count
            metrics.avg_quality_score = (
                (metrics.avg_quality_score * (total - 1) + quality_score) / total
            )


class HybridCoordinator:
    """
    Main coordinator that combines all patterns into a unified system.
    Implements the Hybrid Architecture from Chapter 9.
    """

    def __init__(self):
        self.router = PatternRouter()
        self.execution_history: list[dict] = []
        self.active_tasks: dict[str, HybridTask] = {}

    def register_pattern(self, pattern: PatternInterface):
        """Register a coordination pattern."""
        self.router.register_pattern(pattern)

    async def execute(self, task: HybridTask) -> dict:
        """Execute a task using the appropriate pattern."""
        start_time = time.time()
        self.active_tasks[task.id] = task

        try:
            # Route to appropriate pattern
            pattern_type = await self.router.route(task)
            pattern = self.router.patterns[pattern_type]

            # Execute
            result = await pattern.execute(task)

            # Update task
            execution_time = time.time() - start_time
            task.result = result
            task.pattern_used = pattern_type
            task.execution_time = execution_time

            # Update metrics
            success = not (isinstance(result, dict) and result.get("blocked"))
            self.router.update_metrics(pattern_type, success, execution_time)

            # Log execution
            self.execution_history.append({
                "task_id": task.id,
                "task_type": task.type,
                "pattern": pattern_type.value,
                "success": success,
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            return {
                "error": str(e),
                "task_id": task.id,
                "execution_time": time.time() - start_time
            }

        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]

    async def execute_batch(self, tasks: list[HybridTask]) -> list[dict]:
        """Execute multiple tasks, potentially in parallel."""
        # Group by pattern for efficient execution
        pattern_groups: dict[CoordinationPattern, list[HybridTask]] = {}

        for task in tasks:
            pattern_type = await self.router.route(task)
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(task)

        # Execute each group
        all_results = []
        for pattern_type, group_tasks in pattern_groups.items():
            results = await asyncio.gather(
                *[self.execute(task) for task in group_tasks]
            )
            all_results.extend(results)

        return all_results

    def get_metrics_summary(self) -> dict:
        """Get summary of pattern performance."""
        return {
            pattern.value: {
                "success_rate": metrics.success_rate,
                "avg_time": metrics.avg_time,
                "total_executions": metrics.success_count + metrics.failure_count,
                "avg_quality": metrics.avg_quality_score
            }
            for pattern, metrics in self.router.metrics.items()
        }


# =============================================================================
# Example Agents and Handlers
# =============================================================================

async def example_worker_agent(step: dict, task: HybridTask) -> dict:
    """Example worker for orchestrator pattern."""
    await asyncio.sleep(0.1)  # Simulate work
    return {
        "action": step.get("action"),
        "processed": True,
        "input_size": len(str(step.get("input", "")))
    }


async def example_council_member(request: dict) -> Any:
    """Example council member that can propose, critique, and vote."""
    import random
    action = request.get("action")

    if action == "propose":
        return {"approach": f"proposal_{random.randint(1, 100)}", "confidence": random.random()}
    elif action == "critique":
        return {"valid": random.random() > 0.3, "concerns": ["minor issue"]}
    elif action == "vote":
        proposals = request.get("proposals", [])
        return random.randint(0, max(0, len(proposals) - 1))
    return None


async def example_swarm_agent(subtask: dict, task: HybridTask) -> dict:
    """Example swarm agent."""
    await asyncio.sleep(0.05)  # Simulate work
    return {"processed": subtask.get("index"), "status": "complete"}


def example_safety_check(context: dict) -> dict:
    """Example safety check function."""
    phase = context.get("phase")
    if phase == "pre":
        task = context.get("task", {})
        # Block dangerous operations
        if task.get("action") == "delete_all":
            return {"safe": False, "reason": "Dangerous operation blocked"}
        return {"safe": True}
    elif phase == "post":
        result = context.get("result", {})
        # Validate output
        if isinstance(result, dict) and result.get("error"):
            return {"valid": False, "reason": "Result contains errors"}
        return {"valid": True}
    return {"safe": True, "valid": True}


async def main():
    """Demonstration of hybrid architecture."""
    print("=" * 60)
    print("Hybrid Architecture Demonstration")
    print("=" * 60)

    # Create coordinator
    coordinator = HybridCoordinator()

    # Create and register patterns
    orchestrator = OrchestratorPattern(workers={
        "analyzer": example_worker_agent,
        "processor": example_worker_agent,
        "validator": example_worker_agent
    })

    council = CouncilPattern(
        council_members=[example_council_member for _ in range(5)],
        quorum=3
    )

    swarm = SwarmPattern(
        agent_pool=[example_swarm_agent for _ in range(5)]
    )

    guardian = GuardianPattern(
        inner_pattern=orchestrator,
        safety_checks=[example_safety_check]
    )

    coordinator.register_pattern(orchestrator)
    coordinator.register_pattern(council)
    coordinator.register_pattern(swarm)
    coordinator.register_pattern(guardian)

    # Create diverse tasks
    tasks = [
        HybridTask(
            id="task-1",
            type="analysis",
            payload={"data": "sample data"},
            complexity=TaskComplexity.MODERATE
        ),
        HybridTask(
            id="task-2",
            type="decision",
            payload={"question": "strategic choice"},
            complexity=TaskComplexity.COMPLEX,
            requires_consensus=True
        ),
        HybridTask(
            id="task-3",
            type="batch_process",
            payload={"items": ["a", "b", "c", "d", "e"]},
            complexity=TaskComplexity.SIMPLE
        ),
        HybridTask(
            id="task-4",
            type="sensitive_operation",
            payload={"action": "update_config"},
            safety_sensitive=True
        ),
        HybridTask(
            id="task-5",
            type="dangerous",
            payload={"action": "delete_all"},
            safety_sensitive=True
        )
    ]

    print(f"\nExecuting {len(tasks)} tasks with different characteristics...")
    print()

    # Execute tasks
    for task in tasks:
        print(f"Task {task.id} ({task.type}):")
        result = await coordinator.execute(task)
        print(f"  Pattern used: {task.pattern_used.value if task.pattern_used else 'N/A'}")
        print(f"  Execution time: {task.execution_time:.3f}s")
        if isinstance(result, dict):
            if result.get("blocked"):
                print(f"  BLOCKED: {result.get('reason')}")
            else:
                print(f"  Success: True")
        print()

    # Show metrics
    print("=" * 60)
    print("Pattern Performance Metrics")
    print("=" * 60)
    metrics = coordinator.get_metrics_summary()
    for pattern, stats in metrics.items():
        if stats["total_executions"] > 0:
            print(f"\n{pattern}:")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Avg time: {stats['avg_time']:.3f}s")
            print(f"  Total executions: {stats['total_executions']}")


if __name__ == "__main__":
    asyncio.run(main())
