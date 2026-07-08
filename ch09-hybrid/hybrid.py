"""
Chapter 9: Hybrid Architecture Implementation
=============================================

Combines multiple coordination patterns (Orchestrator, Council, Swarm, Guardian)
into a unified architecture that adapts to different task requirements.

This module provides a complete hybrid architecture system that:
1. Analyzes incoming tasks
2. Routes to appropriate patterns or pattern combinations
3. Executes with proper error handling
4. Returns structured results with provenance

Demonstrates pattern composition and dynamic routing.
"""

__all__ = [
    "retry_with_backoff",
    "ValidationResult",
    "PatternType",
    "PhaseType",
    "TaskCharacteristics",
    "PatternScore",
    "RoutingDecision",
    "ExecutionResult",
    "HybridPhase",
    "OrchestratorResult",
    "Decision",
    "GuardedOrchestrator",
    "SwarmCouncilPipeline",
    "OrchestratorSwarmHybrid",
    "Constraint",
    "ConstrainedDecision",
    "CouncilDecision",
    "CouncilMessage",
    "Councilor",
    "ConstrainedCouncil",
    "PatternRouter",
    "RoutedResult",
    "PipelinePhase",
    "AdaptivePipeline",
    "PhaseResult",
    "PipelineResult",
    "TaskAnalyzer",
    "PatternScorer",
    "HybridRouter",
    "Phase",
    "ProductionPhaseResult",
    "ProductionPipelineResult",
    "ProductionAdaptivePipeline",
    "ValidationResponse",
    "ApprovalResponse",
    "QuestionCheckResponse",
    "DecisionValidationResponse",
    "SwarmResult",
    "MockOrchestrator",
    "MockGuardian",
    "MockSwarmCoordinator",
    "MockCouncil",
]

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import copy
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Circuit Breaker Import with Fallback
# =============================================================================

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
except ImportError:
    # Inline fallback implementation for standalone operation
    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        def __init__(self, breaker_name: str, time_until_retry: float = 0.0):
            self.breaker_name = breaker_name
            self.time_until_retry = time_until_retry
            super().__init__(
                f"Circuit breaker '{breaker_name}' is OPEN. "
                f"Retry in {time_until_retry:.1f}s"
            )

    @dataclass
    class CircuitBreaker:
        """Fallback circuit breaker for standalone operation."""
        name: str
        failure_threshold: int = 5
        recovery_timeout: float = 30.0

        _failure_count: int = field(default=0, init=False)
        _last_failure_time: float = field(default=0.0, init=False)
        _is_open: bool = field(default=False, init=False)

        def _check_recovery(self) -> None:
            """Check if circuit should recover from open state."""
            if self._is_open and self._last_failure_time > 0:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._is_open = False
                    self._failure_count = 0

        def allow(self) -> bool:
            """Check if requests are allowed through the circuit."""
            self._check_recovery()
            return not self._is_open

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


# =============================================================================
# Retry Utilities
# =============================================================================

async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """Execute an async function with exponential backoff retry.

    Args:
        func: The async function to execute.
        *args: Positional arguments to pass to the function.
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds for exponential backoff (default: 1.0).
        max_delay: Maximum delay between retries in seconds (default: 30.0).
        exceptions: Tuple of exception types to catch and retry on.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries fail.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")
    raise last_error


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ValidationResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class PatternType(Enum):
    ORCHESTRATOR = "orchestrator"
    COUNCIL = "council"
    SWARM = "swarm"
    GUARDIAN = "guardian"
    # Hybrids
    ORCHESTRATOR_GUARDIAN = "orchestrator_guardian"
    COUNCIL_SWARM = "council_swarm"
    ORCHESTRATOR_SWARM = "orchestrator_swarm"
    COUNCIL_GUARDIAN = "council_guardian"
    ADAPTIVE = "adaptive"  # Router decides per-phase


class PhaseType(Enum):
    STRUCTURED = "structured"  # Use standard workers
    EXPLORATORY = "exploratory"  # Use swarm


@dataclass
class TaskCharacteristics:
    """Analyzed characteristics of a task."""

    # Structure
    decomposable: float = 0.0      # Can be broken into subtasks (0-1)
    dependencies_clear: float = 0.0  # Dependencies between parts known (0-1)

    # Uncertainty
    ambiguity: float = 0.0         # Multiple valid interpretations (0-1)
    unknowns: float = 0.0          # Requires discovering information (0-1)

    # Stakes
    consequence_severity: float = 0.0  # Impact of wrong answer (0-1)
    reversibility: float = 0.0     # Can undo mistakes (0-1)

    # Requirements
    needs_deliberation: float = 0.0    # Benefits from multiple perspectives (0-1)
    needs_creativity: float = 0.0       # Requires novel solutions (0-1)
    needs_safety: float = 0.0           # Requires guardrails (0-1)
    needs_speed: float = 0.0            # Time-sensitive (0-1)

    def to_dict(self) -> dict:
        return {
            "decomposable": self.decomposable,
            "dependencies_clear": self.dependencies_clear,
            "ambiguity": self.ambiguity,
            "unknowns": self.unknowns,
            "consequence_severity": self.consequence_severity,
            "reversibility": self.reversibility,
            "needs_deliberation": self.needs_deliberation,
            "needs_creativity": self.needs_creativity,
            "needs_safety": self.needs_safety,
            "needs_speed": self.needs_speed
        }


@dataclass
class PatternScore:
    """Score for a pattern given task characteristics."""
    pattern: PatternType
    score: float
    rationale: str


@dataclass
class RoutingDecision:
    """Decision about which pattern to use."""
    pattern: PatternType
    confidence: float
    rationale: str
    characteristics: TaskCharacteristics
    alternatives: list[tuple[PatternType, float]] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of pattern execution."""
    success: bool
    output: Any = None
    error: str | None = None
    pattern_used: PatternType | None = None
    routing_decision: RoutingDecision | None = None
    execution_time_ms: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class HybridPhase:
    id: str
    description: str
    phase_type: PhaseType
    dependencies: list[str]
    config: dict = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class Decision:
    question: str
    chosen_option: Any
    alternatives: list[dict]
    deliberation_transcript: list
    exploration_artifacts: list


# =============================================================================
# Protocols
# =============================================================================

T = TypeVar('T')


class ExecutablePattern(Protocol):
    """Protocol for patterns that can be executed."""

    async def execute(self, task: str, context: dict | None = None) -> Any:
        """Execute the pattern on a task."""
        ...


class DeliberativePattern(Protocol):
    """Protocol for patterns that deliberate."""

    async def deliberate(
        self,
        question: str,
        context: dict | None = None
    ) -> Any:
        """Deliberate on a question."""
        ...


class ExploratoryPattern(Protocol):
    """Protocol for patterns that explore."""

    async def run(
        self,
        goal: str,
        max_steps: int = 100
    ) -> Any:
        """Run exploration toward a goal."""
        ...


# =============================================================================
# Guarded Orchestrator (Orchestrator + Guardian)
# =============================================================================

@dataclass
class GuardedOrchestrator:
    """Orchestrator with guardian oversight at key checkpoints."""

    orchestrator: "Orchestrator"
    guardian: "Guardian"
    checkpoints: list[str] = field(default_factory=lambda: [
        "pre_plan", "pre_execute", "post_execute", "pre_aggregate"
    ])
    _execution_lock: asyncio.Lock = field(init=False)

    def __post_init__(self):
        self._execution_lock = asyncio.Lock()

    async def execute(self, task: str) -> "OrchestratorResult":
        # Checkpoint: Validate input
        if "pre_plan" in self.checkpoints:
            validation = await self.guardian.validate_input(task)
            if validation.result == ValidationResult.REJECTED:
                return OrchestratorResult(
                    success=False,
                    error=f"Input rejected: {validation.reason}"
                )
            if validation.result == ValidationResult.MODIFIED:
                task = validation.modified_input

        # Plan
        subtasks = await self.orchestrator.plan(task)

        # Checkpoint: Validate plan
        if "pre_execute" in self.checkpoints:
            validation = await self.guardian.validate_plan(subtasks)
            if validation.result == ValidationResult.REJECTED:
                return OrchestratorResult(
                    success=False,
                    error=f"Plan rejected: {validation.reason}"
                )
            if validation.result == ValidationResult.MODIFIED:
                subtasks = validation.modified_plan

        # Execute with per-action validation
        results = {}
        for phase in self.orchestrator.create_execution_plan(subtasks):
            phase_results = await self.execute_phase_guarded(
                phase, subtasks, results)
            results.update(phase_results)

        # Checkpoint: Validate before aggregation
        if "pre_aggregate" in self.checkpoints:
            validation = await self.guardian.validate_results(results)
            if validation.result == ValidationResult.REJECTED:
                return OrchestratorResult(
                    success=False,
                    error=f"Results rejected: {validation.reason}"
                )
            if validation.result == ValidationResult.MODIFIED:
                results = validation.modified_results

        # Aggregate
        final = await self.orchestrator.aggregate(results, task)

        return OrchestratorResult(success=True, output=final)

    async def execute_phase_guarded(
        self,
        phase: list[str],
        subtasks: dict,
        context: dict
    ) -> dict:
        """Execute phase with guardian watching each action."""
        results = {}

        for subtask_id in phase:
            subtask = subtasks[subtask_id]

            # Guardian can inspect and approve/modify/reject each action
            if "post_execute" in self.checkpoints:
                result = await self.execute_subtask_guarded(subtask, context)
            else:
                result = await self.orchestrator.execute_subtask(subtask, context)

            results[subtask_id] = result

        return results

    async def execute_subtask_guarded(
        self,
        subtask: "Subtask",
        context: dict
    ) -> Any:
        """Execute subtask with real-time guardian monitoring."""

        # Hook into worker's tool execution
        worker = self.orchestrator.workers[subtask.worker_type]
        original_execute_tool = worker.execute_tool

        async def guarded_tool_execution(tool_call):
            # Pre-execution check
            approval = await self.guardian.approve_action(
                tool_call.name,
                tool_call.arguments
            )

            if not approval.approved:
                return {"error": f"Action blocked: {approval.reason}"}

            # Execute
            result = await original_execute_tool(tool_call)

            # Post-execution check
            validation = await self.guardian.validate_action_result(
                tool_call.name,
                result
            )

            if validation.result == ValidationResult.REJECTED:
                return {"error": f"Result blocked: {validation.reason}"}

            return result

        # Run the guarded execution on a per-call shallow copy of the
        # worker: patching the copy's tool executor leaves the shared
        # worker untouched and lets guarded subtasks run in parallel
        # instead of serializing on one global lock.
        guarded_worker = copy.copy(worker)
        guarded_worker.execute_tool = guarded_tool_execution
        result = await guarded_worker.execute(subtask, context)

        return result


# =============================================================================
# Swarm + Council Pipeline
# =============================================================================

# Helper functions for clustering (placeholder implementations)
async def get_embedding(content: str) -> list[float]:
    """Get embedding for content (placeholder)."""
    # In production, use actual embedding model
    return [hash(content) % 100 / 100.0 for _ in range(10)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class SwarmCouncilPipeline:
    """Swarm explores, council decides."""

    swarm: "SwarmCoordinator"
    council: "Council"
    exploration_budget: int = 50  # Max swarm steps
    min_options: int = 3

    async def decide(self, question: str, context: dict = None) -> "Decision":
        # Phase 1: Swarm exploration to generate options
        exploration_goal = f"Explore possible approaches to: {question}"

        swarm_result = await self.swarm.run(
            goal=exploration_goal,
            max_steps=self.exploration_budget
        )

        # Extract distinct options from swarm artifacts
        options = await self.extract_options(swarm_result.artifacts)

        if len(options) < self.min_options:
            # Swarm didn't find enough--add obvious alternatives
            options.extend(await self.generate_baseline_options(question))

        # Phase 2: Council deliberation on discovered options
        council_question = self.format_council_question(question, options)

        decision = await self.council.deliberate(
            question=council_question,
            context={
                "original_question": question,
                "explored_options": options,
                "exploration_summary": swarm_result.exploration_map,
                **(context or {})
            }
        )

        return Decision(
            question=question,
            chosen_option=decision.decision,
            alternatives=options,
            deliberation_transcript=decision.transcript,
            exploration_artifacts=swarm_result.artifacts
        )

    async def extract_options(self, artifacts: list["Artifact"]) -> list[dict]:
        """Extract distinct options from swarm exploration."""

        # Cluster artifacts by semantic similarity
        clusters = await self.cluster_artifacts(artifacts)

        options = []
        for cluster in clusters:
            # Synthesize each cluster into an option
            option = await self.synthesize_option(cluster)
            options.append(option)

        return options

    async def cluster_artifacts(
        self,
        artifacts: list["Artifact"]
    ) -> list[list["Artifact"]]:
        """Group similar artifacts together."""
        if not artifacts:
            return []

        # Get embeddings
        embeddings = [
            await get_embedding(a.content)
            for a in artifacts
        ]

        # Simple clustering by similarity threshold
        clusters = []
        used = set()

        for i, artifact in enumerate(artifacts):
            if i in used:
                continue

            cluster = [artifact]
            used.add(i)

            for j, other in enumerate(artifacts):
                if j in used:
                    continue

                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity > 0.7:  # Threshold
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def format_council_question(
        self,
        question: str,
        options: list[dict]
    ) -> str:
        """Format options for council deliberation."""
        options_text = "\n".join(
            f"Option {i+1}: {opt['name']}\n  {opt['description']}"
            for i, opt in enumerate(options)
        )

        return f"""
{question}

The following options emerged from exploration:

{options_text}

Deliberate on these options and recommend the best approach.
"""

    async def generate_baseline_options(self, question: str) -> list[dict]:
        """Generate baseline options when swarm doesn't find enough.

        # Placeholder: Override to generate domain-specific baseline options.
        """
        return []

    async def synthesize_option(self, cluster: list) -> dict:
        """Synthesize a cluster into an option.

        # Placeholder: Override to synthesize meaningful options from clusters.
        """
        return {"name": "option", "description": "synthesized option"}


# =============================================================================
# Orchestrator + Swarm Hybrid
# =============================================================================

class OrchestratorSwarmHybrid:
    """Orchestrator that can delegate to swarm for exploration phases."""

    def __init__(
        self,
        orchestrator: "Orchestrator",
        swarm_coordinator: "SwarmCoordinator"
    ):
        self.orchestrator = orchestrator
        self.swarm = swarm_coordinator

    async def execute(self, task: str) -> "HybridResult":
        # Plan with phase type annotations
        phases = await self.plan_hybrid(task)

        context = {}
        for phase in phases:
            if phase.phase_type == PhaseType.STRUCTURED:
                result = await self.execute_structured(phase, context)
            else:
                result = await self.execute_exploratory(phase, context)

            context[phase.id] = result

        return await self.aggregate(context, task)

    async def plan_hybrid(self, task: str) -> list[HybridPhase]:
        """Plan task with appropriate phase types."""

        planning_prompt = f"""
Analyze this task and create a phased execution plan.

Task: {task}

For each phase, determine if it should be:
- STRUCTURED: Has clear inputs, outputs, and methodology.
  Use for: data processing, formatting, known analyses
- EXPLORATORY: Requires discovering unknowns, creative search.
  Use for: research, brainstorming, investigating novel territory

Return a JSON array of phases:
[
  {{
    "id": "phase_id",
    "description": "what this phase does",
    "phase_type": "structured" or "exploratory",
    "dependencies": ["list", "of", "phase_ids"],
    "config": {{
      "for structured": {{"worker_types": ["research", "analysis"]}},
      "for exploratory": {{"max_steps": 30, "agents": 3}}
    }}
  }}
]
"""

        response = await self.orchestrator.llm.complete([
            {"role": "system", "content": "You are a task planning expert."},
            {"role": "user", "content": planning_prompt}
        ])

        try:
            phase_data = json.loads(response.content)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to single-phase execution if LLM response isn't valid or missing keys
            return [HybridPhase(
                id="fallback",
                description=task,
                phase_type=PhaseType.STRUCTURED,
                dependencies=[],
                config={}
            )]
        return [
            HybridPhase(
                id=p["id"],
                description=p["description"],
                phase_type=PhaseType(p["phase_type"]),
                dependencies=p.get("dependencies", []),
                config=p.get("config", {})
            )
            for p in phase_data
        ]

    async def execute_structured(
        self,
        phase: HybridPhase,
        context: dict
    ) -> dict:
        """Execute phase using standard orchestrator workers."""

        # Create subtasks from phase
        subtasks = await self.orchestrator.plan(
            f"{phase.description}\n\nContext: {json.dumps(context)}"
        )

        # Execute using orchestrator
        results = {}
        for subtask in subtasks:
            result = await self.orchestrator.execute_subtask(
                subtask,
                context
            )
            results[subtask.id] = result

        return results

    async def execute_exploratory(
        self,
        phase: HybridPhase,
        context: dict
    ) -> dict:
        """Execute phase using swarm exploration."""

        # Configure swarm for this phase
        max_steps = phase.config.get("max_steps", 50)

        # Run swarm
        result = await self.swarm.run(
            goal=f"{phase.description}\n\nContext: {json.dumps(context)}",
            max_steps=max_steps
        )

        return {
            "artifacts": result.artifacts,
            "exploration_map": result.exploration_map,
            "findings": await self.summarize_swarm_findings(result)
        }

    async def aggregate(self, context: dict, task: str) -> Any:
        """Aggregate results from all phases.

        # Placeholder: Override to provide meaningful aggregation.
        """
        return {"context": context, "task": task}

    async def summarize_swarm_findings(self, result: Any) -> str:
        """Summarize swarm exploration findings.

        # Placeholder: Override to extract insights from swarm results.
        """
        return "Summary of swarm findings"


# =============================================================================
# Constrained Council (Guardian + Council)
# =============================================================================

@dataclass
class Constraint:
    name: str
    description: str


@dataclass
class ConstrainedDecision:
    success: bool
    decision: Any = None
    error: str | None = None
    original_decision: Any = None
    transcript: list = field(default_factory=list)
    constraint_checks: list = field(default_factory=list)
    audit_trail: list = field(default_factory=list)


@dataclass
class CouncilDecision:
    question: str
    decision: Any
    transcript: list


@dataclass
class CouncilMessage:
    speaker: str
    phase: str
    content: str


@dataclass
class Councilor:
    id: str
    name: str
    perspective: str
    expertise: str
    system_prompt: str


@dataclass
class ConstrainedCouncil:
    """Council with guardian enforcing boundaries on deliberation."""

    council: "Council"
    guardian: "Guardian"
    constraints: list["Constraint"]

    async def deliberate(
        self,
        question: str,
        context: dict = None
    ) -> "ConstrainedDecision":

        # Pre-check: Is this question permissible?
        permissible = await self.guardian.check_question(
            question,
            self.constraints
        )
        if not permissible.allowed:
            return ConstrainedDecision(
                success=False,
                error=f"Question violates constraints: {permissible.violations}"
            )

        # Run deliberation with constrained councilors
        constrained_council = self.create_constrained_council()

        decision = await constrained_council.deliberate(
            question=self.augment_question_with_constraints(question),
            context=context
        )

        # Validate final decision
        validation = await self.guardian.validate_decision(
            decision,
            self.constraints
        )

        if not validation.compliant:
            # Try to remediate
            remediated = await self.remediate_decision(
                decision,
                validation.violations
            )
            if remediated:
                decision = remediated
            else:
                return ConstrainedDecision(
                    success=False,
                    error=f"Decision violates constraints: {validation.violations}",
                    original_decision=decision
                )

        return ConstrainedDecision(
            success=True,
            decision=decision.decision,
            transcript=decision.transcript,
            constraint_checks=validation.checks,
            audit_trail=self.generate_audit_trail(decision, validation)
        )

    def create_constrained_council(self) -> "Council":
        """Create council with constraint-aware prompts."""

        constraint_text = "\n".join(
            f"- {c.name}: {c.description}"
            for c in self.constraints
        )

        constrained_councilors = []
        for councilor in self.council.councilors:
            constrained_prompt = f"""{councilor.system_prompt}

IMPORTANT CONSTRAINTS:
The following constraints MUST be respected in your deliberation:
{constraint_text}

Any recommendation that violates these constraints is unacceptable.
"""
            constrained_councilors.append(
                Councilor(
                    id=councilor.id,
                    name=councilor.name,
                    perspective=councilor.perspective,
                    expertise=councilor.expertise,
                    system_prompt=constrained_prompt
                )
            )

        return Council(constrained_councilors, self.council.llm)

    async def remediate_decision(
        self,
        decision: "CouncilDecision",
        violations: list[str]
    ) -> "CouncilDecision | None":
        """Try to modify decision to comply with constraints."""

        violations_text = '\n'.join(f'- {v}' for v in violations)
        remediation_prompt = f"""
The council reached this decision:
{decision.decision}

However, it violates these constraints:
{violations_text}

Propose a modified decision that:
1. Preserves the core intent of the original decision
2. Complies with all constraints

If compliance is impossible without fundamentally changing the decision,
respond with "CANNOT_REMEDIATE".
"""

        sys_msg = "You remediate decisions to meet constraints."
        response = await self.council.llm.complete([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": remediation_prompt}
        ])

        if "CANNOT_REMEDIATE" in response.content:
            return None

        return CouncilDecision(
            question=decision.question,
            decision=response.content,
            transcript=decision.transcript + [
                CouncilMessage(
                    speaker="guardian",
                    phase="remediation",
                    content=f"Decision modified for compliance: {response.content}"
                )
            ]
        )

    def augment_question_with_constraints(self, question: str) -> str:
        """Augment question with constraint information.

        # Placeholder: Override to add constraint context to questions.
        """
        return question

    def generate_audit_trail(self, decision: Any, validation: Any) -> list:
        """Generate audit trail for the decision.

        # Placeholder: Override to generate comprehensive audit logs.
        """
        return []


# =============================================================================
# Pattern Router
# =============================================================================

class PatternRouter:
    """Routes tasks to appropriate patterns based on characteristics."""

    def __init__(
        self,
        llm_client,
        patterns: dict[PatternType, Any],
        scoring_weights: dict[str, dict[str, float]] = None
    ):
        self.llm = llm_client
        self.patterns = patterns
        self.weights = scoring_weights or self.default_weights()

    def default_weights(self) -> dict[str, dict[str, float]]:
        """Default weights for pattern scoring."""
        return {
            PatternType.ORCHESTRATOR.value: {
                "decomposable": 0.4,
                "dependencies_clear": 0.3,
                "needs_speed": 0.2,
                "ambiguity": -0.2,
                "unknowns": -0.1
            },
            PatternType.COUNCIL.value: {
                "needs_deliberation": 0.4,
                "ambiguity": 0.3,
                "consequence_severity": 0.2,
                "needs_speed": -0.3
            },
            PatternType.SWARM.value: {
                "unknowns": 0.4,
                "needs_creativity": 0.3,
                "ambiguity": 0.2,
                "decomposable": -0.2,
                "needs_speed": -0.2
            },
            PatternType.GUARDIAN.value: {
                "needs_safety": 0.5,
                "consequence_severity": 0.3,
                "reversibility": -0.2
            },
            PatternType.ORCHESTRATOR_GUARDIAN.value: {
                "decomposable": 0.3,
                "needs_safety": 0.3,
                "dependencies_clear": 0.2,
                "consequence_severity": 0.1
            },
            PatternType.COUNCIL_SWARM.value: {
                "unknowns": 0.3,
                "needs_deliberation": 0.3,
                "needs_creativity": 0.2,
                "ambiguity": 0.1
            },
            PatternType.ORCHESTRATOR_SWARM.value: {
                "decomposable": 0.25,
                "unknowns": 0.25,
                "needs_creativity": 0.2,
                "dependencies_clear": 0.15
            },
            PatternType.COUNCIL_GUARDIAN.value: {
                "needs_deliberation": 0.3,
                "needs_safety": 0.3,
                "consequence_severity": 0.2,
                "ambiguity": 0.1
            }
        }

    async def analyze_task(self, task: str,
                           context: dict = None) -> TaskCharacteristics:
        """Analyze task to determine characteristics."""

        analysis_prompt = f"""
Analyze this task to determine its characteristics.

Task: {task}

Context: {json.dumps(context) if context else "None"}

Rate each characteristic from 0.0 to 1.0:

1. decomposable: Can the task be broken into independent subtasks?
   0 = monolithic, 1 = clearly decomposable

2. dependencies_clear: Are dependencies between parts obvious?
   0 = unclear/complex dependencies, 1 = clear linear flow

3. ambiguity: Are there multiple valid interpretations?
   0 = single clear interpretation, 1 = highly ambiguous

4. unknowns: Does it require discovering new information?
   0 = all info available, 1 = significant research needed

5. consequence_severity: What's the impact of a wrong answer?
   0 = low stakes, 1 = severe consequences

6. reversibility: Can mistakes be easily undone?
   0 = irreversible, 1 = easily reversible

7. needs_deliberation: Would multiple perspectives help?
   0 = no benefit, 1 = significant benefit

8. needs_creativity: Does it require novel/unexpected solutions?
   0 = standard approach works, 1 = needs creative thinking

9. needs_safety: Are there safety/compliance concerns?
   0 = no concerns, 1 = significant safety needs

10. needs_speed: Is this time-sensitive?
    0 = no time pressure, 1 = very urgent

Respond with JSON:
{{
  "decomposable": 0.0,
  "dependencies_clear": 0.0,
  "ambiguity": 0.0,
  "unknowns": 0.0,
  "consequence_severity": 0.0,
  "reversibility": 0.0,
  "needs_deliberation": 0.0,
  "needs_creativity": 0.0,
  "needs_safety": 0.0,
  "needs_speed": 0.0
}}
"""

        sys_msg = "You analyze tasks to determine their characteristics."
        response = await self.llm.complete([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": analysis_prompt}
        ])

        try:
            data = json.loads(response.content)
            return TaskCharacteristics(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                f"Task analysis returned malformed JSON; using "
                f"default characteristics: {e}"
            )
            return TaskCharacteristics()

    def score_patterns(
        self,
        characteristics: TaskCharacteristics
    ) -> list[PatternScore]:
        """Score all patterns for given characteristics."""

        scores = []
        char_dict = {
            "decomposable": characteristics.decomposable,
            "dependencies_clear": characteristics.dependencies_clear,
            "ambiguity": characteristics.ambiguity,
            "unknowns": characteristics.unknowns,
            "consequence_severity": characteristics.consequence_severity,
            "reversibility": characteristics.reversibility,
            "needs_deliberation": characteristics.needs_deliberation,
            "needs_creativity": characteristics.needs_creativity,
            "needs_safety": characteristics.needs_safety,
            "needs_speed": characteristics.needs_speed
        }

        for pattern_type, weights in self.weights.items():
            score = 0.0
            rationale_parts = []

            for char_name, weight in weights.items():
                char_value = char_dict.get(char_name, 0)
                contribution = weight * char_value
                score += contribution

                if abs(contribution) > 0.1:
                    direction = "+" if contribution > 0 else "-"
                    rationale_parts.append(
                        f"{direction}{char_name}({char_value:.1f})"
                    )

            scores.append(PatternScore(
                pattern=PatternType(pattern_type),
                score=score,
                rationale=", ".join(rationale_parts)
            ))

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

    async def route(
        self,
        task: str,
        context: dict = None
    ) -> tuple[PatternType, Any]:
        """Route task to best pattern and return pattern instance."""

        # Analyze task
        characteristics = await self.analyze_task(task, context)

        # Score patterns
        scores = self.score_patterns(characteristics)

        # Select best available pattern
        for score in scores:
            if score.pattern in self.patterns:
                return score.pattern, self.patterns[score.pattern]

        # Fallback to orchestrator
        return PatternType.ORCHESTRATOR, self.patterns[PatternType.ORCHESTRATOR]

    async def execute(
        self,
        task: str,
        context: dict = None
    ) -> "RoutedResult":
        """Route and execute task."""

        pattern_type, pattern = await self.route(task, context)

        # Execute based on pattern type
        if pattern_type in [
            PatternType.ORCHESTRATOR,
            PatternType.ORCHESTRATOR_GUARDIAN
        ]:
            result = await pattern.execute(task)
        elif pattern_type in [
            PatternType.COUNCIL,
            PatternType.COUNCIL_GUARDIAN
        ]:
            result = await pattern.deliberate(task, context)
        elif pattern_type == PatternType.SWARM:
            result = await pattern.run(task)
        elif pattern_type == PatternType.COUNCIL_SWARM:
            result = await pattern.decide(task, context)
        elif pattern_type == PatternType.ORCHESTRATOR_SWARM:
            result = await pattern.execute(task)
        else:
            result = await pattern.execute(task)

        return RoutedResult(
            pattern_used=pattern_type,
            result=result,
            characteristics=await self.analyze_task(task, context)
        )


@dataclass
class RoutedResult:
    pattern_used: PatternType
    result: Any
    characteristics: TaskCharacteristics


# =============================================================================
# Adaptive Pipeline
# =============================================================================

@dataclass
class PipelinePhase:
    """A phase in an adaptive pipeline."""
    id: str
    description: str
    pattern_type: PatternType | None = None  # None = auto-select
    config: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


class AdaptivePipeline:
    """Pipeline that adapts pattern selection per phase."""

    def __init__(
        self,
        router: PatternRouter,
        default_patterns: dict[PatternType, Any]
    ):
        self.router = router
        self.patterns = default_patterns
        # Circuit breakers per stage type for failure isolation
        self._stage_circuit_breakers: dict[str, CircuitBreaker] = {}

    def _get_stage_cb(self, stage_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a pipeline stage.

        Each stage gets its own circuit breaker to isolate failures,
        preventing one failing stage from affecting others.

        Args:
            stage_name: Unique identifier for the stage.

        Returns:
            CircuitBreaker instance for this stage.
        """
        if stage_name not in self._stage_circuit_breakers:
            self._stage_circuit_breakers[stage_name] = CircuitBreaker(
                name=f"pipeline_stage_{stage_name}",
                failure_threshold=3,
                recovery_timeout=30.0
            )
        return self._stage_circuit_breakers[stage_name]

    async def execute(
        self,
        task: str,
        phases: list[PipelinePhase] = None
    ) -> "PipelineResult":
        """Execute pipeline with adaptive pattern selection."""

        if phases is None:
            phases = await self.plan_phases(task)

        context = {"original_task": task}
        phase_results = []

        for phase in self.order_phases(phases):
            # Build phase context from dependencies
            phase_context = {
                **context,
                "dependencies": {
                    dep: context.get(dep)
                    for dep in phase.dependencies
                }
            }

            # Select pattern for this phase
            if phase.pattern_type is None:
                pattern_type, pattern = await self.router.route(
                    phase.description,
                    phase_context
                )
            else:
                pattern_type = phase.pattern_type
                pattern = self.patterns[pattern_type]

            # Execute phase
            result = await self.execute_phase(
                phase,
                pattern,
                pattern_type,
                phase_context
            )

            # Store result
            context[phase.id] = result
            phase_results.append(PhaseResult(
                phase_id=phase.id,
                pattern_used=pattern_type,
                result=result
            ))

        return PipelineResult(
            task=task,
            phases=phase_results,
            final_output=await self.aggregate_results(phase_results, task)
        )

    async def plan_phases(self, task: str) -> list[PipelinePhase]:
        """Auto-plan phases for a task."""

        planning_prompt = f"""
Break this task into phases suitable for different execution patterns.

Task: {task}

For each phase, specify:
- id: unique identifier
- description: what this phase accomplishes
- dependencies: which prior phases it needs

Good phase boundaries:
- Research/exploration phases (discovery)
- Analysis/processing phases (structured work)
- Decision/recommendation phases (deliberation)
- Validation/review phases (safety)

Return JSON:
[
  {{
    "id": "phase_1",
    "description": "...",
    "dependencies": []
  }},
  ...
]
"""

        response = await self.router.llm.complete([
            {"role": "system", "content": "You design execution pipelines."},
            {"role": "user", "content": planning_prompt}
        ])

        try:
            phase_data = json.loads(response.content)
            return [
                PipelinePhase(
                    id=p["id"],
                    description=p["description"],
                    dependencies=p.get("dependencies", [])
                )
                for p in phase_data
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                f"Phase planning returned malformed JSON; falling "
                f"back to a single phase: {e}"
            )
            return [PipelinePhase(
                id="phase_1",
                description=task,
                dependencies=[]
            )]

    def order_phases(self, phases: list[PipelinePhase]) -> list[PipelinePhase]:
        """Topologically sort phases by dependencies."""

        # Build dependency graph
        graph = {p.id: set(p.dependencies) for p in phases}
        phase_map = {p.id: p for p in phases}

        ordered = []
        remaining = set(graph.keys())

        while remaining:
            # Find phases with no unmet dependencies
            ready = {
                pid for pid in remaining
                if graph[pid].issubset(set(p.id for p in ordered))
            }

            if not ready:
                raise ValueError("Circular dependency in phases")

            # Add ready phases (could parallelize these)
            for pid in sorted(ready):  # Sort for determinism
                ordered.append(phase_map[pid])
                remaining.remove(pid)

        return ordered

    async def execute_phase(
        self,
        phase: PipelinePhase,
        pattern: Any,
        pattern_type: PatternType,
        context: dict
    ) -> Any:
        """Execute a single phase with the selected pattern.

        Uses circuit breaker to isolate failures per stage, preventing
        cascading failures across the pipeline.
        """
        # Get circuit breaker for this stage
        cb = self._get_stage_cb(phase.id)

        # Check if circuit breaker allows execution
        if not cb.allow():
            raise CircuitBreakerOpen(
                f"pipeline_stage_{phase.id}",
                cb.recovery_timeout
            )

        task_description = f"""
{phase.description}

Context from previous phases:
{json.dumps(context.get('dependencies', {}), indent=2)}
"""

        try:
            if pattern_type in [
                PatternType.ORCHESTRATOR,
                PatternType.ORCHESTRATOR_GUARDIAN,
                PatternType.ORCHESTRATOR_SWARM
            ]:
                result = await pattern.execute(task_description)

            elif pattern_type in [
                PatternType.COUNCIL,
                PatternType.COUNCIL_GUARDIAN
            ]:
                result = await pattern.deliberate(task_description, context)

            elif pattern_type == PatternType.SWARM:
                result = await pattern.run(task_description)

            elif pattern_type == PatternType.COUNCIL_SWARM:
                result = await pattern.decide(task_description, context)

            else:
                # Generic execution
                result = await pattern.execute(task_description)

            # Record success with circuit breaker
            cb.record_success()
            return result

        except Exception as e:
            # Record failure with circuit breaker
            cb.record_failure()
            raise

    async def aggregate_results(
        self,
        phase_results: list["PhaseResult"],
        task: str
    ) -> str:
        """Aggregate all phase results into final output."""

        def fmt_result(r):
            if isinstance(r, dict):
                return json.dumps(r, indent=2)
            return str(r)

        results_summary = "\n\n".join(
            f"Phase: {pr.phase_id} (using {pr.pattern_used.value})\n"
            f"Result: {fmt_result(pr.result)}"
            for pr in phase_results
        )

        aggregation_prompt = f"""
Synthesize the results from all phases into a cohesive response.

Original task: {task}

Phase results:
{results_summary}

Create a comprehensive final output that addresses the original task
using the work from all phases.
"""

        sys_msg = "You synthesize multi-phase work into coherent output."
        response = await self.router.llm.complete([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": aggregation_prompt}
        ])

        return response.content


@dataclass
class PhaseResult:
    """Result of executing a phase."""
    phase_id: str
    pattern_used: PatternType
    result: Any


@dataclass
class PipelineResult:
    """Result of executing a pipeline."""
    task: str
    phases: list[PhaseResult]
    final_output: Any


# =============================================================================
# Complete Hybrid Router - Production Implementation
# =============================================================================

class TaskAnalyzer:
    """Analyzes tasks to determine characteristics."""

    def __init__(self, llm_client, max_cache_size: int = 1000):
        self.llm = llm_client
        self._cache: dict[str, TaskCharacteristics] = {}
        self._max_cache_size = max_cache_size
        self._cache_order: list[str] = []  # Track insertion order for LRU eviction

    async def analyze(
        self,
        task: str,
        context: dict | None = None
    ) -> TaskCharacteristics:
        """Analyze task characteristics."""

        # Check cache
        cache_key = f"{task}:{json.dumps(context or {}, sort_keys=True)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_analysis_prompt(task, context)

        try:
            response = await self.llm.complete([
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ])

            data = json.loads(response.content)
            characteristics = TaskCharacteristics(**data)

        except json.JSONDecodeError:
            logger.warning("Failed to parse task analysis, using defaults")
            characteristics = TaskCharacteristics()
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Task analysis failed: {e}")
            characteristics = TaskCharacteristics()

        # LRU eviction if at capacity
        if len(self._cache) >= self._max_cache_size:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
        self._cache[cache_key] = characteristics
        self._cache_order.append(cache_key)
        return characteristics

    def _system_prompt(self) -> str:
        return """You analyze tasks to determine their characteristics.
Respond ONLY with valid JSON matching the requested format.
Be precise in your ratings - use the full 0.0 to 1.0 range."""

    def _build_analysis_prompt(
        self,
        task: str,
        context: dict | None
    ) -> str:
        return f"""
Analyze this task and rate each characteristic from 0.0 to 1.0.

Task: {task}

Context: {json.dumps(context) if context else "None"}

Characteristics to rate:
- decomposable: Can be broken into independent subtasks
- dependencies_clear: Dependencies between parts are obvious
- ambiguity: Multiple valid interpretations exist
- unknowns: Requires discovering new information
- consequence_severity: Impact of wrong answer is high
- reversibility: Mistakes can be easily undone
- needs_deliberation: Multiple perspectives would help
- needs_creativity: Requires novel solutions
- needs_safety: Has safety/compliance concerns
- needs_speed: Time-sensitive

Respond with JSON only:
{{"decomposable": 0.0, "dependencies_clear": 0.0, "ambiguity": 0.0,
  "unknowns": 0.0, "consequence_severity": 0.0, "reversibility": 0.0,
  "needs_deliberation": 0.0, "needs_creativity": 0.0,
  "needs_safety": 0.0, "needs_speed": 0.0}}
"""


class PatternScorer:
    """Scores patterns based on task characteristics."""

    def __init__(self, weights: dict[str, dict[str, float]] | None = None):
        self.weights = weights or self._default_weights()

    def _default_weights(self) -> dict[str, dict[str, float]]:
        return {
            PatternType.ORCHESTRATOR.value: {
                "decomposable": 0.35,
                "dependencies_clear": 0.25,
                "needs_speed": 0.15,
                "ambiguity": -0.15,
                "unknowns": -0.1
            },
            PatternType.COUNCIL.value: {
                "needs_deliberation": 0.35,
                "ambiguity": 0.25,
                "consequence_severity": 0.15,
                "needs_speed": -0.25
            },
            PatternType.SWARM.value: {
                "unknowns": 0.35,
                "needs_creativity": 0.25,
                "ambiguity": 0.15,
                "decomposable": -0.15,
                "needs_speed": -0.1
            },
            PatternType.GUARDIAN.value: {
                "needs_safety": 0.4,
                "consequence_severity": 0.3,
                "reversibility": -0.2
            },
            PatternType.ORCHESTRATOR_GUARDIAN.value: {
                "decomposable": 0.25,
                "needs_safety": 0.25,
                "dependencies_clear": 0.15,
                "consequence_severity": 0.1,
                "needs_speed": 0.1
            },
            PatternType.COUNCIL_SWARM.value: {
                "unknowns": 0.25,
                "needs_deliberation": 0.25,
                "needs_creativity": 0.2,
                "ambiguity": 0.15
            },
            PatternType.ORCHESTRATOR_SWARM.value: {
                "decomposable": 0.2,
                "unknowns": 0.25,
                "needs_creativity": 0.2,
                "dependencies_clear": 0.1,
                "needs_speed": 0.1
            },
            PatternType.COUNCIL_GUARDIAN.value: {
                "needs_deliberation": 0.25,
                "needs_safety": 0.25,
                "consequence_severity": 0.2,
                "ambiguity": 0.15
            }
        }

    def score(
        self,
        characteristics: TaskCharacteristics
    ) -> list[tuple[PatternType, float, str]]:
        """Score all patterns, return sorted by score descending."""

        char_dict = characteristics.to_dict()
        scores = []

        for pattern_name, weights in self.weights.items():
            score = 0.0
            contributions = []

            for char_name, weight in weights.items():
                char_value = char_dict.get(char_name, 0)
                contribution = weight * char_value
                score += contribution

                if abs(contribution) > 0.05:
                    sign = "+" if contribution > 0 else ""
                    contributions.append(
                        f"{char_name}: {sign}{contribution:.2f}"
                    )

            rationale = "; ".join(contributions) if contributions else "baseline"
            scores.append((PatternType(pattern_name), score, rationale))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class HybridRouter:
    """
    Routes tasks to appropriate patterns based on analysis.

    Usage:
        router = HybridRouter(llm_client)
        router.register_pattern(PatternType.ORCHESTRATOR, orchestrator)
        router.register_pattern(PatternType.COUNCIL, council)

        result = await router.execute("Analyze competitor landscape")
    """

    def __init__(
        self,
        llm_client,
        analyzer: TaskAnalyzer | None = None,
        scorer: PatternScorer | None = None
    ):
        self.llm = llm_client
        self.analyzer = analyzer or TaskAnalyzer(llm_client)
        self.scorer = scorer or PatternScorer()
        self.patterns: dict[PatternType, Any] = {}
        self._execution_hooks: list[Callable] = []

    def register_pattern(self, pattern_type: PatternType, pattern: Any):
        """Register a pattern implementation."""
        self.patterns[pattern_type] = pattern

    def add_execution_hook(
        self,
        hook: Callable[["RoutingDecision", Any], None]
    ):
        """Add hook called after each execution."""
        self._execution_hooks.append(hook)

    async def route(
        self,
        task: str,
        context: dict | None = None
    ) -> RoutingDecision:
        """Determine best pattern for task."""

        # Analyze task
        characteristics = await self.analyzer.analyze(task, context)

        # Score patterns
        scores = self.scorer.score(characteristics)

        # Find best available pattern
        for pattern_type, score, rationale in scores:
            if pattern_type in self.patterns:
                alternatives = [
                    (pt, s) for pt, s, _ in scores[1:5]
                    if pt in self.patterns
                ]

                return RoutingDecision(
                    pattern=pattern_type,
                    confidence=min(1.0, max(0.0, (score + 1) / 2)),
                    rationale=rationale,
                    characteristics=characteristics,
                    alternatives=alternatives
                )

        raise ValueError("No patterns registered")

    async def execute(
        self,
        task: str,
        context: dict | None = None,
        force_pattern: PatternType | None = None
    ) -> ExecutionResult:
        """Route and execute task."""

        start_time = datetime.now()

        try:
            # Route (unless forced)
            if force_pattern:
                if force_pattern not in self.patterns:
                    return ExecutionResult(
                        success=False,
                        error=f"Pattern {force_pattern} not registered"
                    )
                routing = RoutingDecision(
                    pattern=force_pattern,
                    confidence=1.0,
                    rationale="forced",
                    characteristics=await self.analyzer.analyze(task, context)
                )
            else:
                routing = await self.route(task, context)

            # Get pattern
            pattern = self.patterns[routing.pattern]

            # Execute based on pattern type
            output = await self._execute_pattern(
                pattern,
                routing.pattern,
                task,
                context
            )

            # Calculate execution time
            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            result = ExecutionResult(
                success=True,
                output=output,
                pattern_used=routing.pattern,
                routing_decision=routing,
                execution_time_ms=execution_time
            )

            # Call hooks
            for hook in self._execution_hooks:
                try:
                    hook(routing, result)
                except Exception as e:
                    logger.warning(f"Execution hook failed: {e}")

            return result

        except Exception as e:
            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )

    async def _execute_pattern(
        self,
        pattern: Any,
        pattern_type: PatternType,
        task: str,
        context: dict | None,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Any:
        """Execute the appropriate method based on pattern type with retry logic.

        Args:
            pattern: The pattern instance to execute.
            pattern_type: The type of pattern being executed.
            task: The task description.
            context: Optional context dictionary.
            max_retries: Maximum retry attempts on failure (default: 3).
            base_delay: Base delay for exponential backoff (default: 1.0s).

        Returns:
            The result of pattern execution.

        Raises:
            ValueError: If pattern has no suitable execution method.
        """
        # Determine the execution function based on pattern type
        async def execute_func():
            # Deliberative patterns (Council variants)
            if pattern_type in [
                PatternType.COUNCIL,
                PatternType.COUNCIL_GUARDIAN
            ]:
                if hasattr(pattern, 'deliberate'):
                    return await pattern.deliberate(task, context)

            # Exploratory patterns (Swarm)
            if pattern_type == PatternType.SWARM:
                if hasattr(pattern, 'run'):
                    return await pattern.run(task)

            # Hybrid exploration + deliberation
            if pattern_type == PatternType.COUNCIL_SWARM:
                if hasattr(pattern, 'decide'):
                    return await pattern.decide(task, context)

            # Default: execute method
            if hasattr(pattern, 'execute'):
                return await pattern.execute(task)

            raise ValueError(
                f"Pattern {pattern_type} has no suitable execution method"
            )

        # Execute with retry logic
        return await retry_with_backoff(
            execute_func,
            max_retries=max_retries,
            base_delay=base_delay,
            exceptions=(RuntimeError, asyncio.TimeoutError, ConnectionError)
        )


# =============================================================================
# Production Adaptive Pipeline
# =============================================================================

@dataclass
class Phase:
    """A phase in an adaptive pipeline."""
    id: str
    description: str
    pattern_hint: PatternType | None = None
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class ProductionPhaseResult:
    """Result of executing a phase."""
    phase_id: str
    pattern_used: PatternType
    output: Any
    execution_time_ms: int
    success: bool = True
    error: str | None = None


@dataclass
class ProductionPipelineResult:
    """Result of executing a pipeline."""
    task: str
    phases: list[ProductionPhaseResult]
    final_output: Any
    total_execution_time_ms: int
    success: bool = True


class ProductionAdaptivePipeline:
    """
    Multi-phase pipeline with per-phase pattern selection.

    Features circuit breakers per phase to isolate failures and prevent
    cascading failures across the pipeline.

    Usage:
        pipeline = ProductionAdaptivePipeline(router)
        result = await pipeline.execute(
            "Research and recommend solution for X",
            phases=[
                Phase("research", "Explore the problem space"),
                Phase("analyze", "Structure findings", dependencies=["research"]),
                Phase("decide", "Recommend action", dependencies=["analyze"])
            ]
        )
    """

    def __init__(self, router: HybridRouter):
        self.router = router
        # Circuit breakers per stage type for failure isolation
        self._stage_circuit_breakers: dict[str, CircuitBreaker] = {}

    def _get_stage_cb(self, stage_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a pipeline stage.

        Each stage gets its own circuit breaker to isolate failures,
        preventing one failing stage from affecting others.

        Args:
            stage_name: Unique identifier for the stage.

        Returns:
            CircuitBreaker instance for this stage.
        """
        if stage_name not in self._stage_circuit_breakers:
            self._stage_circuit_breakers[stage_name] = CircuitBreaker(
                name=f"pipeline_stage_{stage_name}",
                failure_threshold=3,
                recovery_timeout=30.0
            )
        return self._stage_circuit_breakers[stage_name]

    async def execute(
        self,
        task: str,
        phases: list[Phase] | None = None
    ) -> ProductionPipelineResult:
        """Execute pipeline with adaptive pattern selection per phase."""

        start_time = datetime.now()

        if phases is None:
            phases = await self._auto_plan_phases(task)

        # Order phases by dependencies
        ordered_phases = self._topological_sort(phases)

        # Execute phases
        context = {"original_task": task}
        phase_results = []

        for phase in ordered_phases:
            # Build phase context
            phase_context = {
                **context,
                "phase_dependencies": {
                    dep: context.get(dep)
                    for dep in phase.dependencies
                    if dep in context
                }
            }

            # Execute phase
            phase_result = await self._execute_phase(phase, phase_context)
            phase_results.append(phase_result)

            # Update context
            if phase_result.success:
                context[phase.id] = phase_result.output
            else:
                # Decide whether to continue or abort
                if not self._can_continue(phase, phase_results):
                    break

        # Aggregate results
        final_output = await self._aggregate(phase_results, task)

        total_time = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        return ProductionPipelineResult(
            task=task,
            phases=phase_results,
            final_output=final_output,
            total_execution_time_ms=total_time,
            success=all(pr.success for pr in phase_results)
        )

    async def _execute_phase(
        self,
        phase: Phase,
        context: dict
    ) -> ProductionPhaseResult:
        """Execute a single phase with circuit breaker protection.

        Uses circuit breaker to isolate failures per stage, preventing
        cascading failures across the pipeline.
        """
        start_time = datetime.now()

        # Get circuit breaker for this stage
        cb = self._get_stage_cb(phase.id)

        # Check if circuit breaker allows execution
        if not cb.allow():
            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=PatternType.ORCHESTRATOR,
                output=None,
                execution_time_ms=0,
                success=False,
                error=f"Circuit breaker open for stage {phase.id}. "
                      f"Retry after {cb.recovery_timeout:.1f}s"
            )

        try:
            # Build task description for this phase
            phase_task = self._build_phase_task(phase, context)

            # Execute with timeout
            result = await asyncio.wait_for(
                self.router.execute(
                    phase_task,
                    context,
                    force_pattern=phase.pattern_hint
                ),
                timeout=phase.timeout_seconds
            )

            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            # Record success or failure with circuit breaker
            if result.success:
                cb.record_success()
            else:
                cb.record_failure()

            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=result.pattern_used or PatternType.ORCHESTRATOR,
                output=result.output,
                execution_time_ms=execution_time,
                success=result.success,
                error=result.error
            )

        except asyncio.TimeoutError:
            cb.record_failure()
            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=PatternType.ORCHESTRATOR,
                output=None,
                execution_time_ms=phase.timeout_seconds * 1000,
                success=False,
                error=f"Phase timed out after {phase.timeout_seconds}s"
            )
        except Exception as e:
            cb.record_failure()
            execution_time = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=PatternType.ORCHESTRATOR,
                output=None,
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )

    def _build_phase_task(self, phase: Phase, context: dict) -> str:
        """Build task description for a phase."""

        deps_text = ""
        if phase.dependencies and context.get("phase_dependencies"):
            deps_text = "\n\nContext from previous phases:\n"
            for dep_id, dep_result in context["phase_dependencies"].items():
                if dep_result is not None:
                    result_str = (
                        json.dumps(dep_result, indent=2)
                        if isinstance(dep_result, (dict, list))
                        else str(dep_result)
                    )
                    deps_text += f"\n[{dep_id}]:\n{result_str}\n"

        return f"{phase.description}{deps_text}"

    def _topological_sort(self, phases: list[Phase]) -> list[Phase]:
        """Sort phases by dependencies."""

        phase_map = {p.id: p for p in phases}
        graph = {p.id: set(p.dependencies) for p in phases}

        ordered = []
        remaining = set(graph.keys())

        while remaining:
            ready = {
                pid for pid in remaining
                if graph[pid].issubset({p.id for p in ordered})
            }

            if not ready:
                raise ValueError("Circular dependency in phases")

            for pid in sorted(ready):
                ordered.append(phase_map[pid])
                remaining.remove(pid)

        return ordered

    def _can_continue(
        self,
        failed_phase: Phase,
        results: list[ProductionPhaseResult]
    ) -> bool:
        """Determine if pipeline can continue after phase failure."""
        # Simple policy: continue if no other phase depends on failed phase
        # More sophisticated policies could be implemented
        return True

    async def _auto_plan_phases(self, task: str) -> list[Phase]:
        """Auto-generate phases for a task."""

        prompt = f"""
Break this task into execution phases.

Task: {task}

Create 2-4 phases with clear boundaries. Good phases:
- Research/exploration: discover information
- Analysis/processing: structure and analyze
- Decision/synthesis: conclude and recommend
- Validation/review: check quality

Return JSON array:
[
  {{"id": "phase_1", "description": "...", "dependencies": []}},
  {{"id": "phase_2", "description": "...", "dependencies": ["phase_1"]}}
]
"""

        sys_msg = "You design execution pipelines. Return only valid JSON."
        response = await self.router.llm.complete([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ])

        try:
            phase_data = json.loads(response.content)
            return [
                Phase(
                    id=p["id"],
                    description=p["description"],
                    dependencies=p.get("dependencies", [])
                )
                for p in phase_data
            ]
        except json.JSONDecodeError:
            # Fallback: single phase
            return [Phase(id="main", description=task)]

    async def _aggregate(
        self,
        results: list[ProductionPhaseResult],
        task: str
    ) -> Any:
        """Aggregate phase results into final output."""

        if not results:
            return None

        # If only one phase, return its output
        if len(results) == 1:
            return results[0].output

        # Build summary
        def fmt_out(o):
            if isinstance(o, (dict, list)):
                return json.dumps(o, indent=2)
            return str(o)

        results_text = "\n\n".join(
            f"Phase [{pr.phase_id}] ({pr.pattern_used.value}):\n"
            f"{'Success' if pr.success else 'Failed'}: {fmt_out(pr.output)}"
            for pr in results
        )

        prompt = f"""
Synthesize phase results into a final response.

Original task: {task}

Phase results:
{results_text}

Create a cohesive final output addressing the original task.
"""

        response = await self.router.llm.complete([
            {"role": "system", "content": "You synthesize multi-phase results."},
            {"role": "user", "content": prompt}
        ])

        return response.content


# =============================================================================
# Protocol Definitions for Pattern Integration
# =============================================================================
# These protocols define the expected interface for each pattern component.
# Use these to ensure your implementations are compatible with the hybrid
# architecture, and for type checking with mypy/pyright.
# =============================================================================


class OrchestratorProtocol(Protocol):
    """Protocol defining the Orchestrator interface.

    Implement this protocol when creating custom orchestrators for use
    with GuardedOrchestrator or OrchestratorSwarmHybrid.

    Example implementation:
        class MyOrchestrator:
            def __init__(self, llm, workers: dict[str, WorkerProtocol]):
                self.llm = llm
                self.workers = workers

            async def plan(self, task: str) -> dict[str, Subtask]:
                # Use LLM to decompose task into subtasks
                ...

            def create_execution_plan(self, subtasks: dict) -> list[list[str]]:
                # Return phases of subtask IDs that can run in parallel
                ...

            async def execute_subtask(self, subtask: Subtask, context: dict) -> Any:
                worker = self.workers[subtask.worker_type]
                return await worker.execute(subtask, context)

            async def aggregate(self, results: dict, task: str) -> Any:
                # Combine subtask results into final output
                ...
    """

    llm: Any
    workers: dict[str, Any]

    async def plan(self, task: str) -> dict:
        """Decompose task into subtasks."""
        ...

    def create_execution_plan(self, subtasks: dict) -> list:
        """Create ordered phases of subtask execution."""
        ...

    async def execute_subtask(self, subtask: Any, context: dict) -> Any:
        """Execute a single subtask."""
        ...

    async def aggregate(self, results: dict, task: str) -> Any:
        """Aggregate subtask results into final output."""
        ...


class GuardianProtocol(Protocol):
    """Protocol defining the Guardian interface.

    Implement this protocol for custom guardians in GuardedOrchestrator
    or ConstrainedCouncil.

    Example implementation:
        class MyGuardian:
            def __init__(self, policies: list[Policy]):
                self.policies = policies

            async def validate_input(self, task: str) -> ValidationResponse:
                for policy in self.policies:
                    if not policy.allows_input(task):
                        return ValidationResponse(
                            result=ValidationResult.REJECTED,
                            reason=f"Policy {policy.name} rejected input"
                        )
                return ValidationResponse(result=ValidationResult.APPROVED)
    """

    async def validate_input(self, task: str) -> "ValidationResponse":
        """Validate task input before processing."""
        ...

    async def validate_plan(self, subtasks: dict) -> "ValidationResponse":
        """Validate execution plan before running."""
        ...

    async def validate_results(self, results: dict) -> "ValidationResponse":
        """Validate results before returning."""
        ...

    async def approve_action(self, name: str, arguments: dict) -> "ApprovalResponse":
        """Approve or reject a specific action."""
        ...

    async def validate_action_result(self, name: str, result: Any) -> "ValidationResponse":
        """Validate the result of an action."""
        ...

    async def check_question(self, question: str, constraints: list) -> "QuestionCheckResponse":
        """Check if a question is permissible given constraints."""
        ...

    async def validate_decision(self, decision: Any, constraints: list) -> "DecisionValidationResponse":
        """Validate a decision against constraints."""
        ...


class SwarmCoordinatorProtocol(Protocol):
    """Protocol defining the SwarmCoordinator interface.

    Implement this protocol for custom swarm implementations in
    SwarmCouncilPipeline or OrchestratorSwarmHybrid.

    Example implementation:
        class MySwarmCoordinator:
            def __init__(self, agents: list[SwarmAgent], shared_memory: SharedMemory):
                self.agents = agents
                self.memory = shared_memory

            async def run(self, goal: str, max_steps: int = 100) -> SwarmResult:
                for step in range(max_steps):
                    for agent in self.agents:
                        result = await agent.act(goal, self.memory)
                        self.memory.store(result)
                return SwarmResult(
                    artifacts=self.memory.get_artifacts(),
                    exploration_map=self.memory.get_map()
                )
    """

    async def run(self, goal: str, max_steps: int = 100) -> Any:
        """Run swarm exploration toward a goal."""
        ...


class CouncilProtocol(Protocol):
    """Protocol defining the Council interface.

    Implement this protocol for custom council implementations in
    SwarmCouncilPipeline or ConstrainedCouncil.

    Example implementation:
        class MyCouncil:
            def __init__(self, councilors: list[Councilor], llm):
                self.councilors = councilors
                self.llm = llm

            async def deliberate(self, question: str, context: dict = None) -> CouncilDecision:
                transcript = []
                for round in range(3):
                    for councilor in self.councilors:
                        response = await councilor.respond(question, transcript, context)
                        transcript.append(response)
                decision = await self.synthesize(transcript)
                return CouncilDecision(question=question, decision=decision, transcript=transcript)
    """

    councilors: list
    llm: Any

    async def deliberate(self, question: str, context: dict = None) -> Any:
        """Conduct deliberation on a question."""
        ...


# =============================================================================
# Response Data Classes
# =============================================================================

@dataclass
class ValidationResponse:
    """Response from Guardian validation methods."""
    result: ValidationResult = ValidationResult.APPROVED
    reason: str = ""
    modified_input: str = ""
    modified_plan: Any = None
    modified_results: Any = None


@dataclass
class ApprovalResponse:
    """Response from Guardian approval methods."""
    approved: bool = True
    reason: str = ""


@dataclass
class QuestionCheckResponse:
    """Response from Guardian question check."""
    allowed: bool = True
    violations: list[str] = field(default_factory=list)


@dataclass
class DecisionValidationResponse:
    """Response from Guardian decision validation."""
    compliant: bool = True
    checks: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


@dataclass
class SwarmResult:
    """Result from swarm exploration."""
    artifacts: list["Artifact"] = field(default_factory=list)
    exploration_map: dict = field(default_factory=dict)


# =============================================================================
# Mock Implementations for Testing
# =============================================================================
# These mock implementations can be used in unit tests to verify hybrid
# architecture behavior without real LLM calls or complex dependencies.
# =============================================================================


class MockOrchestrator:
    """Mock Orchestrator for testing hybrid architectures.

    Usage in tests:
        orchestrator = MockOrchestrator()
        orchestrator.set_plan_result({
            "subtask_1": Subtask(id="subtask_1", worker_type="research"),
            "subtask_2": Subtask(id="subtask_2", worker_type="analysis")
        })

        guarded = GuardedOrchestrator(orchestrator=orchestrator, guardian=mock_guardian)
        result = await guarded.execute("Test task")
    """

    def __init__(self, llm=None, workers=None):
        self.llm = llm
        self.workers = workers or {}
        self._plan_result: dict = {}
        self._execution_plan: list = []
        self._subtask_results: dict = {}
        self._aggregate_result: Any = None
        self.call_log: list[dict] = []

    def set_plan_result(self, result: dict):
        """Set the result that plan() will return."""
        self._plan_result = result

    def set_execution_plan(self, plan: list):
        """Set the result that create_execution_plan() will return."""
        self._execution_plan = plan

    def set_subtask_result(self, subtask_id: str, result: Any):
        """Set the result for a specific subtask."""
        self._subtask_results[subtask_id] = result

    def set_aggregate_result(self, result: Any):
        """Set the result that aggregate() will return."""
        self._aggregate_result = result

    async def plan(self, task: str) -> dict:
        """Return configured plan result."""
        self.call_log.append({"method": "plan", "task": task})
        return self._plan_result

    def create_execution_plan(self, subtasks: dict) -> list:
        """Return configured execution plan."""
        self.call_log.append({"method": "create_execution_plan", "subtasks": subtasks})
        return self._execution_plan or [list(subtasks.keys())]

    async def execute_subtask(self, subtask: Any, context: dict) -> Any:
        """Return configured subtask result."""
        subtask_id = getattr(subtask, 'id', str(subtask))
        self.call_log.append({
            "method": "execute_subtask",
            "subtask_id": subtask_id,
            "context": context
        })
        return self._subtask_results.get(subtask_id, {"status": "success", "mock": True})

    async def aggregate(self, results: dict, task: str) -> Any:
        """Return configured aggregate result."""
        self.call_log.append({"method": "aggregate", "results": results, "task": task})
        return self._aggregate_result if self._aggregate_result is not None else results


class MockGuardian:
    """Mock Guardian for testing hybrid architectures.

    Usage in tests:
        guardian = MockGuardian()
        guardian.set_input_validation(ValidationResponse(
            result=ValidationResult.REJECTED,
            reason="Test rejection"
        ))

        guarded = GuardedOrchestrator(orchestrator=mock_orch, guardian=guardian)
        result = await guarded.execute("Dangerous task")
        assert result.success == False
    """

    def __init__(self):
        self._input_validation = ValidationResponse(result=ValidationResult.APPROVED)
        self._plan_validation = ValidationResponse(result=ValidationResult.APPROVED)
        self._results_validation = ValidationResponse(result=ValidationResult.APPROVED)
        self._action_approval = ApprovalResponse(approved=True)
        self._action_result_validation = ValidationResponse(result=ValidationResult.APPROVED)
        self._question_check = QuestionCheckResponse(allowed=True)
        self._decision_validation = DecisionValidationResponse(compliant=True)
        self.call_log: list[dict] = []

    def set_input_validation(self, response: ValidationResponse):
        """Set response for validate_input()."""
        self._input_validation = response

    def set_plan_validation(self, response: ValidationResponse):
        """Set response for validate_plan()."""
        self._plan_validation = response

    def set_results_validation(self, response: ValidationResponse):
        """Set response for validate_results()."""
        self._results_validation = response

    def set_action_approval(self, response: ApprovalResponse):
        """Set response for approve_action()."""
        self._action_approval = response

    def set_question_check(self, response: QuestionCheckResponse):
        """Set response for check_question()."""
        self._question_check = response

    def set_decision_validation(self, response: DecisionValidationResponse):
        """Set response for validate_decision()."""
        self._decision_validation = response

    async def validate_input(self, task: str) -> ValidationResponse:
        self.call_log.append({"method": "validate_input", "task": task})
        return self._input_validation

    async def validate_plan(self, subtasks: dict) -> ValidationResponse:
        self.call_log.append({"method": "validate_plan", "subtasks": subtasks})
        return self._plan_validation

    async def validate_results(self, results: dict) -> ValidationResponse:
        self.call_log.append({"method": "validate_results", "results": results})
        return self._results_validation

    async def approve_action(self, name: str, arguments: dict) -> ApprovalResponse:
        self.call_log.append({"method": "approve_action", "name": name, "arguments": arguments})
        return self._action_approval

    async def validate_action_result(self, name: str, result: Any) -> ValidationResponse:
        self.call_log.append({"method": "validate_action_result", "name": name, "result": result})
        return self._action_result_validation

    async def check_question(self, question: str, constraints: list) -> QuestionCheckResponse:
        self.call_log.append({"method": "check_question", "question": question})
        return self._question_check

    async def validate_decision(self, decision: Any, constraints: list) -> DecisionValidationResponse:
        self.call_log.append({"method": "validate_decision", "decision": decision})
        return self._decision_validation


class MockSwarmCoordinator:
    """Mock SwarmCoordinator for testing hybrid architectures.

    Usage in tests:
        swarm = MockSwarmCoordinator()
        swarm.set_result(SwarmResult(
            artifacts=[Artifact(content="Found option A"), Artifact(content="Found option B")],
            exploration_map={"nodes": 5, "edges": 4}
        ))

        pipeline = SwarmCouncilPipeline(swarm=swarm, council=mock_council)
        decision = await pipeline.decide("What approach should we take?")
    """

    def __init__(self):
        self._result = SwarmResult()
        self.call_log: list[dict] = []

    def set_result(self, result: SwarmResult):
        """Set the result that run() will return."""
        self._result = result

    async def run(self, goal: str, max_steps: int = 100) -> SwarmResult:
        self.call_log.append({"method": "run", "goal": goal, "max_steps": max_steps})
        return self._result


class MockCouncil:
    """Mock Council for testing hybrid architectures.

    Usage in tests:
        council = MockCouncil()
        council.set_decision(CouncilDecision(
            question="Test question",
            decision="Option A is best because...",
            transcript=[...]
        ))

        pipeline = SwarmCouncilPipeline(swarm=mock_swarm, council=council)
        result = await pipeline.decide("Which option?")
    """

    def __init__(self, councilors=None, llm=None):
        self.councilors = councilors or []
        self.llm = llm
        self._decision = CouncilDecision(question="", decision=None, transcript=[])
        self.call_log: list[dict] = []

    def set_decision(self, decision: CouncilDecision):
        """Set the decision that deliberate() will return."""
        self._decision = decision

    async def deliberate(self, question: str, context: dict = None) -> CouncilDecision:
        self.call_log.append({"method": "deliberate", "question": question, "context": context})
        # Return decision with updated question if not set
        if not self._decision.question:
            return CouncilDecision(
                question=question,
                decision=self._decision.decision,
                transcript=self._decision.transcript
            )
        return self._decision


# =============================================================================
# Stub Classes for Standalone Operation
# =============================================================================
# These stubs provide minimal implementations that allow this module to run
# standalone for demonstration. In production, replace with real implementations.
# =============================================================================


class Orchestrator:
    """Stub Orchestrator for standalone demonstration.

    PRODUCTION: Import from ch05-orchestrator/orchestrator.py:
        from ch05_orchestrator.orchestrator import SimpleOrchestrator as Orchestrator

    This stub provides minimal functionality for the hybrid module to work
    standalone. All methods return empty/default values.
    """

    def __init__(self, llm=None, workers=None):
        self.llm = llm
        self.workers = workers or {}

    async def plan(self, task: str) -> dict:
        """Plan task execution. Returns empty dict in stub."""
        return {}

    def create_execution_plan(self, subtasks: dict) -> list:
        """Create execution plan. Returns list of all subtask IDs in stub."""
        return [list(subtasks.keys())] if subtasks else []

    async def execute_subtask(self, subtask: Any, context: dict) -> Any:
        """Execute a subtask. Returns empty dict in stub."""
        return {}

    async def aggregate(self, results: dict, task: str) -> Any:
        """Aggregate results. Returns results unchanged in stub."""
        return results


class Guardian:
    """Stub Guardian for standalone demonstration.

    PRODUCTION: Import from ch08-guardian/guardian.py:
        from ch08_guardian.guardian import SecurityGuardian as Guardian

    This stub approves all validations. Use MockGuardian for testing
    rejection/modification scenarios.
    """

    async def validate_input(self, task: str) -> ValidationResponse:
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def validate_plan(self, subtasks: dict) -> ValidationResponse:
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def validate_results(self, results: dict) -> ValidationResponse:
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def approve_action(self, name: str, arguments: dict) -> ApprovalResponse:
        return ApprovalResponse(approved=True)

    async def validate_action_result(self, name: str, result: Any) -> ValidationResponse:
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def check_question(self, question: str, constraints: list) -> QuestionCheckResponse:
        return QuestionCheckResponse(allowed=True)

    async def validate_decision(self, decision: Any, constraints: list) -> DecisionValidationResponse:
        return DecisionValidationResponse(compliant=True, checks=[])


class SwarmCoordinator:
    """Stub SwarmCoordinator for standalone demonstration.

    PRODUCTION: Import from ch07-swarm/swarm.py:
        from ch07_swarm.swarm import SwarmCoordinator

    This stub returns empty results. Use MockSwarmCoordinator for testing
    with controlled outputs.
    """

    async def run(self, goal: str, max_steps: int = 100) -> SwarmResult:
        return SwarmResult(artifacts=[], exploration_map={})


class Council:
    """Stub Council for standalone demonstration.

    PRODUCTION: Import from ch06-council/council.py:
        from ch06_council.council import Council

    This stub returns empty decisions. Use MockCouncil for testing
    with controlled outputs.
    """

    def __init__(self, councilors=None, llm=None):
        self.councilors = councilors or []
        self.llm = llm

    async def deliberate(self, question: str, context: dict = None) -> CouncilDecision:
        return CouncilDecision(question=question, decision=None, transcript=[])


class Subtask:
    """Subtask representation for orchestrator patterns.

    PRODUCTION: Import from ch05-orchestrator/orchestrator.py or define
    in a shared types module.
    """

    def __init__(self, id: str = "", worker_type: str = "", description: str = ""):
        self.id = id
        self.worker_type = worker_type
        self.description = description


class Artifact:
    """Artifact from swarm exploration.

    PRODUCTION: Define in a shared types module for consistency across
    all pattern implementations.
    """

    def __init__(self, content: str = "", artifact_type: str = "text", metadata: dict = None):
        self.content = content
        self.artifact_type = artifact_type
        self.metadata = metadata or {}


# =============================================================================
# Example Usage and Demo
# =============================================================================

async def main():
    """Demonstration of hybrid architecture."""
    print("=" * 60)
    print("Hybrid Architecture Demonstration")
    print("=" * 60)

    # Note: In production, you would use actual LLM client and pattern implementations
    # This demo shows the structure and flow

    print("\nThis module provides the following hybrid architecture components:")
    print()
    print("1. GuardedOrchestrator - Orchestrator + Guardian combination")
    print("   - Validates inputs, plans, and outputs at checkpoints")
    print("   - Per-action guardian monitoring during execution")
    print()
    print("2. SwarmCouncilPipeline - Swarm + Council combination")
    print("   - Swarm explores to generate options")
    print("   - Council deliberates to choose best option")
    print()
    print("3. OrchestratorSwarmHybrid - Orchestrator + Swarm combination")
    print("   - Structured phases use orchestrator workers")
    print("   - Exploratory phases use swarm")
    print()
    print("4. ConstrainedCouncil - Guardian + Council combination")
    print("   - Council deliberates with constraint-aware prompts")
    print("   - Guardian enforces compliance on decisions")
    print()
    print("5. PatternRouter - Dynamic pattern selection")
    print("   - Analyzes task characteristics")
    print("   - Scores patterns based on characteristics")
    print("   - Routes to best matching pattern")
    print()
    print("6. AdaptivePipeline - Multi-phase adaptive execution")
    print("   - Auto-plans phases from task description")
    print("   - Selects pattern per-phase based on phase characteristics")
    print("   - Aggregates results into final output")
    print()
    print("7. HybridRouter - Production-ready routing")
    print("   - TaskAnalyzer for characteristic extraction")
    print("   - PatternScorer for pattern ranking")
    print("   - Execution hooks for observability")
    print()
    print("8. ProductionAdaptivePipeline - Production pipeline")
    print("   - Timeout handling per phase")
    print("   - Failure recovery policies")
    print("   - Comprehensive result tracking")
    print()
    print("=" * 60)
    print("TaskCharacteristics fields:")
    print("=" * 60)
    tc = TaskCharacteristics()
    for field_name, value in tc.to_dict().items():
        print(f"  - {field_name}: {value}")
    print()
    print("=" * 60)
    print("PatternType enum values:")
    print("=" * 60)
    for pt in PatternType:
        print(f"  - {pt.name}: {pt.value}")


if __name__ == "__main__":
    asyncio.run(main())
