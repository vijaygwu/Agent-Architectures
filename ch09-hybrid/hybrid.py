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

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


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

        # Temporarily replace tool executor (with lock for thread safety)
        async with self._execution_lock:
            worker.execute_tool = guarded_tool_execution
            try:
                result = await worker.execute(subtask, context)
            finally:
                worker.execute_tool = original_execute_tool

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
        except json.JSONDecodeError:
            # Fallback to single-phase execution if LLM response isn't valid JSON
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

        data = json.loads(response.content)
        return TaskCharacteristics(**data)

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

        phase_data = json.loads(response.content)
        return [
            PipelinePhase(
                id=p["id"],
                description=p["description"],
                dependencies=p.get("dependencies", [])
            )
            for p in phase_data
        ]

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
        """Execute a single phase with the selected pattern."""

        task_description = f"""
{phase.description}

Context from previous phases:
{json.dumps(context.get('dependencies', {}), indent=2)}
"""

        if pattern_type in [
            PatternType.ORCHESTRATOR,
            PatternType.ORCHESTRATOR_GUARDIAN,
            PatternType.ORCHESTRATOR_SWARM
        ]:
            return await pattern.execute(task_description)

        elif pattern_type in [
            PatternType.COUNCIL,
            PatternType.COUNCIL_GUARDIAN
        ]:
            return await pattern.deliberate(task_description, context)

        elif pattern_type == PatternType.SWARM:
            return await pattern.run(task_description)

        elif pattern_type == PatternType.COUNCIL_SWARM:
            return await pattern.decide(task_description, context)

        else:
            # Generic execution
            return await pattern.execute(task_description)

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

    def __init__(self, llm_client):
        self.llm = llm_client
        self._cache: dict[str, TaskCharacteristics] = {}

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

        self._cache[cache_key] = characteristics
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
        context: dict | None
    ) -> Any:
        """Execute the appropriate method based on pattern type."""

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
        """Execute a single phase."""

        start_time = datetime.now()

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

            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=result.pattern_used or PatternType.ORCHESTRATOR,
                output=result.output,
                execution_time_ms=execution_time,
                success=result.success,
                error=result.error
            )

        except asyncio.TimeoutError:
            return ProductionPhaseResult(
                phase_id=phase.id,
                pattern_used=PatternType.ORCHESTRATOR,
                output=None,
                execution_time_ms=phase.timeout_seconds * 1000,
                success=False,
                error=f"Phase timed out after {phase.timeout_seconds}s"
            )
        except Exception as e:
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
# Stub Classes for Type Hints
# =============================================================================
# IMPORTANT: These are PLACEHOLDER stubs for type hints and testing only.
# In production, import the real implementations:
#
#   from ch05_orchestrator.orchestrator import SimpleOrchestrator, MultiAgentOrchestrator
#   from ch06_council.council import Council
#   from ch07_swarm.swarm import SwarmCoordinator
#   from ch08_guardian.guardian import GuardianPipeline
#
# These stubs allow this module to run standalone for demonstration purposes.
# =============================================================================

class Orchestrator:
    """Stub for Orchestrator pattern.

    PLACEHOLDER: In production, import from ch05-orchestrator/orchestrator.py:
        from ch05_orchestrator.orchestrator import SimpleOrchestrator as Orchestrator
    """
    def __init__(self, llm=None, workers=None):
        self.llm = llm
        self.workers = workers or {}

    async def plan(self, task: str) -> dict:
        """Plan task execution. # Placeholder: Returns empty dict."""
        return {}

    def create_execution_plan(self, subtasks: dict) -> list:
        """Create execution plan. # Placeholder: Returns empty list."""
        return []

    async def execute_subtask(self, subtask: Any, context: dict) -> Any:
        """Execute a subtask. # Placeholder: Returns empty dict."""
        return {}

    async def aggregate(self, results: dict, task: str) -> Any:
        """Aggregate results. # Placeholder: Returns results unchanged."""
        return results


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


class Guardian:
    """Stub for Guardian pattern (from Chapter 8).

    In production, import from ch08-guardian/guardian.py
    """
    async def validate_input(self, task: str) -> ValidationResponse:
        """Validate input. Placeholder: Always approves."""
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def validate_plan(self, subtasks: dict) -> ValidationResponse:
        """Validate plan. Placeholder: Always approves."""
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def validate_results(self, results: dict) -> ValidationResponse:
        """Validate results. Placeholder: Always approves."""
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def approve_action(self, name: str, arguments: dict) -> ApprovalResponse:
        """Approve action. Placeholder: Always approves."""
        return ApprovalResponse(approved=True)

    async def validate_action_result(self, name: str, result: Any) -> ValidationResponse:
        """Validate action result. Placeholder: Always approves."""
        return ValidationResponse(result=ValidationResult.APPROVED)

    async def check_question(self, question: str, constraints: list) -> QuestionCheckResponse:
        """Check question. Placeholder: Always allows."""
        return QuestionCheckResponse(allowed=True)

    async def validate_decision(self, decision: Any, constraints: list) -> DecisionValidationResponse:
        """Validate decision. Placeholder: Always compliant."""
        return DecisionValidationResponse(compliant=True, checks=[])


class SwarmCoordinator:
    """Stub for Swarm pattern (from Chapter 7).

    # Placeholder: In production, import from ch07-swarm/swarm.py
    """
    async def run(self, goal: str, max_steps: int = 100) -> Any:
        """Run swarm exploration. # Placeholder: Returns empty results."""
        return type('obj', (object,), {
            'artifacts': [],
            'exploration_map': {}
        })()


class Council:
    """Stub for Council pattern (from Chapter 6).

    # Placeholder: In production, import from ch06-council/council.py
    """
    def __init__(self, councilors=None, llm=None):
        self.councilors = councilors or []
        self.llm = llm

    async def deliberate(self, question: str, context: dict = None) -> Any:
        """Deliberate on question. # Placeholder: Returns empty decision."""
        return type('obj', (object,), {
            'decision': None,
            'transcript': []
        })()


class Subtask:
    """Stub for Subtask.

    # Placeholder: In production, import from ch05-orchestrator/orchestrator.py
    """
    def __init__(self, id: str = "", worker_type: str = ""):
        self.id = id
        self.worker_type = worker_type


class Artifact:
    """Stub for Artifact.

    # Placeholder: In production, define in shared types module
    """
    def __init__(self, content: str = ""):
        self.content = content


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
