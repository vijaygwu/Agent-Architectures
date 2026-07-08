"""
Chapter 5: The Orchestrator Pattern
====================================
Implementation of the orchestrator pattern for multi-agent coordination.

The orchestrator pattern uses a central agent to:
1. Decompose complex tasks into subtasks
2. Route subtasks to specialized worker agents
3. Coordinate parallel/sequential execution
4. Aggregate results into final output

Two implementations are provided:
- SimpleOrchestrator: No external dependencies, good starting point
- MultiAgentOrchestrator: Uses LangGraph for advanced features (checkpointing, etc.)

To run without LangGraph, use SimpleOrchestrator:
    orchestrator = SimpleOrchestrator()
    result = await orchestrator.run("your task")

To use the full LangGraph version, install: pip install langgraph

NOTE ON EXAMPLES: This module uses asyncio.sleep() to simulate API calls
for demonstration purposes. Production implementations require:
- Actual LLM client calls (see common/utils.py for patterns)
- Retry logic with exponential backoff (see @with_retry decorator)
- Timeout handling (see asyncio.wait_for patterns)
- Circuit breakers for external service failures (see Guardian chapter)
- Structured error responses, not just exceptions

See the Research Assistant chapter (ch12) for a more complete production example.

Production Notes:
- Add retry logic with exponential backoff for worker failures
- Implement circuit breakers for external service calls
- Use structured logging for task tracing across workers
- Consider adding task timeouts and cancellation support
"""

__all__ = [
    "WorkerType",
    "Subtask",
    "OrchestratorState",
    "PersistenceBackend",
    "InMemoryPersistence",
    "RedisPersistence",
    "BaseWorker",
    "ResearchWorker",
    "AnalysisWorker",
    "WritingWorker",
    "CodeWorker",
    "ReviewWorker",
    "TaskPlanner",
    "ResultAggregator",
    "MultiAgentOrchestrator",
    "SimpleOrchestrator",
]

import asyncio
import json
import logging
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import uuid


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries, used for Annotated type reducer."""
    return {**a, **b}

# Import common utilities for structured logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
try:
    from utils import configure_logging, get_tracer
    logger = configure_logging(level="INFO", json_output=True, logger_name="orchestrator")
    tracer = get_tracer("orchestrator")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("orchestrator")
    # No-op tracer fallback
    class _NoOpSpan:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def set_attribute(self, k, v): pass
        def add_event(self, n, a=None): pass
    class _NoOpTracer:
        def start_as_current_span(self, n, **kw): return _NoOpSpan()
    tracer = _NoOpTracer()

# =============================================================================
# Circuit Breaker for Worker Resilience
# =============================================================================
# Each worker type gets its own circuit breaker to isolate failures.
# If the research worker is failing (e.g., search API down), the circuit
# opens and fails fast, but analysis and writing workers continue normally.
#
# This prevents:
# - Wasting resources on workers that are consistently failing
# - Cascading failures across the orchestrator
# - Long timeouts when a service is known to be down
# =============================================================================

try:
    from resilience import CircuitBreaker, CircuitBreakerOpen
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    # Inline fallback for when resilience module is not available
    CIRCUIT_BREAKER_AVAILABLE = False

    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        def __init__(self, breaker_name: str, time_until_retry: float = 30.0):
            self.breaker_name = breaker_name
            self.time_until_retry = time_until_retry
            super().__init__(f"Circuit breaker '{breaker_name}' is OPEN")

    class CircuitBreaker:
        """Minimal inline circuit breaker when resilience module unavailable."""
        def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 30.0, **kwargs):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._state = "closed"
            import time
            self._time = time

        @property
        def state(self):
            if self._state == "open":
                if self._time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "half_open"
            return self._state

        def record_success(self):
            self._failure_count = 0
            self._state = "closed"

        def record_failure(self):
            self._last_failure_time = self._time.time()
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = "open"

        async def __aenter__(self):
            if self.state == "open":
                raise CircuitBreakerOpen(self.name)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.record_success()
            elif not isinstance(exc_val, CircuitBreakerOpen):
                self.record_failure()
            return False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

from typing import Protocol


# =============================================================================
# Persistence Layer
# =============================================================================

class PersistenceBackend(Protocol):
    """Protocol for state persistence backends."""
    async def save_state(self, task_id: str, state: dict) -> None: ...
    async def load_state(self, task_id: str) -> dict | None: ...
    async def delete_state(self, task_id: str) -> None: ...


class InMemoryPersistence:
    """Default in-memory persistence (no durability)."""
    def __init__(self):
        self._store: dict[str, dict] = {}

    async def save_state(self, task_id: str, state: dict) -> None:
        self._store[task_id] = state

    async def load_state(self, task_id: str) -> dict | None:
        return self._store.get(task_id)

    async def delete_state(self, task_id: str) -> None:
        self._store.pop(task_id, None)


# Optional Redis persistence
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None


class RedisPersistence:
    """Redis-backed persistence for production use."""
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required: pip install redis")
        self._redis = aioredis.from_url(redis_url)
        self._prefix = "orchestrator:task:"

    async def save_state(self, task_id: str, state: dict) -> None:
        await self._redis.set(f"{self._prefix}{task_id}", json.dumps(state), ex=86400)

    async def load_state(self, task_id: str) -> dict | None:
        data = await self._redis.get(f"{self._prefix}{task_id}")
        return json.loads(data) if data else None

    async def delete_state(self, task_id: str) -> None:
        await self._redis.delete(f"{self._prefix}{task_id}")


# =============================================================================
# Configuration and Models
# =============================================================================

class WorkerType(str, Enum):
    """Available worker agent types"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODE = "code"
    REVIEW = "review"


@dataclass
class Subtask:
    """A subtask assigned to a worker"""
    id: str
    description: str
    worker_type: WorkerType
    dependencies: list[str] = field(default_factory=list)
    input_data: dict = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "worker_type": self.worker_type.value,
            "dependencies": self.dependencies,
            "input_data": self.input_data,
            "status": self.status,
            "result": self.result,
            "error": self.error
        }


# =============================================================================
# Orchestrator State
# =============================================================================

class OrchestratorState(TypedDict):
    """
    State maintained throughout orchestration.

    The state flows through the graph, accumulating results
    as workers complete their subtasks.
    """
    # Input
    task: str
    context: dict

    # Planning
    subtasks: list[Subtask]
    execution_plan: list[list[str]]  # Ordered groups of subtask IDs

    # Execution
    current_phase: int
    completed_subtasks: Annotated[list[str], operator.add]
    failed_subtasks: Annotated[list[str], operator.add]
    results: Annotated[dict[str, Any], _merge_dicts]

    # Output
    final_output: str | None
    status: str  # planning, executing, aggregating, completed, failed
    error: str | None


# =============================================================================
# Worker Agents
# =============================================================================

class BaseWorker:
    """Base class for worker agents"""

    def __init__(self, worker_type: WorkerType, model_client: Any = None):
        self.worker_type = worker_type
        self.model_client = model_client

    async def execute(self, subtask: Subtask, context: dict) -> Any:
        """Execute the subtask - override in subclasses"""
        raise NotImplementedError


class ResearchWorker(BaseWorker):
    """Worker for research and information gathering tasks"""

    def __init__(self, model_client: Any = None):
        super().__init__(WorkerType.RESEARCH, model_client)

    async def execute(self, subtask: Subtask, context: dict) -> dict:
        """
        Execute research subtask.

        In production, this would:
        1. Use web search tools
        2. Query knowledge bases
        3. Analyze documents
        """
        # DEMO ONLY: Replace with actual LLM call + retry logic in production
        await asyncio.sleep(0.5)  # Simulates API latency

        return {
            "findings": [
                {
                    "topic": subtask.description,
                    "summary": f"Research findings for: {subtask.description}",
                    "sources": ["source_1", "source_2"],
                    "confidence": 0.85
                }
            ],
            "metadata": {
                "search_queries": 3,
                "documents_analyzed": 5,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }


class AnalysisWorker(BaseWorker):
    """Worker for data analysis tasks"""

    def __init__(self, model_client: Any = None):
        super().__init__(WorkerType.ANALYSIS, model_client)

    async def execute(self, subtask: Subtask, context: dict) -> dict:
        """
        Execute analysis subtask.

        In production, this would:
        1. Process data from previous research
        2. Run statistical analysis
        3. Generate insights
        """
        await asyncio.sleep(0.3)

        # Use input from dependencies
        input_data = subtask.input_data

        return {
            "analysis": {
                "key_insights": [
                    f"Insight 1 from analysis of: {subtask.description}",
                    "Insight 2: Pattern detected",
                    "Insight 3: Recommendation identified"
                ],
                "metrics": {
                    "data_points_analyzed": 100,
                    "confidence_score": 0.78
                }
            },
            "visualizations": ["chart_1_url", "chart_2_url"]
        }


class WritingWorker(BaseWorker):
    """Worker for content generation tasks"""

    def __init__(self, model_client: Any = None):
        super().__init__(WorkerType.WRITING, model_client)

    async def execute(self, subtask: Subtask, context: dict) -> dict:
        """
        Execute writing subtask.

        In production, this would:
        1. Use LLM to generate content
        2. Apply style guidelines
        3. Format output appropriately
        """
        await asyncio.sleep(0.4)

        return {
            "content": f"Generated content for: {subtask.description}\n\n"
                       "This is the synthesized output based on research and analysis.",
            "word_count": 250,
            "format": "markdown"
        }


class CodeWorker(BaseWorker):
    """Worker for code generation and execution tasks"""

    def __init__(self, model_client: Any = None):
        super().__init__(WorkerType.CODE, model_client)

    async def execute(self, subtask: Subtask, context: dict) -> dict:
        """
        Execute code generation/execution subtask.

        In production, this would:
        1. Generate code using LLM
        2. Execute in sandbox
        3. Return results
        """
        await asyncio.sleep(0.3)

        return {
            "code": "# Generated code\nprint('Hello, World!')",
            "language": "python",
            "execution_result": "Hello, World!",
            "success": True
        }


class ReviewWorker(BaseWorker):
    """Worker for review and validation tasks"""

    def __init__(self, model_client: Any = None):
        super().__init__(WorkerType.REVIEW, model_client)

    async def execute(self, subtask: Subtask, context: dict) -> dict:
        """
        Execute review subtask.

        In production, this would:
        1. Review content from other workers
        2. Check for errors/issues
        3. Suggest improvements
        """
        await asyncio.sleep(0.2)

        return {
            "review": {
                "approved": True,
                "feedback": [
                    "Content is accurate",
                    "Consider adding more examples"
                ],
                "score": 8.5
            }
        }


# =============================================================================
# Orchestrator Components
# =============================================================================

class TaskPlanner:
    """
    Plans task execution by decomposing into subtasks.

    The planner analyzes the main task and creates a structured
    execution plan with dependencies.
    """

    def __init__(self, model_client: Any = None):
        self.model_client = model_client

    async def create_plan(
        self,
        task: str,
        context: dict
    ) -> tuple[list[Subtask], list[list[str]]]:
        """
        Create execution plan for a task.

        Returns:
            - List of subtasks
            - Execution phases (groups of subtask IDs that can run in parallel)
        """
        # Input validation
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        # In production, use LLM to analyze task and create plan
        # This is a demonstration of the planning logic

        subtasks = []
        execution_plan = []

        # Phase 1: Research (parallel)
        research_1 = Subtask(
            id=f"research_{uuid.uuid4().hex[:8]}",
            description=f"Research background on: {task}",
            worker_type=WorkerType.RESEARCH
        )
        research_2 = Subtask(
            id=f"research_{uuid.uuid4().hex[:8]}",
            description=f"Research current trends related to: {task}",
            worker_type=WorkerType.RESEARCH
        )
        subtasks.extend([research_1, research_2])
        execution_plan.append([research_1.id, research_2.id])

        # Phase 2: Analysis (depends on research)
        analysis = Subtask(
            id=f"analysis_{uuid.uuid4().hex[:8]}",
            description=f"Analyze research findings for: {task}",
            worker_type=WorkerType.ANALYSIS,
            dependencies=[research_1.id, research_2.id]
        )
        subtasks.append(analysis)
        execution_plan.append([analysis.id])

        # Phase 3: Writing (depends on analysis)
        writing = Subtask(
            id=f"writing_{uuid.uuid4().hex[:8]}",
            description=f"Write final report on: {task}",
            worker_type=WorkerType.WRITING,
            dependencies=[analysis.id]
        )
        subtasks.append(writing)
        execution_plan.append([writing.id])

        return subtasks, execution_plan


class ResultAggregator:
    """
    Aggregates results from all workers into final output.

    The aggregator synthesizes individual worker outputs into
    a coherent final response.
    """

    def __init__(self, model_client: Any = None):
        self.model_client = model_client

    async def aggregate(
        self,
        task: str,
        subtasks: list[Subtask],
        results: dict[str, Any]
    ) -> str:
        """
        Aggregate all results into final output.

        In production, use LLM to synthesize results into
        a coherent response addressing the original task.
        """
        # Collect all outputs
        research_findings = []
        analysis_results = []
        written_content = []
        review_feedback = []

        for subtask in subtasks:
            result = results.get(subtask.id)
            if not result:
                continue

            if subtask.worker_type == WorkerType.RESEARCH:
                research_findings.extend(result.get("findings", []))
            elif subtask.worker_type == WorkerType.ANALYSIS:
                analysis_results.append(result.get("analysis", {}))
            elif subtask.worker_type == WorkerType.WRITING:
                written_content.append(result.get("content", ""))
            elif subtask.worker_type == WorkerType.REVIEW:
                review_feedback.append(result.get("review", {}))

        # Synthesize final output
        output_parts = [
            f"# Task: {task}\n",
            "\n## Research Summary\n",
            *[f"- {f.get('summary', '')}\n" for f in research_findings],
            "\n## Analysis\n",
            *[f"Insights: {a.get('key_insights', [])}\n" for a in analysis_results],
            "\n## Content\n",
            *written_content,
            "\n## Review\n",
            *[f"Score: {r.get('score', 'N/A')}, Feedback: {r.get('feedback', [])}\n"
              for r in review_feedback]
        ]

        return "".join(output_parts)


# =============================================================================
# Orchestrator Graph
# =============================================================================

class MultiAgentOrchestrator:
    """
    Main orchestrator coordinating multiple worker agents.

    Uses LangGraph to manage the execution flow with:
    - State management across phases
    - Parallel execution within phases
    - Dependency resolution between phases
    - Error handling and recovery
    """

    def __init__(
        self,
        persistence: PersistenceBackend | None = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_timeout: float = 60.0
    ):
        # Input validation
        if circuit_breaker_threshold <= 0:
            raise ValueError("circuit_breaker_threshold must be positive")
        if circuit_breaker_timeout <= 0:
            raise ValueError("circuit_breaker_timeout must be positive")

        self.planner = TaskPlanner()
        self.aggregator = ResultAggregator()
        self._persistence = persistence or InMemoryPersistence()

        # Initialize workers
        self.workers: dict[WorkerType, BaseWorker] = {
            WorkerType.RESEARCH: ResearchWorker(),
            WorkerType.ANALYSIS: AnalysisWorker(),
            WorkerType.WRITING: WritingWorker(),
            WorkerType.CODE: CodeWorker(),
            WorkerType.REVIEW: ReviewWorker(),
        }

        # Circuit breakers per worker type - isolates failures by worker category
        # If research workers fail repeatedly, analysis workers continue normally
        self._worker_circuit_breakers: dict[WorkerType, CircuitBreaker] = {
            worker_type: CircuitBreaker(
                name=f"worker:{worker_type.value}",
                failure_threshold=circuit_breaker_threshold,
                recovery_timeout=circuit_breaker_timeout
            )
            for worker_type in WorkerType
        }

        # Build the orchestration graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph orchestration graph"""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph is required for MultiAgentOrchestrator. Install with: pip install langgraph")

        # Create graph with state schema
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_phase", self._execute_phase_node)
        graph.add_node("aggregate", self._aggregate_node)

        # Set entry point
        graph.set_entry_point("plan")

        # Add edges
        graph.add_edge("plan", "execute_phase")
        graph.add_conditional_edges(
            "execute_phase",
            self._should_continue,
            {
                "continue": "execute_phase",
                "aggregate": "aggregate"
            }
        )
        graph.add_edge("aggregate", END)

        # Compile with checkpointing
        return graph.compile(checkpointer=MemorySaver())

    async def _plan_node(self, state: OrchestratorState) -> dict:
        """Planning node - decompose task into subtasks"""
        subtasks, execution_plan = await self.planner.create_plan(
            state["task"],
            state["context"]
        )

        return {
            "subtasks": subtasks,
            "execution_plan": execution_plan,
            "current_phase": 0,
            "status": "executing"
        }

    async def _execute_phase_node(self, state: OrchestratorState) -> dict:
        """Execute current phase of subtasks in parallel"""
        current_phase = state["current_phase"]
        execution_plan = state["execution_plan"]

        if current_phase >= len(execution_plan):
            return {"status": "aggregating"}

        # Get subtasks for current phase
        phase_subtask_ids = execution_plan[current_phase]
        subtasks_map = {s.id: s for s in state["subtasks"]}

        # Prepare input data from dependencies
        for subtask_id in phase_subtask_ids:
            subtask = subtasks_map[subtask_id]
            for dep_id in subtask.dependencies:
                if dep_id in state["results"]:
                    subtask.input_data[dep_id] = state["results"][dep_id]

        # Execute subtasks in parallel
        tasks = []
        for subtask_id in phase_subtask_ids:
            subtask = subtasks_map[subtask_id]
            worker = self.workers.get(subtask.worker_type)
            if worker:
                tasks.append(self._execute_subtask(worker, subtask, state["context"]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        phase_results = {}
        completed = []
        failed = []

        for subtask_id, result in zip(phase_subtask_ids, results):
            subtask = subtasks_map[subtask_id]
            if isinstance(result, Exception):
                subtask.status = "failed"
                subtask.error = str(result)
                failed.append(subtask_id)
                logger.warning(
                    f"Subtask {subtask_id} failed; dependent phases "
                    f"will run without its results: {result}"
                )
            else:
                subtask.status = "completed"
                subtask.result = result
                phase_results[subtask_id] = result
                completed.append(subtask_id)

        return {
            "current_phase": current_phase + 1,
            "completed_subtasks": completed,
            "failed_subtasks": failed,
            "results": phase_results
        }

    async def _execute_subtask(
        self,
        worker: BaseWorker,
        subtask: Subtask,
        context: dict,
        max_retries: int = 3,
        timeout_seconds: float = 120.0
    ) -> Any:
        """
        Execute a single subtask with a worker, with circuit breaker, retry logic and timeout.

        The circuit breaker pattern protects against repeatedly failing workers:
        - If a worker type fails repeatedly, the circuit opens
        - Subsequent requests to that worker type fail fast
        - After recovery timeout, the circuit half-opens to test recovery
        - Other worker types continue operating normally (fault isolation)

        Args:
            worker: The worker to execute the subtask
            subtask: The subtask to execute
            context: Context for execution
            max_retries: Maximum number of retry attempts (default: 3)
            timeout_seconds: Timeout per attempt in seconds (default: 120)

        Returns:
            The result from the worker

        Raises:
            CircuitBreakerOpen: If worker's circuit breaker is open
            TimeoutError: If all retry attempts timeout
            Exception: If all retry attempts fail with errors
        """
        # Input validation
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        with tracer.start_as_current_span(f"subtask.{subtask.id}") as span:
            span.set_attribute("worker.type", worker.__class__.__name__)
            span.set_attribute("subtask.id", subtask.id)
            span.set_attribute("subtask.type", subtask.worker_type.value)
            span.set_attribute("max_retries", max_retries)

            # Get the circuit breaker for this worker type
            circuit_breaker = self._worker_circuit_breakers[subtask.worker_type]
            span.set_attribute("circuit_breaker.state", circuit_breaker.state.value if hasattr(circuit_breaker.state, 'value') else str(circuit_breaker.state))

            subtask.status = "in_progress"
            last_error = None

            # Check circuit breaker before attempting work
            # This fails fast if this worker type has been failing
            try:
                async with circuit_breaker:
                    for attempt in range(max_retries):
                        span.add_event(f"attempt.{attempt + 1}")
                        try:
                            result = await asyncio.wait_for(
                                worker.execute(subtask, context),
                                timeout=timeout_seconds
                            )
                            span.set_attribute("status", "success")
                            return result
                        except asyncio.TimeoutError:
                            last_error = TimeoutError(
                                f"Worker {worker.__class__.__name__} timed out on subtask {subtask.id} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                        except Exception as e:
                            last_error = e

                        # Exponential backoff before retry
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)

                    # All retries exhausted - raise the last error
                    # The circuit breaker context manager will record this as a failure
                    raise last_error

            except CircuitBreakerOpen as e:
                # Circuit is open - fail fast without attempting work
                logger.warning(
                    f"Circuit breaker open for worker type {subtask.worker_type.value}",
                    extra={"extra_fields": {
                        "subtask_id": subtask.id,
                        "worker_type": subtask.worker_type.value,
                        "retry_in": e.time_until_retry
                    }}
                )
                span.set_attribute("status", "circuit_breaker_open")
                raise

            span.set_attribute("status", "failed")
            span.set_attribute("error", str(last_error))
            raise last_error

    def _should_continue(self, state: OrchestratorState) -> str:
        """Determine if more phases need execution"""
        if state["current_phase"] >= len(state["execution_plan"]):
            return "aggregate"
        return "continue"

    async def _aggregate_node(self, state: OrchestratorState) -> dict:
        """Aggregate all results into final output"""
        final_output = await self.aggregator.aggregate(
            state["task"],
            state["subtasks"],
            state["results"]
        )

        failed = state.get("failed_subtasks", [])
        if failed and not state["results"]:
            status = "failed"
        elif failed:
            status = "partial"
        else:
            status = "completed"

        return {
            "final_output": final_output,
            "status": status,
            "error": (
                f"{len(failed)} subtask(s) failed: {failed}"
                if failed else None
            )
        }

    async def run(
        self,
        task: str,
        context: dict | None = None,
        thread_id: str | None = None
    ) -> dict:
        """
        Run the orchestrator on a task.

        Args:
            task: The main task to accomplish
            context: Additional context for the task
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final state including output
        """
        # Input validation
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        initial_state: OrchestratorState = {
            "task": task,
            "context": context or {},
            "subtasks": [],
            "execution_plan": [],
            "current_phase": 0,
            "completed_subtasks": [],
            "failed_subtasks": [],
            "results": {},
            "final_output": None,
            "status": "planning",
            "error": None
        }

        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state, config)

        return final_state


# =============================================================================
# Simpler Orchestrator (Without LangGraph)
# =============================================================================

class SimpleOrchestrator:
    """
    Simplified orchestrator without external dependencies.

    Use this as a starting point before adding LangGraph complexity.
    Follows Anthropic's guidance: "many patterns can be implemented
    in a few lines of code."
    """

    def __init__(self, persistence: PersistenceBackend | None = None):
        self._persistence = persistence or InMemoryPersistence()
        self.workers: dict[WorkerType, BaseWorker] = {
            WorkerType.RESEARCH: ResearchWorker(),
            WorkerType.ANALYSIS: AnalysisWorker(),
            WorkerType.WRITING: WritingWorker(),
            WorkerType.CODE: CodeWorker(),
            WorkerType.REVIEW: ReviewWorker(),
        }

    async def run(self, task: str, context: dict | None = None, task_id: str | None = None) -> dict:
        """Run task through orchestration pipeline with optional persistence."""
        # Input validation
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        context = context or {}
        task_id = task_id or str(uuid.uuid4())
        results = {}

        # Try to restore from checkpoint
        saved_state = await self._persistence.load_state(task_id)
        if saved_state:
            results = saved_state.get("results", {})
            logger.info(f"Restored checkpoint for task {task_id}")

        # Phase 1: Research (parallel)
        research_tasks = [
            self.workers[WorkerType.RESEARCH].execute(
                Subtask(id="research_1", description=f"Research: {task}",
                        worker_type=WorkerType.RESEARCH),
                context
            ),
            self.workers[WorkerType.RESEARCH].execute(
                Subtask(id="research_2", description=f"Research trends: {task}",
                        worker_type=WorkerType.RESEARCH),
                context
            )
        ]
        research_results = await asyncio.gather(*research_tasks)
        results["research"] = research_results
        await self._persistence.save_state(task_id, {"results": results, "phase": "research"})

        # Phase 2: Analysis
        analysis_result = await self.workers[WorkerType.ANALYSIS].execute(
            Subtask(id="analysis", description=f"Analyze: {task}",
                    worker_type=WorkerType.ANALYSIS,
                    input_data={"research": research_results}),
            context
        )
        results["analysis"] = analysis_result
        await self._persistence.save_state(task_id, {"results": results, "phase": "analysis"})

        # Phase 3: Writing
        writing_result = await self.workers[WorkerType.WRITING].execute(
            Subtask(id="writing", description=f"Write report: {task}",
                    worker_type=WorkerType.WRITING,
                    input_data={"analysis": analysis_result}),
            context
        )
        results["writing"] = writing_result

        # Clean up checkpoint on successful completion
        await self._persistence.delete_state(task_id)

        return {
            "task": task,
            "task_id": task_id,
            "results": results,
            "final_output": writing_result.get("content", ""),
            "status": "completed"
        }


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate orchestrator usage"""

    print("=" * 60)
    print("Simple Orchestrator Demo")
    print("=" * 60)

    simple_orchestrator = SimpleOrchestrator()
    result = await simple_orchestrator.run(
        task="Analyze the impact of AI agents on enterprise software development",
        context={"industry": "technology", "timeframe": "2024-2026"}
    )

    print(f"\nTask: {result['task']}")
    print(f"Status: {result['status']}")
    print(f"\nFinal Output:\n{result['final_output'][:500]}...")

    if LANGGRAPH_AVAILABLE:
        print("\n" + "=" * 60)
        print("LangGraph Orchestrator Demo")
        print("=" * 60)

        orchestrator = MultiAgentOrchestrator()
        result = await orchestrator.run(
            task="Create a market analysis report on AI agent platforms",
            context={"focus": "enterprise", "competitors": ["Google", "AWS", "Azure"]}
        )

        print(f"\nTask: {result['task']}")
        print(f"Status: {result['status']}")
        print(f"Phases Completed: {result['current_phase']}")
        print(f"Subtasks: {len(result['subtasks'])}")
        print(f"\nFinal Output:\n{result['final_output'][:500]}...")
    else:
        print("\n" + "=" * 60)
        print("LangGraph not available - skipping MultiAgentOrchestrator demo")
        print("Install with: pip install langgraph")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
