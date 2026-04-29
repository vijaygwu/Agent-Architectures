"""
Chapter 5: The Orchestrator Pattern
====================================
Implementation of the orchestrator pattern for multi-agent coordination.

The orchestrator pattern uses a central agent to:
1. Decompose complex tasks into subtasks
2. Route subtasks to specialized worker agents
3. Coordinate parallel/sequential execution
4. Aggregate results into final output

This implementation uses LangGraph for the orchestration graph.
"""

import asyncio
import json
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


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
    results: Annotated[dict[str, Any], lambda a, b: {**a, **b}]

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
        # Simulate research with structured output
        await asyncio.sleep(0.5)  # Simulate API call

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

        # Phase 4: Review (depends on writing)
        review = Subtask(
            id=f"review_{uuid.uuid4().hex[:8]}",
            description="Review and validate final output",
            worker_type=WorkerType.REVIEW,
            dependencies=[writing.id]
        )
        subtasks.append(review)
        execution_plan.append([review.id])

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
            *[f"Key insights: {a.get('key_insights', [])}\n" for a in analysis_results],
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

    def __init__(self):
        self.planner = TaskPlanner()
        self.aggregator = ResultAggregator()

        # Initialize workers
        self.workers: dict[WorkerType, BaseWorker] = {
            WorkerType.RESEARCH: ResearchWorker(),
            WorkerType.ANALYSIS: AnalysisWorker(),
            WorkerType.WRITING: WritingWorker(),
            WorkerType.CODE: CodeWorker(),
            WorkerType.REVIEW: ReviewWorker(),
        }

        # Build the orchestration graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph orchestration graph"""

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

        for subtask_id, result in zip(phase_subtask_ids, results):
            subtask = subtasks_map[subtask_id]
            if isinstance(result, Exception):
                subtask.status = "failed"
                subtask.error = str(result)
            else:
                subtask.status = "completed"
                subtask.result = result
                phase_results[subtask_id] = result
                completed.append(subtask_id)

        return {
            "current_phase": current_phase + 1,
            "completed_subtasks": completed,
            "results": phase_results
        }

    async def _execute_subtask(
        self,
        worker: BaseWorker,
        subtask: Subtask,
        context: dict
    ) -> Any:
        """Execute a single subtask with a worker"""
        subtask.status = "in_progress"
        return await worker.execute(subtask, context)

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

        return {
            "final_output": final_output,
            "status": "completed"
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
        initial_state: OrchestratorState = {
            "task": task,
            "context": context or {},
            "subtasks": [],
            "execution_plan": [],
            "current_phase": 0,
            "completed_subtasks": [],
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

    def __init__(self):
        self.workers: dict[WorkerType, BaseWorker] = {
            WorkerType.RESEARCH: ResearchWorker(),
            WorkerType.ANALYSIS: AnalysisWorker(),
            WorkerType.WRITING: WritingWorker(),
            WorkerType.CODE: CodeWorker(),
            WorkerType.REVIEW: ReviewWorker(),
        }

    async def run(self, task: str, context: dict | None = None) -> dict:
        """
        Run task through orchestration pipeline.

        This simple version uses a fixed pipeline:
        Research -> Analysis -> Writing -> Review
        """
        context = context or {}
        results = {}

        # Phase 1: Research (parallel)
        research_tasks = [
            self.workers[WorkerType.RESEARCH].execute(
                Subtask(
                    id="research_1",
                    description=f"Research: {task}",
                    worker_type=WorkerType.RESEARCH
                ),
                context
            ),
            self.workers[WorkerType.RESEARCH].execute(
                Subtask(
                    id="research_2",
                    description=f"Research trends: {task}",
                    worker_type=WorkerType.RESEARCH
                ),
                context
            )
        ]
        research_results = await asyncio.gather(*research_tasks)
        results["research"] = research_results

        # Phase 2: Analysis
        analysis_result = await self.workers[WorkerType.ANALYSIS].execute(
            Subtask(
                id="analysis",
                description=f"Analyze: {task}",
                worker_type=WorkerType.ANALYSIS,
                input_data={"research": research_results}
            ),
            context
        )
        results["analysis"] = analysis_result

        # Phase 3: Writing
        writing_result = await self.workers[WorkerType.WRITING].execute(
            Subtask(
                id="writing",
                description=f"Write report: {task}",
                worker_type=WorkerType.WRITING,
                input_data={"analysis": analysis_result}
            ),
            context
        )
        results["writing"] = writing_result

        # Phase 4: Review
        review_result = await self.workers[WorkerType.REVIEW].execute(
            Subtask(
                id="review",
                description="Review output",
                worker_type=WorkerType.REVIEW,
                input_data={"writing": writing_result}
            ),
            context
        )
        results["review"] = review_result

        return {
            "task": task,
            "results": results,
            "final_output": writing_result.get("content", ""),
            "review_score": review_result.get("review", {}).get("score", 0),
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
    print(f"Review Score: {result['review_score']}")
    print(f"\nFinal Output:\n{result['final_output'][:500]}...")

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


if __name__ == "__main__":
    asyncio.run(main())
