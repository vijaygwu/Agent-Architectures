"""
Complete Example: Multi-Agent Research Assistant
================================================

A demonstration research assistant showing:
- Multi-agent collaboration for complex research tasks
- Swarm-based parallel information gathering
- Council-based synthesis and validation
- Source attribution and citation management
- Quality control and fact-checking

Note: This example uses simulated data sources for demonstration.
For production use, integrate with real APIs and search services.

Handles research queries from planning through final report.
"""

__all__ = [
    "cosine_similarity",
    "PercentileHistogram",
    "SourceType",
    "SourceCredibility",
    "ResearchStatus",
    "Source",
    "Fact",
    "ResearchSection",
    "ResearchQuery",
    "ResearchProject",
    "ResearchAgent",
    "PlannerAgent",
    "GathererAgent",
    "AnalyzerAgent",
    "SynthesizerAgent",
    "ValidatorAgent",
    "ReportGenerator",
    "ResearchOrchestrator",
]

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import defaultdict
import hashlib
import sys
from pathlib import Path

# Import structured logging from common utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
try:
    from utils import configure_logging
    logger = configure_logging(level="INFO", json_output=True, logger_name="research_agents")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("research_agents")

# Import metrics collection for duration tracking
try:
    from metrics import MetricsCollector, timed_block
except ImportError:
    # Fallback: define minimal stubs if metrics module unavailable
    class MetricsCollector:
        def __init__(self, namespace: str = "", default_labels: dict = None):
            self.namespace = namespace
        def observe(self, name: str, value: float, labels: dict = None):
            pass
        def export_json(self) -> dict:
            return {}

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def timed_block(metrics, metric_name: str, labels: dict = None):
        yield


# =============================================================================
# Vector Similarity Utilities
# =============================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy for performance when available, falls back to pure Python.
    """
    try:
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    except ImportError:
        # Fallback for environments without numpy
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# SLO Metrics Support
# =============================================================================

@dataclass
class PercentileHistogram:
    """Histogram with percentile calculations for SLO monitoring.

    Uses reservoir sampling to maintain bounded memory while
    providing accurate percentile estimates.
    """
    values: list = field(default_factory=list)
    max_samples: int = 10000

    def observe(self, value: float) -> None:
        if len(self.values) < self.max_samples:
            self.values.append(value)
        else:
            # Reservoir sampling
            idx = random.randint(0, len(self.values))
            if idx < self.max_samples:
                self.values[idx] = value

    def percentile(self, p: float) -> float | None:
        """Calculate percentile (0-100)."""
        if not self.values:
            return None
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def slo_summary(self) -> dict:
        """Return p50, p95, p99 for SLO dashboards."""
        return {
            "p50_ms": round(self.percentile(50) * 1000, 2) if self.percentile(50) else None,
            "p95_ms": round(self.percentile(95) * 1000, 2) if self.percentile(95) else None,
            "p99_ms": round(self.percentile(99) * 1000, 2) if self.percentile(99) else None,
            "count": len(self.values),
        }


# =============================================================================
# Domain Models
# =============================================================================

class SourceType(Enum):
    """Types of research sources."""
    ACADEMIC = "academic"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    OFFICIAL = "official"
    UNKNOWN = "unknown"


class SourceCredibility(Enum):
    """Credibility levels for sources."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class ResearchStatus(Enum):
    """Status of a research task."""
    PLANNING = "planning"
    GATHERING = "gathering"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Source:
    """A research source."""
    id: str
    url: str
    title: str
    source_type: SourceType
    credibility: SourceCredibility = SourceCredibility.UNVERIFIED
    content_summary: str = ""
    retrieved_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Generate citation string."""
        return f"[{self.id}] {self.title} ({self.url})"


@dataclass
class Fact:
    """A fact extracted from research."""
    id: str
    statement: str
    sources: list[str]  # Source IDs
    confidence: float = 0.0
    verified: bool = False
    contradicted_by: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class ResearchSection:
    """A section of the research report."""
    title: str
    content: str
    facts: list[str] = field(default_factory=list)  # Fact IDs
    sources: list[str] = field(default_factory=list)  # Source IDs


@dataclass
class ResearchQuery:
    """A research query to be processed."""
    id: str
    query: str
    scope: str = "comprehensive"  # quick, standard, comprehensive
    required_sources: int = 5
    max_depth: int = 2
    focus_areas: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)


@dataclass
class ResearchProject:
    """A complete research project."""
    id: str
    query: ResearchQuery
    status: ResearchStatus = ResearchStatus.PLANNING
    plan: dict = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)
    sections: list[ResearchSection] = field(default_factory=list)
    final_report: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Agent Definitions
# =============================================================================

class ResearchAgent:
    """Base class for research agents."""

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.tasks_completed = 0

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Process a research task. Override in subclasses."""
        raise NotImplementedError


class PlannerAgent(ResearchAgent):
    """
    Plans research strategy and decomposes queries.
    """

    def __init__(self):
        super().__init__("planner-agent", "Research Planner")

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Create research plan."""
        self.tasks_completed += 1
        logger.info(
            "Creating research plan",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "query": project.query.query}}
        )

        query = project.query

        # Decompose query into sub-questions
        sub_questions = self._decompose_query(query.query)

        # Identify required source types
        source_types = self._identify_source_types(query)

        # Create search strategies
        search_strategies = []
        for sq in sub_questions:
            for st in source_types:
                search_strategies.append({
                    "sub_question": sq,
                    "source_type": st.value,
                    "priority": self._calculate_priority(sq, st)
                })

        # Sort by priority
        search_strategies.sort(key=lambda x: x["priority"], reverse=True)

        # Estimate time
        estimated_time = len(search_strategies) * 2  # 2 seconds per search

        plan = {
            "sub_questions": sub_questions,
            "source_types": [st.value for st in source_types],
            "search_strategies": search_strategies[:query.required_sources * 2],
            "estimated_sources": query.required_sources,
            "estimated_time_seconds": estimated_time,
            "outline": self._create_outline(sub_questions)
        }

        project.plan = plan
        project.status = ResearchStatus.GATHERING

        return plan

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose query into sub-questions."""
        # Simplified decomposition
        sub_questions = [query]  # Main query

        # Generate related questions
        prefixes = [
            "What is the history of",
            "What are the current trends in",
            "What are the key challenges of",
            "What are the benefits of",
            "What are the future prospects of"
        ]

        # Extract main topic (simplified)
        words = query.lower().split()
        topic = " ".join(words[-3:]) if len(words) > 3 else query

        for prefix in prefixes[:3]:  # Limit to 3 sub-questions
            sub_questions.append(f"{prefix} {topic}?")

        return sub_questions

    def _identify_source_types(self, query: ResearchQuery) -> list[SourceType]:
        """Identify relevant source types."""
        types = [SourceType.DOCUMENTATION, SourceType.NEWS]

        if query.scope == "comprehensive":
            types.append(SourceType.ACADEMIC)
            types.append(SourceType.OFFICIAL)

        if "technical" in query.query.lower():
            types.append(SourceType.DOCUMENTATION)

        return list(set(types))

    def _calculate_priority(self, question: str, source_type: SourceType) -> float:
        """Calculate search priority."""
        priority = 0.5

        # Academic sources for complex questions
        if source_type == SourceType.ACADEMIC:
            priority += 0.2

        # News for recent topics
        if "recent" in question.lower() or "current" in question.lower():
            if source_type == SourceType.NEWS:
                priority += 0.3

        return priority

    def _create_outline(self, sub_questions: list[str]) -> list[dict]:
        """Create report outline."""
        outline = [
            {"section": "Executive Summary", "questions": []},
            {"section": "Background", "questions": [sub_questions[0]]},
        ]

        if len(sub_questions) > 1:
            outline.append({
                "section": "Analysis",
                "questions": sub_questions[1:3]
            })

        outline.append({"section": "Conclusions", "questions": []})

        return outline


class GathererAgent(ResearchAgent):
    """
    Gathers information from sources.
    Works in a swarm pattern for parallel collection.
    """

    def __init__(self):
        super().__init__("gatherer-agent", "Information Gatherer")
        self.simulated_sources = self._create_simulated_sources()

    async def process(self, project: ResearchProject, context: dict,
                       per_source_timeout: float = 30.0) -> dict:
        """Gather information for research.

        Args:
            project: The research project to gather information for.
            context: Additional context (unused).
            per_source_timeout: Timeout per source in seconds (default: 30.0).
        """
        # Input validation
        if per_source_timeout is not None and per_source_timeout <= 0:
            raise ValueError("Timeout must be positive")

        self.tasks_completed += 1
        logger.info(
            "Gathering information from sources",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "strategies_count": len(project.plan.get("search_strategies", []))}}
        )

        plan = project.plan
        strategies = plan.get("search_strategies", [])

        # Parallel gathering with per-task timeout
        async def gather_with_timeout(strategy: dict) -> Optional[Source]:
            try:
                return await asyncio.wait_for(
                    self._gather_for_strategy(strategy),
                    timeout=per_source_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout gathering source for strategy: {strategy.get('sub_question', 'unknown')}"
                )
                return None

        gather_tasks = [gather_with_timeout(s) for s in strategies]
        results = await asyncio.gather(*gather_tasks)

        # Process results
        sources_found = []
        for result in results:
            if result:
                source = result
                project.sources[source.id] = source
                sources_found.append(source.id)

        project.status = ResearchStatus.ANALYZING

        return {
            "sources_gathered": len(sources_found),
            "source_ids": sources_found,
            "by_type": self._count_by_type(project.sources)
        }

    async def _gather_for_strategy(self, strategy: dict) -> Optional[Source]:
        """Gather sources for a single strategy."""
        await asyncio.sleep(0.1)  # Simulate network latency

        # Find matching simulated source
        source_type = SourceType(strategy["source_type"])
        for source in self.simulated_sources:
            if source.source_type == source_type:
                # Return a copy with unique ID
                return Source(
                    id=f"src_{uuid.uuid4().hex[:8]}",
                    url=source.url,
                    title=source.title,
                    source_type=source.source_type,
                    credibility=source.credibility,
                    content_summary=source.content_summary
                )

        return None

    def _create_simulated_sources(self) -> list[Source]:
        """Create simulated sources for demo."""
        return [
            Source("s1", "https://example.com/academic/paper1",
                   "Academic Study on AI Agents",
                   SourceType.ACADEMIC, SourceCredibility.HIGH,
                   "This paper examines the effectiveness of multi-agent systems..."),
            Source("s2", "https://news.example.com/ai-trends",
                   "Latest Trends in AI Development",
                   SourceType.NEWS, SourceCredibility.MEDIUM,
                   "Recent developments in AI show significant progress..."),
            Source("s3", "https://docs.example.com/agents",
                   "Agent Framework Documentation",
                   SourceType.DOCUMENTATION, SourceCredibility.HIGH,
                   "This documentation covers the implementation of agents..."),
            Source("s4", "https://official.gov/ai-policy",
                   "Official AI Policy Guidelines",
                   SourceType.OFFICIAL, SourceCredibility.HIGH,
                   "Government guidelines for responsible AI development..."),
            Source("s5", "https://forum.example.com/discussion",
                   "Community Discussion on AI Agents",
                   SourceType.FORUM, SourceCredibility.LOW,
                   "Users discuss their experiences with AI agents..."),
        ]

    def _count_by_type(self, sources: dict[str, Source]) -> dict:
        """Count sources by type."""
        counts = defaultdict(int)
        for source in sources.values():
            counts[source.source_type.value] += 1
        return dict(counts)


class AnalyzerAgent(ResearchAgent):
    """
    Analyzes gathered information and extracts facts.
    """

    def __init__(self):
        super().__init__("analyzer-agent", "Content Analyzer")

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Analyze sources and extract facts."""
        self.tasks_completed += 1
        logger.info(
            "Analyzing sources and extracting facts",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "sources_count": len(project.sources)}}
        )

        facts_extracted = []

        # Analyze each source
        for source_id, source in project.sources.items():
            source_facts = await self._extract_facts(source)
            for fact in source_facts:
                fact.sources = [source_id]
                project.facts[fact.id] = fact
                facts_extracted.append(fact.id)

        # Cross-reference facts
        contradictions = await self._find_contradictions(project.facts)

        # Calculate confidence scores
        await self._calculate_confidence(project.facts)

        project.status = ResearchStatus.SYNTHESIZING

        return {
            "facts_extracted": len(facts_extracted),
            "fact_ids": facts_extracted,
            "contradictions_found": len(contradictions),
            "avg_confidence": (sum(f.confidence for f in project.facts.values()) /
                            len(project.facts)) if project.facts else 0
        }

    async def _extract_facts(self, source: Source) -> list[Fact]:
        """Extract facts from a source."""
        # Simulated fact extraction
        facts = []

        # Generate 2-3 facts per source
        base_statements = [
            f"According to {source.title}, AI agents show promise in automation.",
            f"{source.title} reports that multi-agent systems improve efficiency.",
            f"Research from {source.title} indicates growing adoption of AI agents."
        ]

        for i, stmt in enumerate(base_statements[:2]):
            facts.append(Fact(
                id=f"fact_{uuid.uuid4().hex[:8]}",
                statement=stmt,
                sources=[source.id],
                confidence=0.7 if source.credibility == SourceCredibility.HIGH else 0.5,
                category="finding"
            ))

        return facts

    async def _find_contradictions(self, facts: dict[str, Fact]) -> list[tuple[str, str]]:
        """Find contradicting facts."""
        # Simplified: no contradictions in demo
        return []

    async def _calculate_confidence(self, facts: dict[str, Fact]):
        """Calculate confidence scores for facts."""
        for fact in facts.values():
            # Base confidence from source credibility
            base = fact.confidence

            # Boost for multiple sources
            source_count = len(fact.sources)
            if source_count > 1:
                base = min(0.95, base + 0.1 * (source_count - 1))

            # Penalty for contradictions
            if fact.contradicted_by:
                base = max(0.1, base - 0.2 * len(fact.contradicted_by))

            fact.confidence = base


class SynthesizerAgent(ResearchAgent):
    """
    Synthesizes analyzed information into coherent sections.
    """

    def __init__(self):
        super().__init__("synthesizer-agent", "Content Synthesizer")

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Synthesize research into report sections."""
        self.tasks_completed += 1
        logger.info(
            "Synthesizing research into report sections",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "facts_count": len(project.facts)}}
        )

        outline = project.plan.get("outline", [])
        sections = []

        for outline_section in outline:
            section = await self._create_section(
                outline_section,
                project.facts,
                project.sources
            )
            sections.append(section)

        project.sections = sections
        project.status = ResearchStatus.VALIDATING

        return {
            "sections_created": len(sections),
            "section_titles": [s.title for s in sections],
            "total_facts_used": sum(len(s.facts) for s in sections),
            "total_sources_cited": len(set(
                src for s in sections for src in s.sources
            ))
        }

    async def _create_section(self, outline: dict,
                               facts: dict[str, Fact],
                               sources: dict[str, Source]) -> ResearchSection:
        """Create a report section."""
        title = outline["section"]
        questions = outline.get("questions", [])

        # Select relevant facts
        relevant_facts = list(facts.keys())[:3]  # Simplified

        # Build content
        content_parts = [f"## {title}\n"]

        if questions:
            content_parts.append(f"This section addresses: {', '.join(questions)}\n")

        for fact_id in relevant_facts:
            fact = facts.get(fact_id)
            if fact:
                content_parts.append(f"- {fact.statement} (Confidence: {fact.confidence:.0%})")

        # Collect sources
        section_sources = []
        for fact_id in relevant_facts:
            fact = facts.get(fact_id)
            if fact:
                section_sources.extend(fact.sources)

        return ResearchSection(
            title=title,
            content="\n".join(content_parts),
            facts=relevant_facts,
            sources=list(set(section_sources))
        )


class ValidatorAgent(ResearchAgent):
    """
    Validates research quality and completeness.
    Implements council-based review.
    """

    def __init__(self):
        super().__init__("validator-agent", "Quality Validator")
        self.quality_criteria = [
            ("source_diversity", "Diverse source types"),
            ("fact_confidence", "Adequate fact confidence"),
            ("coverage", "Query coverage"),
            ("citations", "Proper citations")
        ]

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Validate research quality."""
        self.tasks_completed += 1
        logger.info(
            "Validating research quality",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "sections_count": len(project.sections)}}
        )

        validation_results = {}

        # Check each criterion
        for criterion_id, description in self.quality_criteria:
            passed, score, notes = await self._check_criterion(
                criterion_id, project
            )
            validation_results[criterion_id] = {
                "passed": passed,
                "score": score,
                "description": description,
                "notes": notes
            }

        # Overall assessment
        passed_count = sum(1 for r in validation_results.values() if r["passed"])
        overall_score = sum(r["score"] for r in validation_results.values()) / len(validation_results)
        overall_passed = passed_count >= len(self.quality_criteria) * 0.75

        if overall_passed:
            project.status = ResearchStatus.COMPLETE
            project.completed_at = time.time()
        else:
            project.status = ResearchStatus.FAILED

        return {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "criteria_passed": passed_count,
            "criteria_total": len(self.quality_criteria),
            "details": validation_results
        }

    async def _check_criterion(self, criterion_id: str,
                                project: ResearchProject) -> tuple[bool, float, str]:
        """Check a single quality criterion."""
        if criterion_id == "source_diversity":
            types = set(s.source_type for s in project.sources.values())
            score = len(types) / 4  # Expecting 4 types
            return len(types) >= 2, score, f"{len(types)} source types found"

        elif criterion_id == "fact_confidence":
            if not project.facts:
                return False, 0, "No facts extracted"
            avg_conf = sum(f.confidence for f in project.facts.values()) / len(project.facts)
            return avg_conf >= 0.6, avg_conf, f"Average confidence: {avg_conf:.0%}"

        elif criterion_id == "coverage":
            questions = project.plan.get("sub_questions", [])
            sections = len(project.sections)
            score = min(1.0, sections / max(len(questions), 1))
            return sections >= len(questions) * 0.5, score, f"{sections} sections for {len(questions)} questions"

        elif criterion_id == "citations":
            all_sources = set()
            for section in project.sections:
                all_sources.update(section.sources)
            score = len(all_sources) / max(len(project.sources), 1)
            return len(all_sources) > 0, score, f"{len(all_sources)} sources cited"

        return False, 0, "Unknown criterion"


class ReportGenerator(ResearchAgent):
    """
    Generates the final research report.
    """

    def __init__(self):
        super().__init__("report-generator", "Report Generator")

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Generate final report."""
        self.tasks_completed += 1
        logger.info(
            "Generating final report",
            extra={"extra_fields": {"task_id": project.id, "agent_name": self.name, "sources_count": len(project.sources)}}
        )

        report_parts = []

        # Title
        report_parts.append(f"# Research Report: {project.query.query}\n")
        report_parts.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*\n")

        # Sections
        for section in project.sections:
            report_parts.append(section.content)
            report_parts.append("")

        # References
        report_parts.append("## References\n")
        for source_id, source in project.sources.items():
            report_parts.append(source.citation())

        # Methodology note
        report_parts.append("\n## Methodology\n")
        report_parts.append(f"- Sources analyzed: {len(project.sources)}")
        report_parts.append(f"- Facts extracted: {len(project.facts)}")
        report_parts.append(f"- Research scope: {project.query.scope}")

        project.final_report = "\n".join(report_parts)

        return {
            "report_length": len(project.final_report),
            "sections": len(project.sections),
            "sources_cited": len(project.sources),
            "facts_included": len(project.facts)
        }


# =============================================================================
# Orchestrator
# =============================================================================

class ResearchOrchestrator:
    """
    Orchestrates the multi-agent research workflow.

    Includes configurable timeouts for each phase to prevent indefinite hangs.
    """

    # Default timeouts for each phase (in seconds)
    DEFAULT_TIMEOUTS = {
        "planning": 60.0,
        "gathering": 120.0,
        "analysis": 120.0,
        "synthesis": 90.0,
        "validation": 60.0,
        "report": 60.0,
    }

    def __init__(self, timeouts: dict[str, float] | None = None):
        """Initialize the research orchestrator.

        Args:
            timeouts: Optional dict of phase timeouts in seconds.
                      Keys: planning, gathering, analysis, synthesis, validation, report.
        """
        # Input validation for timeouts
        if timeouts is not None:
            for phase, timeout in timeouts.items():
                if timeout is not None and timeout <= 0:
                    raise ValueError(f"Timeout for phase '{phase}' must be positive")

        self.planner = PlannerAgent()
        self.gatherer = GathererAgent()
        self.analyzer = AnalyzerAgent()
        self.synthesizer = SynthesizerAgent()
        self.validator = ValidatorAgent()
        self.reporter = ReportGenerator()

        self.projects: dict[str, ResearchProject] = {}
        self._max_projects = 1000
        self.metrics = defaultdict(int)
        self.timeouts = {**self.DEFAULT_TIMEOUTS, **(timeouts or {})}

        # Duration metrics collector for research phases
        self.duration_metrics = MetricsCollector(namespace="research")

        # Percentile histograms for SLO monitoring
        self.phase_histograms: dict[str, PercentileHistogram] = {
            phase: PercentileHistogram() for phase in self.DEFAULT_TIMEOUTS
        }

    async def research(self, query: ResearchQuery) -> ResearchProject:
        """Execute complete research workflow."""
        # Input validation for query
        if not query.query or len(query.query.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if len(query.query) > 10000:
            raise ValueError("Query too long (max 10000 characters)")

        # Input validation for required_sources
        if query.required_sources <= 0:
            raise ValueError("required_sources must be positive")
        if query.required_sources > 100:
            raise ValueError("required_sources too large (max 100)")

        # Input validation for max_depth
        if query.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if query.max_depth > 100:
            raise ValueError("max_depth too large (max 100)")

        # Create project
        project = ResearchProject(
            # uuid suffix: second-resolution timestamps collide when
            # two research() calls start in the same second
            id=f"proj_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            query=query
        )
        logger.info(
            "Starting research project",
            extra={"extra_fields": {"task_id": project.id, "query": query.query, "scope": query.scope}}
        )
        # Cleanup completed projects if at capacity
        if len(self.projects) >= self._max_projects:
            completed = [pid for pid, p in self.projects.items()
                        if p.status in (ResearchStatus.COMPLETE, ResearchStatus.FAILED)]
            for pid in completed[:100]:
                del self.projects[pid]
        self.projects[project.id] = project
        self.metrics["projects_started"] += 1

        print(f"\n[RESEARCH] Starting project {project.id}")
        print(f"  Query: {query.query}")
        print(f"  Scope: {query.scope}")

        try:
            # Phase 1: Planning (with timeout)
            print("\n[PHASE 1] Planning research strategy...")
            planning_start = time.time()
            async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "planning"}):
                plan_result = await asyncio.wait_for(
                    self.planner.process(project, {}),
                    timeout=self.timeouts["planning"]
                )
            self.phase_histograms["planning"].observe(time.time() - planning_start)
            print(f"  Sub-questions: {len(plan_result['sub_questions'])}")
            print(f"  Search strategies: {len(plan_result['search_strategies'])}")

            # Phase 2: Gathering (with timeout)
            print("\n[PHASE 2] Gathering information...")
            gathering_start = time.time()
            async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "gathering"}):
                gather_result = await asyncio.wait_for(
                    self.gatherer.process(project, {}),
                    timeout=self.timeouts["gathering"]
                )
            self.phase_histograms["gathering"].observe(time.time() - gathering_start)
            print(f"  Sources gathered: {gather_result['sources_gathered']}")
            print(f"  By type: {gather_result['by_type']}")

            # Phase 3: Analysis (with timeout)
            print("\n[PHASE 3] Analyzing content...")
            analysis_start = time.time()
            async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "analysis"}):
                analyze_result = await asyncio.wait_for(
                    self.analyzer.process(project, {}),
                    timeout=self.timeouts["analysis"]
                )
            self.phase_histograms["analysis"].observe(time.time() - analysis_start)
            print(f"  Facts extracted: {analyze_result['facts_extracted']}")
            print(f"  Average confidence: {analyze_result['avg_confidence']:.0%}")

            # Phase 4: Synthesis (with timeout)
            print("\n[PHASE 4] Synthesizing findings...")
            synthesis_start = time.time()
            async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "synthesis"}):
                synth_result = await asyncio.wait_for(
                    self.synthesizer.process(project, {}),
                    timeout=self.timeouts["synthesis"]
                )
            self.phase_histograms["synthesis"].observe(time.time() - synthesis_start)
            print(f"  Sections created: {synth_result['sections_created']}")

            # Phase 5: Validation (with timeout)
            print("\n[PHASE 5] Validating quality...")
            validation_start = time.time()
            async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "validation"}):
                valid_result = await asyncio.wait_for(
                    self.validator.process(project, {}),
                    timeout=self.timeouts["validation"]
                )
            self.phase_histograms["validation"].observe(time.time() - validation_start)
            print(f"  Validation passed: {valid_result['overall_passed']}")
            print(f"  Score: {valid_result['overall_score']:.0%}")

            # Phase 6: Report Generation (with timeout)
            if valid_result['overall_passed']:
                print("\n[PHASE 6] Generating report...")
                report_start = time.time()
                async with timed_block(self.duration_metrics, "phase_duration_seconds", {"phase": "report"}):
                    report_result = await asyncio.wait_for(
                        self.reporter.process(project, {}),
                        timeout=self.timeouts["report"]
                    )
                self.phase_histograms["report"].observe(time.time() - report_start)
                print(f"  Report length: {report_result['report_length']} chars")
                self.metrics["projects_completed"] += 1
                logger.info(
                    "Research project completed successfully",
                    extra={"extra_fields": {"task_id": project.id, "report_length": report_result['report_length']}}
                )
            else:
                print("\n[PHASE 6] Skipped - validation failed")
                self.metrics["projects_failed"] += 1
                logger.warning(
                    "Research project validation failed",
                    extra={"extra_fields": {"task_id": project.id, "validation_score": valid_result['overall_score']}}
                )

        except asyncio.TimeoutError as e:
            project.status = ResearchStatus.FAILED
            self.metrics["projects_failed"] += 1
            self.metrics["timeout_errors"] += 1
            logger.error(
                "Research project timed out",
                extra={"extra_fields": {"task_id": project.id, "error": str(e)}}
            )
            print(f"\n[ERROR] Project timed out: {e}")

        return project

    def get_metrics(self) -> dict:
        """Get orchestrator metrics."""
        return {
            **dict(self.metrics),
            "agents": {
                agent.name: agent.tasks_completed
                for agent in [
                    self.planner, self.gatherer, self.analyzer,
                    self.synthesizer, self.validator, self.reporter
                ]
            },
            "phase_durations": self.duration_metrics.export_json()
        }

    def get_slo_metrics(self) -> dict:
        """Return SLO metrics (p50, p95, p99) for each research phase."""
        return {
            phase: hist.slo_summary()
            for phase, hist in self.phase_histograms.items()
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of research assistant."""
    print("=" * 60)
    print("Multi-Agent Research Assistant")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = ResearchOrchestrator()

    # Create research query
    query = ResearchQuery(
        id="query-001",
        query="What are the key patterns for building multi-agent AI systems?",
        scope="comprehensive",
        required_sources=5,
        focus_areas=["architecture", "coordination", "scalability"]
    )

    # Execute research
    project = await orchestrator.research(query)

    # Display results
    print("\n" + "=" * 60)
    print("Research Results")
    print("=" * 60)

    print(f"\nProject ID: {project.id}")
    print(f"Status: {project.status.value}")

    if project.final_report:
        print("\n" + "-" * 40)
        print("FINAL REPORT (excerpt)")
        print("-" * 40)
        print(project.final_report[:1500] + "...")

    # Sources summary
    print("\n" + "-" * 40)
    print("Sources Used")
    print("-" * 40)
    for source in project.sources.values():
        print(f"  [{source.source_type.value}] {source.title}")
        print(f"    Credibility: {source.credibility.value}")

    # Facts summary
    print("\n" + "-" * 40)
    print("Key Facts Extracted")
    print("-" * 40)
    for fact in list(project.facts.values())[:5]:
        print(f"  - {fact.statement[:80]}...")
        print(f"    Confidence: {fact.confidence:.0%}")

    # Metrics
    print("\n" + "=" * 60)
    print("System Metrics")
    print("=" * 60)

    metrics = orchestrator.get_metrics()
    print(f"\nProjects started: {metrics['projects_started']}")
    print(f"Projects completed: {metrics.get('projects_completed', 0)}")

    print("\nAgent activity:")
    for name, count in metrics['agents'].items():
        print(f"  {name}: {count} tasks")


if __name__ == "__main__":
    asyncio.run(main())
