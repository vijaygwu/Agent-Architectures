"""
Complete Example: Multi-Agent Research Assistant
================================================

A production-ready research assistant demonstrating:
- Multi-agent collaboration for complex research tasks
- Swarm-based parallel information gathering
- Council-based synthesis and validation
- Source attribution and citation management
- Quality control and fact-checking

Handles research queries from planning through final report.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from collections import defaultdict
import hashlib


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

    async def process(self, project: ResearchProject, context: dict) -> dict:
        """Gather information for research."""
        self.tasks_completed += 1

        plan = project.plan
        strategies = plan.get("search_strategies", [])

        # Simulate parallel gathering
        gather_tasks = []
        for strategy in strategies:
            gather_tasks.append(self._gather_for_strategy(strategy))

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
                    id=f"src_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
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
            "avg_confidence": sum(f.confidence for f in project.facts.values()) /
                            len(project.facts) if project.facts else 0
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
                id=f"fact_{hashlib.md5((source.id + str(i)).encode()).hexdigest()[:8]}",
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
    """

    def __init__(self):
        self.planner = PlannerAgent()
        self.gatherer = GathererAgent()
        self.analyzer = AnalyzerAgent()
        self.synthesizer = SynthesizerAgent()
        self.validator = ValidatorAgent()
        self.reporter = ReportGenerator()

        self.projects: dict[str, ResearchProject] = {}
        self.metrics = defaultdict(int)

    async def research(self, query: ResearchQuery) -> ResearchProject:
        """Execute complete research workflow."""
        # Create project
        project = ResearchProject(
            id=f"proj_{int(time.time())}",
            query=query
        )
        self.projects[project.id] = project
        self.metrics["projects_started"] += 1

        print(f"\n[RESEARCH] Starting project {project.id}")
        print(f"  Query: {query.query}")
        print(f"  Scope: {query.scope}")

        # Phase 1: Planning
        print("\n[PHASE 1] Planning research strategy...")
        plan_result = await self.planner.process(project, {})
        print(f"  Sub-questions: {len(plan_result['sub_questions'])}")
        print(f"  Search strategies: {len(plan_result['search_strategies'])}")

        # Phase 2: Gathering
        print("\n[PHASE 2] Gathering information...")
        gather_result = await self.gatherer.process(project, {})
        print(f"  Sources gathered: {gather_result['sources_gathered']}")
        print(f"  By type: {gather_result['by_type']}")

        # Phase 3: Analysis
        print("\n[PHASE 3] Analyzing content...")
        analyze_result = await self.analyzer.process(project, {})
        print(f"  Facts extracted: {analyze_result['facts_extracted']}")
        print(f"  Average confidence: {analyze_result['avg_confidence']:.0%}")

        # Phase 4: Synthesis
        print("\n[PHASE 4] Synthesizing findings...")
        synth_result = await self.synthesizer.process(project, {})
        print(f"  Sections created: {synth_result['sections_created']}")

        # Phase 5: Validation
        print("\n[PHASE 5] Validating quality...")
        valid_result = await self.validator.process(project, {})
        print(f"  Validation passed: {valid_result['overall_passed']}")
        print(f"  Score: {valid_result['overall_score']:.0%}")

        # Phase 6: Report Generation
        if valid_result['overall_passed']:
            print("\n[PHASE 6] Generating report...")
            report_result = await self.reporter.process(project, {})
            print(f"  Report length: {report_result['report_length']} chars")
            self.metrics["projects_completed"] += 1
        else:
            print("\n[PHASE 6] Skipped - validation failed")
            self.metrics["projects_failed"] += 1

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
            }
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
