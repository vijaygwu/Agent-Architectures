"""
Chapter 6: The Council Pattern
===============================
Implementation of multi-agent deliberation for high-stakes decisions.

The council pattern enables multiple agents to:
1. Propose solutions independently
2. Critique each other's proposals
3. Deliberate through structured debate
4. Reach consensus or escalate to human

Use this pattern when single-agent decisions aren't trustworthy enough.
"""

__all__ = [
    "VotingMechanism",
    "ExpertiseArea",
    "Councilor",
    "CouncilConfig",
    "Proposal",
    "Critique",
    "Vote",
    "RankedVote",
    "VoteResult",
    "DeliberationRound",
    "CouncilDecision",
    "TranscriptEntry",
    "ConsensusResult",
    "Position",
    "DeliberationState",
    "CouncilMember",
    "Council",
    "create_technical_council",
    "create_business_council",
    "create_product_council",
    "BOOK_COUNCILORS",
    "SECURITY_COUNCILOR",
]

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    from anthropic import AsyncAnthropic, APIError, APITimeoutError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None
    APIError = Exception
    APITimeoutError = Exception

# Import common utilities for structured logging and retry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
try:
    from utils import configure_logging, with_retry, get_tracer
    logger = configure_logging(level="INFO", json_output=True, logger_name="council")
    tracer = get_tracer("council")
except ImportError:
    # No-op tracer fallback
    class _NoOpSpan:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def set_attribute(self, k, v): pass
        def add_event(self, n, a=None): pass
    class _NoOpTracer:
        def start_as_current_span(self, n, **kw): return _NoOpSpan()
    tracer = _NoOpTracer()
    # Fallback to basic logging if utils not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("council")

    # Define a simple retry decorator fallback (matching utils.py signature)
    def with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0, exponential_base=2.0, retryable_exceptions=(Exception,)):
        def decorator(func):
            async def wrapper(*args, **kw):
                last_error = None
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kw)
                    except retryable_exceptions as e:
                        last_error = e
                        if attempt < max_attempts - 1:
                            delay = min(base_delay * (exponential_base ** attempt), max_delay)
                            await asyncio.sleep(delay)
                raise last_error
            return wrapper
        return decorator


# =============================================================================
# Circuit Breaker for LLM Resilience
# =============================================================================
# The circuit breaker pattern prevents repeated failures when the LLM API is
# having issues. This is critical for council deliberations because:
#
# 1. Fail Fast: If the LLM API is down, don't waste time waiting for timeouts
#    on every council member's proposal, critique, and vote.
#
# 2. Recovery Detection: Automatically detect when the API recovers and
#    resume normal operation.
#
# 3. Cost Control: Prevent runaway retry costs when the API is having issues.
#
# Each CouncilMember shares a class-level circuit breaker to aggregate
# failure signals across all council members.
# =============================================================================

import os

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen, RateLimiter, RateLimitExceeded
    RESILIENCE_AVAILABLE = True
except ImportError:
    # Inline fallback for when resilience module is not available
    RESILIENCE_AVAILABLE = False

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

    class RateLimitExceeded(Exception):
        """Exception raised when rate limit is exceeded."""
        def __init__(self, limiter_name: str, retry_after: float = 1.0):
            self.limiter_name = limiter_name
            self.retry_after = retry_after
            super().__init__(f"Rate limit exceeded for '{limiter_name}', retry after {retry_after:.2f}s")

    class RateLimiter:
        """Minimal inline token bucket rate limiter when resilience module unavailable."""
        def __init__(self, name: str, max_tokens: float = 10.0, refill_rate: float = 1.0):
            self.name = name
            self.max_tokens = max_tokens
            self.refill_rate = refill_rate
            self._tokens = max_tokens
            import time
            self._time = time
            self._last_refill = time.monotonic()

        def _refill(self):
            now = self._time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

        def allow(self, tokens: float = 1.0) -> bool:
            """Check if request is allowed (non-blocking)."""
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

        def get_wait_time(self, tokens: float = 1.0) -> float:
            """Calculate time until tokens available."""
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            return (tokens - self._tokens) / self.refill_rate


# Shared circuit breaker for LLM API calls across all council members
# This aggregates failure signals - if any council member experiences
# repeated LLM failures, all members will fail fast until recovery
_LLM_CIRCUIT_BREAKER = CircuitBreaker(
    name="council:llm_api",
    failure_threshold=5,       # Open after 5 consecutive failures
    recovery_timeout=60.0      # Try again after 60 seconds
)

# Shared rate limiter for LLM API calls (configurable via environment variable)
# Default: 60 requests per minute (refill_rate=1.0 means 1 token/second)
_llm_rate_limiter = RateLimiter(
    name="council:llm_rate",
    max_tokens=float(os.environ.get("COUNCIL_LLM_RATE_LIMIT", "60")),
    refill_rate=float(os.environ.get("COUNCIL_LLM_RATE_LIMIT", "60")) / 60.0  # tokens per second
)


# =============================================================================
# Council Configuration
# =============================================================================

class VotingMechanism(str, Enum):
    # Naming note: PLURALITY is the choice with the most votes regardless
    # of whether it crosses 50%; MAJORITY is the choice with strictly more
    # than half the votes (and may return no winner, requiring a runoff
    # or further deliberation). Earlier revisions of this code conflated
    # the two; keep them distinct because they pick different winners
    # whenever no option clears 50%.
    """Methods for reaching council decisions"""
    PLURALITY = "plurality"         # Most votes wins (may be below 50%)
    MAJORITY = "majority"           # Strictly more than half required
    SUPERMAJORITY = "supermajority" # 2/3 or 3/4 required
    UNANIMOUS = "unanimous"         # All must agree
    RANKED = "ranked"               # Ranked choice voting
    WEIGHTED = "weighted"           # Some votes count more


class ExpertiseArea(str, Enum):
    """Areas of expertise for council members"""
    TECHNICAL = "technical"
    SECURITY = "security"
    BUSINESS = "business"
    LEGAL = "legal"
    ETHICS = "ethics"
    OPERATIONS = "operations"


# =============================================================================
# Councilor Dataclass (from book)
# =============================================================================

@dataclass
class Councilor:
    """
    Councilor definition as shown in the book.
    Each councilor has a distinct perspective defined by their system prompt.
    """
    id: str
    name: str
    perspective: str
    expertise: list[str]
    system_prompt: str

    def create_agent(self, llm_client) -> "CouncilMember":
        """Create an agent from this councilor definition."""
        # Map expertise from list to enum
        expertise_map = {
            "security": ExpertiseArea.SECURITY,
            "technical": ExpertiseArea.TECHNICAL,
            "business": ExpertiseArea.BUSINESS,
            "legal": ExpertiseArea.LEGAL,
            "ethics": ExpertiseArea.ETHICS,
            "operations": ExpertiseArea.OPERATIONS,
        }
        expertise = ExpertiseArea.TECHNICAL
        for exp in self.expertise:
            if exp.lower() in expertise_map:
                expertise = expertise_map[exp.lower()]
                break
        return CouncilMember(
            member_id=self.id,
            name=self.name,
            expertise=expertise,
            persona=self.system_prompt,
            mock_mode=False
        )


@dataclass
class CouncilConfig:
    """Configuration for a council"""
    name: str
    description: str
    voting_method: VotingMechanism = VotingMechanism.MAJORITY
    max_deliberation_rounds: int = 3
    min_confidence_threshold: float = 0.7
    require_human_for_deadlock: bool = True
    require_human_above_impact: str | None = "high"  # low, medium, high, critical

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_deliberation_rounds <= 0:
            raise ValueError("max_deliberation_rounds must be positive")
        if not (0 <= self.min_confidence_threshold <= 1):
            raise ValueError("min_confidence_threshold must be between 0 and 1")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Proposal:
    """A proposed solution from a council member"""
    id: str
    member_id: str
    content: str
    reasoning: str
    confidence: float
    timestamp: str
    supporting_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass
class Critique:
    """A critique of another member's proposal"""
    id: str
    member_id: str
    target_proposal_id: str
    assessment: Literal["support", "oppose", "neutral"]
    concerns: list[str]
    suggestions: list[str]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "target_proposal_id": self.target_proposal_id,
            "assessment": self.assessment,
            "concerns": self.concerns,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


@dataclass
class Vote:
    """A vote on a proposal"""
    member_id: str
    proposal_id: str
    vote: Literal["approve", "reject", "abstain"]
    reasoning: str
    timestamp: str
    choice: str = ""  # Populated in vote() for tally compatibility


@dataclass
class RankedVote:
    """A ranked choice vote for multiple options"""
    member_id: str
    rankings: dict[str, int]  # option -> rank (1 = first choice)
    timestamp: str


@dataclass
class VoteResult:
    """Result of a vote tally"""
    winner: str
    vote_count: int
    total_votes: int
    margin: float = 0.0
    unanimous: bool = False
    dissents: list[str] = field(default_factory=list)
    weighted_score: float = 0.0
    weights_used: dict[str, float] = field(default_factory=dict)


@dataclass
class DeliberationRound:
    """Record of a single deliberation round"""
    round_number: int
    proposals: list[Proposal]
    critiques: list[Critique]
    votes: list[Vote] | None = None
    outcome: str | None = None  # consensus, deadlock, escalate


@dataclass
class CouncilDecision:
    """Final decision from the council"""
    decision_id: str
    question: str
    decision: str
    confidence: float
    voting_result: dict
    deliberation_rounds: list[DeliberationRound]
    dissenting_opinions: list[dict]
    requires_human_approval: bool
    timestamp: str


# =============================================================================
# Deliberation State Management (from book)
# =============================================================================

@dataclass
class TranscriptEntry:
    """Entry in the deliberation transcript"""
    phase: str
    speaker: str
    content: str


@dataclass
class ConsensusResult:
    """Result of a consensus check"""
    consensus: str  # "yes", "no", "partial"
    confidence: str  # "high", "medium", "low"
    analysis: str


@dataclass
class Position:
    """A councilor's position in a round"""
    round: int
    content: str
    confidence: float


@dataclass
class DeliberationState:
    """Tracks the state of an ongoing deliberation."""
    question: str
    context: dict
    current_round: int = 0
    proposals: dict[str, list[Proposal]] = field(default_factory=dict)
    critiques: dict[str, list[Critique]] = field(default_factory=dict)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    consensus_checks: list[ConsensusResult] = field(default_factory=list)

    def get_active_proposals(self) -> list[Proposal]:
        """Get proposals from the current round."""
        return self.proposals.get(str(self.current_round), [])

    def get_full_transcript(self) -> str:
        """Format the complete deliberation transcript."""
        lines = []
        for entry in self.transcript:
            lines.append(f"[{entry.phase}] {entry.speaker}: {entry.content}")
        return "\n\n".join(lines)

    def get_councilor_evolution(self, councilor_id: str) -> list[Position]:
        """Track how a councilor's position evolved across rounds."""
        positions = []
        for round_num, proposals in sorted(self.proposals.items()):
            for proposal in proposals:
                if proposal.member_id == councilor_id:
                    positions.append(Position(
                        round=int(round_num),
                        content=proposal.content,
                        confidence=proposal.confidence
                    ))
        return positions


# =============================================================================
# Council Member Agent
# =============================================================================

class CouncilMember:
    """
    An agent participating in council deliberation.

    Each member has:
    - Specific expertise area
    - Unique perspective/persona
    - Ability to propose, critique, and vote
    """

    def __init__(
        self,
        member_id: str,
        name: str,
        expertise: ExpertiseArea,
        persona: str,
        model: str = "claude-sonnet-4",
        mock_mode: bool = False
    ):
        # Input validation
        if not member_id or not member_id.strip():
            raise ValueError("member_id cannot be empty")
        if not name or not name.strip():
            raise ValueError("name cannot be empty")

        self.member_id = member_id
        self.name = name
        self.expertise = expertise
        self.persona = persona
        self.model = model
        self.mock_mode = mock_mode
        self.client = None

        # Per-member circuit breaker for failure isolation
        # Each council member has their own circuit breaker to prevent
        # one failing member from affecting others
        self._circuit_breaker = CircuitBreaker(
            name=f"council_member_{member_id}",
            failure_threshold=3,
            recovery_timeout=30.0
        )

        if not mock_mode:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic library required. Install with: pip install anthropic\n"
                    "Or use mock_mode=True for testing without API calls"
                )
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Either:\n"
                    "  1. Set the environment variable: export ANTHROPIC_API_KEY=your-key\n"
                    "  2. Use mock_mode=True for testing without API calls"
                )
            self.client = AsyncAnthropic()

    async def __aenter__(self):
        """Support async context manager for standalone usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close resources on context exit."""
        await self.close()

    async def close(self):
        """Close the client connection to prevent resource leaks."""
        if self.client:
            await self.client.close()

    @with_retry(max_attempts=3, base_delay=1.0, retryable_exceptions=(asyncio.TimeoutError, APIError, APITimeoutError))
    async def _call_llm(
        self,
        system_prompt: str,
        user_content: str,
        operation_name: str
    ) -> str:
        """
        Make an LLM API call with circuit breakers, rate limiting, retry logic and timeout.

        Resilience layers (in order):
        1. Rate limiter: Prevents exceeding LLM API rate limits (shared across all members)
        2. Per-member circuit breaker: Isolates failures for each council member
        3. Shared circuit breaker: Aggregates failure signals across all members
        4. Retry with backoff: Handles transient failures (via decorator)

        Args:
            system_prompt: The system prompt for the LLM
            user_content: The user message content
            operation_name: Name of the operation (for error messages)

        Returns:
            The raw text response from the LLM

        Raises:
            RateLimitExceeded: If LLM rate limit is exceeded
            CircuitBreakerOpen: If circuit breaker is open (fail fast)
            TimeoutError: If the API call times out
        """
        # Check rate limiter first (shared across all council members)
        if not _llm_rate_limiter.allow():
            wait_time = _llm_rate_limiter.get_wait_time()
            logger.warning(
                f"LLM rate limit exceeded in {operation_name}",
                extra={"extra_fields": {
                    "member_id": self.member_id,
                    "operation": operation_name,
                    "retry_after": wait_time
                }}
            )
            raise RateLimitExceeded(_llm_rate_limiter.name, wait_time)

        # Use per-member circuit breaker for failure isolation
        # This prevents one failing member from blocking others
        try:
            async with self._circuit_breaker:
                # Also use shared circuit breaker for global LLM API health
                async with _LLM_CIRCUIT_BREAKER:
                    response = await asyncio.wait_for(
                        self.client.messages.create(
                            model=self.model,
                            max_tokens=1024,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_content}]
                        ),
                        timeout=60.0
                    )
                    return response.content[0].text
        except CircuitBreakerOpen as e:
            # Circuit is open - fail fast
            logger.warning(
                f"Circuit breaker open in {operation_name}",
                extra={"extra_fields": {
                    "member_id": self.member_id,
                    "operation": operation_name,
                    "breaker_name": e.breaker_name,
                    "retry_in": e.time_until_retry
                }}
            )
            raise
        except asyncio.TimeoutError:
            logger.warning(
                f"LLM timeout in {operation_name}",
                extra={"extra_fields": {"member_id": self.member_id, "operation": operation_name}}
            )
            raise TimeoutError(
                f"LLM API call timed out after 60 seconds in {operation_name} for {self.member_id}"
            )

    async def propose(
        self,
        question: str,
        context: dict,
        previous_proposals: list[Proposal] | None = None
    ) -> Proposal:
        """
        Generate a proposal for the question.

        If previous proposals exist (in later rounds), consider them
        when formulating the response.
        """
        # Input validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        with tracer.start_as_current_span(f"council.propose.{self.member_id}") as span:
            span.set_attribute("member.id", self.member_id)
            span.set_attribute("member.expertise", self.expertise.value)
            span.set_attribute("has_previous_proposals", previous_proposals is not None)

            system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Your persona: {self.persona}

You are participating in a council deliberation to make an important decision.
Provide your proposal with clear reasoning and confidence level.

Respond in JSON format:
{{
    "proposal": "Your proposed decision/solution",
    "reasoning": "Step-by-step reasoning for your proposal",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", "evidence2"]
}}"""

            user_content = f"Question for deliberation: {question}\n\nContext: {json.dumps(context)}"

            if previous_proposals:
                user_content += "\n\nPrevious proposals from other members:\n"
                for p in previous_proposals:
                    user_content += f"- {p.member_id}: {p.content}\n"

            # Mock mode for testing without API
            if self.mock_mode:
                result = {
                    "proposal": f"[Mock] {self.name} proposes a balanced approach to: {question[:50]}...",
                    "reasoning": f"As a {self.expertise.value} expert, I recommend careful evaluation.",
                    "confidence": 0.75,
                    "supporting_evidence": ["Mock evidence 1", "Mock evidence 2"]
                }
            else:
                # Use retry-wrapped LLM call
                response_text = await self._call_llm(system_prompt, user_content, "propose")

                # Parse response
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse proposal JSON from {self.member_id}",
                        extra={"extra_fields": {"error": str(e), "response_preview": response_text[:200]}}
                    )
                    result = {
                        "proposal": response_text,
                        "reasoning": "Unable to parse structured response",
                        "confidence": 0.5,
                        "supporting_evidence": []
                    }

            # Create Proposal from result (works for both mock and real modes)
            proposal = Proposal(
                id=f"prop_{uuid.uuid4().hex[:8]}",
                member_id=self.member_id,
                content=result.get("proposal", ""),
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.5),
                timestamp=datetime.now(timezone.utc).isoformat(),
                supporting_evidence=result.get("supporting_evidence", [])
            )
            span.set_attribute("proposal.id", proposal.id)
            span.set_attribute("proposal.confidence", proposal.confidence)
            return proposal

    async def critique(
        self,
        proposal: Proposal,
        question: str,
        context: dict
    ) -> Critique:
        """
        Critique another member's proposal.

        Provide assessment, concerns, and constructive suggestions.
        """
        # Input validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if proposal is None:
            raise ValueError("Proposal cannot be None")

        with tracer.start_as_current_span(f"council.critique.{self.member_id}") as span:
            span.set_attribute("member.id", self.member_id)
            span.set_attribute("target.proposal_id", proposal.id)
            span.set_attribute("target.member_id", proposal.member_id)

            system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Your persona: {self.persona}

You are reviewing another council member's proposal. Provide a fair but thorough critique
from your area of expertise.

Respond in JSON format:
{{
    "assessment": "support" | "oppose" | "neutral",
    "concerns": ["concern1", "concern2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

            user_content = f"""Question: {question}

Proposal from {proposal.member_id}:
{proposal.content}

Their reasoning:
{proposal.reasoning}

Confidence: {proposal.confidence}

Context: {json.dumps(context)}"""

            # Mock mode for testing without API
            if self.mock_mode:
                result = {
                    "assessment": "support",
                    "concerns": [f"[Mock] Consider {self.expertise.value} implications"],
                    "suggestions": ["[Mock] Further analysis recommended"]
                }
            else:
                # Use retry-wrapped LLM call
                response_text = await self._call_llm(system_prompt, user_content, "critique")

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse critique JSON from {self.member_id}",
                        extra={"extra_fields": {"error": str(e)}}
                    )
                    result = {
                        "assessment": "neutral",
                        "concerns": ["Unable to parse structured critique"],
                        "suggestions": []
                    }

            # Create Critique from result (works for both mock and real modes)
            critique = Critique(
                id=f"crit_{uuid.uuid4().hex[:8]}",
                member_id=self.member_id,
                target_proposal_id=proposal.id,
                assessment=result.get("assessment", "neutral"),
                concerns=result.get("concerns", []),
                suggestions=result.get("suggestions", []),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            span.set_attribute("critique.id", critique.id)
            span.set_attribute("critique.assessment", critique.assessment)
            return critique

    async def vote(
        self,
        proposal: Proposal,
        critiques: list[Critique],
        question: str
    ) -> Vote:
        """
        Vote on a proposal after reviewing all critiques.
        """
        # Input validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if proposal is None:
            raise ValueError("Proposal cannot be None")

        with tracer.start_as_current_span(f"council.vote.{self.member_id}") as span:
            span.set_attribute("member.id", self.member_id)
            span.set_attribute("proposal.id", proposal.id)
            span.set_attribute("critiques.count", len(critiques))

            system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Based on the proposal and all critiques, cast your vote.

Respond in JSON format:
{{
    "vote": "approve" | "reject" | "abstain",
    "reasoning": "Brief explanation for your vote"
}}"""

            critiques_text = "\n".join([
                f"- {c.member_id} ({c.assessment}): {c.concerns}"
                for c in critiques
            ])

            user_content = f"""Question: {question}

Proposal:
{proposal.content}

Critiques:
{critiques_text}"""

            # Mock mode for testing without API
            if self.mock_mode:
                result = {"vote": "approve", "reasoning": f"[Mock] {self.name} approves after review"}
            else:
                # Use retry-wrapped LLM call
                response_text = await self._call_llm(system_prompt, user_content, "vote")

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse vote JSON from {self.member_id}",
                        extra={"extra_fields": {"error": str(e)}}
                    )
                    result = {"vote": "abstain", "reasoning": "Unable to parse vote"}

            # Create Vote from result (works for both mock and real modes)
            vote_value = result.get("vote", "abstain")
            vote_obj = Vote(
                member_id=self.member_id,
                proposal_id=proposal.id,
                vote=vote_value,
                reasoning=result.get("reasoning", ""),
                timestamp=datetime.now(timezone.utc).isoformat(),
                choice=vote_value
            )
            span.set_attribute("vote.value", vote_value)
            return vote_obj


# =============================================================================
# Council Implementation
# =============================================================================

class Council:
    """
    Multi-agent deliberation council for high-stakes decisions.

    The council follows this process:
    1. Each member proposes independently
    2. Members critique each other's proposals
    3. Voting occurs based on configured method
    4. If no consensus, iterate or escalate

    Usage:
        async with Council(config, members) as council:
            decision = await council.deliberate(question, context)
    """

    def __init__(
        self,
        config: CouncilConfig,
        members: list[CouncilMember]
    ):
        # Input validation
        if not members or len(members) == 0:
            raise ValueError("Council must have at least one member")

        self.config = config
        self.members = members
        self.deliberation_log: list[DeliberationRound] = []

    async def close(self):
        """Close all member clients to prevent resource leaks."""
        for member in self.members:
            await member.close()

    async def __aenter__(self):
        """Support async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context exit."""
        await self.close()

    async def deliberate(
        self,
        question: str,
        context: dict,
        impact_level: str = "medium"
    ) -> CouncilDecision:
        """
        Run full deliberation process on a question.

        Args:
            question: The decision question
            context: Relevant context and data
            impact_level: low, medium, high, critical

        Returns:
            CouncilDecision with final outcome and audit trail
        """
        # Input validation
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        decision_id = f"decision_{uuid.uuid4().hex[:8]}"
        self.deliberation_log = []

        with tracer.start_as_current_span(f"council.deliberate.{decision_id}") as span:
            span.set_attribute("decision.id", decision_id)
            span.set_attribute("impact_level", impact_level)
            span.set_attribute("members.count", len(self.members))

            for round_num in range(self.config.max_deliberation_rounds):
                span.add_event(f"round.{round_num + 1}.start")
                round_result = await self._run_deliberation_round(
                    round_num + 1,
                    question,
                    context
                )
                self.deliberation_log.append(round_result)

                # Check for consensus
                if round_result.outcome == "consensus":
                    span.set_attribute("outcome", "consensus")
                    break
                elif round_result.outcome == "deadlock":
                    if self.config.require_human_for_deadlock:
                        span.set_attribute("outcome", "escalated")
                        return self._create_escalation_decision(
                            decision_id, question, "Deadlock after deliberation"
                        )

            # Final voting
            winning_proposal, voting_result = await self._final_vote(question)

            # Check if human approval required
            requires_human = self._check_human_approval_required(
                impact_level,
                voting_result
            )

            # Collect dissenting opinions
            dissenting = self._collect_dissenting_opinions(
                winning_proposal,
                self.deliberation_log[-1].critiques
            )

            span.set_attribute("rounds.total", len(self.deliberation_log))
            span.set_attribute("requires_human_approval", requires_human)

            return CouncilDecision(
                decision_id=decision_id,
                question=question,
                decision=winning_proposal.content if winning_proposal else "No consensus reached",
                confidence=winning_proposal.confidence if winning_proposal else 0.0,
                voting_result=voting_result,
                deliberation_rounds=self.deliberation_log,
                dissenting_opinions=dissenting,
                requires_human_approval=requires_human,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    async def _run_deliberation_round(
        self,
        round_number: int,
        question: str,
        context: dict
    ) -> DeliberationRound:
        """Run a single round of deliberation"""

        # Get previous proposals if not first round
        previous_proposals = None
        if round_number > 1 and self.deliberation_log:
            previous_proposals = self.deliberation_log[-1].proposals

        # Phase 1: Gather proposals (parallel)
        proposal_tasks = [
            member.propose(question, context, previous_proposals)
            for member in self.members
        ]
        proposals = await asyncio.gather(*proposal_tasks)

        # Phase 2: Cross-critique (parallel)
        critique_tasks = []
        for member in self.members:
            other_proposals = [p for p in proposals if p.member_id != member.member_id]
            for proposal in other_proposals:
                critique_tasks.append(
                    member.critique(proposal, question, context)
                )

        critiques = await asyncio.gather(*critique_tasks)

        # Determine round outcome
        outcome = self._assess_round_outcome(proposals, critiques)

        return DeliberationRound(
            round_number=round_number,
            proposals=list(proposals),
            critiques=list(critiques),
            outcome=outcome
        )

    def _assess_round_outcome(
        self,
        proposals: list[Proposal],
        critiques: list[Critique]
    ) -> str:
        """Assess whether consensus is emerging"""

        # Count supportive vs opposing critiques
        support_count = sum(1 for c in critiques if c.assessment == "support")
        oppose_count = sum(1 for c in critiques if c.assessment == "oppose")

        total_critiques = len(critiques)
        if total_critiques == 0:
            return "deadlock"

        support_ratio = support_count / total_critiques

        # Check confidence levels
        avg_confidence = sum(p.confidence for p in proposals) / len(proposals)

        if support_ratio > 0.7 and avg_confidence > self.config.min_confidence_threshold:
            return "consensus"
        elif oppose_count > support_count * 2:
            return "deadlock"
        else:
            return "continue"

    async def _final_vote(
        self,
        question: str
    ) -> tuple[Proposal | None, dict]:
        """Conduct final voting on best proposal"""

        if not self.deliberation_log:
            return None, {"error": "No deliberation rounds"}

        last_round = self.deliberation_log[-1]
        proposals = last_round.proposals
        critiques = last_round.critiques

        # Each member votes on each proposal
        all_votes: list[Vote] = []

        for proposal in proposals:
            relevant_critiques = [
                c for c in critiques if c.target_proposal_id == proposal.id
            ]

            vote_tasks = [
                member.vote(proposal, relevant_critiques, question)
                for member in self.members
                if member.member_id != proposal.member_id  # Don't vote on own proposal
            ]

            votes = await asyncio.gather(*vote_tasks)
            all_votes.extend(votes)

        # Tally votes
        vote_counts: dict[str, dict] = {}
        for proposal in proposals:
            proposal_votes = [v for v in all_votes if v.proposal_id == proposal.id]
            vote_counts[proposal.id] = {
                "approve": sum(1 for v in proposal_votes if v.vote == "approve"),
                "reject": sum(1 for v in proposal_votes if v.vote == "reject"),
                "abstain": sum(1 for v in proposal_votes if v.vote == "abstain")
            }

        # Determine winner based on voting method
        winner = self._determine_winner(proposals, vote_counts)

        return winner, vote_counts

    def _determine_winner(
        self,
        proposals: list[Proposal],
        vote_counts: dict[str, dict]
    ) -> Proposal | None:
        """Determine winning proposal based on voting method"""

        total_voters = len(self.members) - 1  # Exclude proposer

        if self.config.voting_method == VotingMechanism.UNANIMOUS:
            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] == total_voters:
                    return proposal
            return None

        elif self.config.voting_method == VotingMechanism.MAJORITY:
            threshold = total_voters / 2
            best_proposal = None
            best_approval = 0

            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] > threshold and counts["approve"] > best_approval:
                    best_proposal = proposal
                    best_approval = counts["approve"]

            return best_proposal

        elif self.config.voting_method == VotingMechanism.SUPERMAJORITY:
            threshold = total_voters * 0.66

            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] >= threshold:
                    return proposal

            return None

        elif self.config.voting_method == VotingMechanism.RANKED:
            # Use ranked choice voting
            # Convert vote_counts to ranked votes format for tally_ranked_choice
            ranked_votes = self._convert_to_ranked_votes(proposals, vote_counts)
            result = self.tally_ranked_choice(ranked_votes)
            for proposal in proposals:
                if proposal.id == result.winner:
                    return proposal
            return None

        else:  # CONSENSUS - return highest approval
            best_proposal = max(
                proposals,
                key=lambda p: vote_counts[p.id]["approve"]
            )
            return best_proposal

    def _convert_to_ranked_votes(
        self, proposals: list[Proposal], vote_counts: dict[str, dict]
    ) -> dict[str, RankedVote]:
        """Convert approval-based votes to ranked votes for ranked choice voting."""
        # Create synthetic ranked votes based on approval counts
        ranked_votes = {}
        for member in self.members:
            # Rank proposals by approval count (higher approval = lower rank number)
            sorted_proposals = sorted(
                proposals,
                key=lambda p: vote_counts[p.id]["approve"],
                reverse=True
            )
            rankings = {p.id: i + 1 for i, p in enumerate(sorted_proposals)}
            ranked_votes[member.member_id] = RankedVote(
                member_id=member.member_id,
                rankings=rankings,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        return ranked_votes

    def tally_plurality(self, votes: dict[str, Vote]) -> VoteResult:
        """Plurality tally: the choice with the most votes wins, even if it
        falls short of a strict majority. Conflating plurality and strict
        majority is a common social-choice-theory mistake; they pick
        different winners whenever no option crosses 50 percent. See
        tally_majority below for the strict-majority variant.
        """
        counts = Counter(v.choice for v in votes.values())
        winner, count = counts.most_common(1)[0]
        total = len(votes)

        return VoteResult(
            winner=winner,
            vote_count=count,
            total_votes=total,
            margin=count / total,
            unanimous=(count == total),
            dissents=[cid for cid, v in votes.items() if v.choice != winner]
        )

    def tally_majority(self, votes: dict[str, Vote]) -> VoteResult | None:
        """Strict majority tally: return a winner only if some option
        receives more than half of the votes. Returns None to signal
        that a runoff or further deliberation is required.
        """
        counts = Counter(v.choice for v in votes.values())
        winner, count = counts.most_common(1)[0]
        total = len(votes)
        if count <= total / 2:
            return None
        return VoteResult(
            winner=winner,
            vote_count=count,
            total_votes=total,
            margin=count / total,
            unanimous=(count == total),
            dissents=[cid for cid, v in votes.items() if v.choice != winner],
        )

    def tally_ranked_choice(self, votes: dict[str, RankedVote]) -> VoteResult:
        """Instant runoff ranked choice voting."""
        remaining_options = set(votes[list(votes.keys())[0]].rankings.keys())
        total = 0

        while len(remaining_options) > 1:
            # Count first-choice votes for remaining options
            first_choices = Counter()
            for vote in votes.values():
                for choice in vote.rankings:
                    if choice in remaining_options:
                        first_choices[choice] += 1
                        break

            # Check for majority
            total = sum(first_choices.values())
            for option, count in first_choices.items():
                if count > total / 2:
                    return VoteResult(winner=option, vote_count=count, total_votes=total)

            # Eliminate last place (tie-break: alphabetically first)
            min_count = first_choices.most_common()[-1][1]
            tied_last = [
                opt for opt, cnt in first_choices.items() if cnt == min_count
            ]
            last_place = sorted(tied_last)[0]
            remaining_options.remove(last_place)

        winner = remaining_options.pop()
        return VoteResult(winner=winner, vote_count=len(votes), total_votes=len(votes))

    def _check_human_approval_required(
        self,
        impact_level: str,
        voting_result: dict
    ) -> bool:
        """Check if decision requires human approval"""

        # Always require human for configured impact level
        impact_hierarchy = ["low", "medium", "high", "critical"]
        if self.config.require_human_above_impact:
            threshold_idx = impact_hierarchy.index(self.config.require_human_above_impact)
            current_idx = impact_hierarchy.index(impact_level)
            if current_idx >= threshold_idx:
                return True

        # Require human if close vote
        for proposal_id, counts in voting_result.items():
            if isinstance(counts, dict):
                approve = counts.get("approve", 0)
                reject = counts.get("reject", 0)
                if approve > 0 and reject > 0:
                    ratio = min(approve, reject) / max(approve, reject)
                    if ratio > 0.6:  # Close vote
                        return True

        return False

    def _collect_dissenting_opinions(
        self,
        winning_proposal: Proposal | None,
        critiques: list[Critique]
    ) -> list[dict]:
        """Collect dissenting opinions for the audit trail"""

        if not winning_proposal:
            return []

        dissenting = []
        for critique in critiques:
            if (critique.target_proposal_id == winning_proposal.id and
                critique.assessment == "oppose"):
                dissenting.append({
                    "member_id": critique.member_id,
                    "concerns": critique.concerns,
                    "suggestions": critique.suggestions
                })

        return dissenting

    def _create_escalation_decision(
        self,
        decision_id: str,
        question: str,
        reason: str
    ) -> CouncilDecision:
        """Create a decision that escalates to human"""
        return CouncilDecision(
            decision_id=decision_id,
            question=question,
            decision=f"ESCALATED: {reason}",
            confidence=0.0,
            voting_result={},
            deliberation_rounds=self.deliberation_log,
            dissenting_opinions=[],
            requires_human_approval=True,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


# =============================================================================
# Pre-built Council Configurations
# =============================================================================

def create_technical_council(mock_mode: bool = False) -> Council:
    """Create a council for technical decisions.

    Args:
        mock_mode: If True, use simulated responses (no API key required)
    """
    config = CouncilConfig(
        name="Technical Review Council",
        description="Reviews technical decisions and architecture changes",
        voting_method=VotingMechanism.MAJORITY,
        max_deliberation_rounds=2,
        require_human_above_impact="high"
    )

    members = [
        CouncilMember(
            member_id="architect",
            name="Alex Architect",
            expertise=ExpertiseArea.TECHNICAL,
            persona="Senior software architect focused on scalability and maintainability",
            mock_mode=mock_mode
        ),
        CouncilMember(
            member_id="security",
            name="Sam Security",
            expertise=ExpertiseArea.SECURITY,
            persona="Security engineer focused on threat modeling and secure design",
            mock_mode=mock_mode
        ),
        CouncilMember(
            member_id="ops",
            name="Olivia Ops",
            expertise=ExpertiseArea.OPERATIONS,
            persona="SRE focused on reliability, observability, and operational excellence",
            mock_mode=mock_mode
        )
    ]

    return Council(config, members)


def create_business_council(mock_mode: bool = False) -> Council:
    """Create a council for business decisions.

    Args:
        mock_mode: If True, use simulated responses (no API key required)
    """
    config = CouncilConfig(
        name="Business Decision Council",
        description="Reviews business-critical decisions",
        voting_method=VotingMechanism.SUPERMAJORITY,
        max_deliberation_rounds=3,
        require_human_above_impact="medium"
    )

    members = [
        CouncilMember(
            member_id="business",
            name="Blake Business",
            expertise=ExpertiseArea.BUSINESS,
            persona="Business strategist focused on ROI and market impact",
            mock_mode=mock_mode
        ),
        CouncilMember(
            member_id="legal",
            name="Leslie Legal",
            expertise=ExpertiseArea.LEGAL,
            persona="Legal counsel focused on compliance and risk mitigation",
            mock_mode=mock_mode
        ),
        CouncilMember(
            member_id="ethics",
            name="Evan Ethics",
            expertise=ExpertiseArea.ETHICS,
            persona="Ethics advisor focused on fairness and responsible AI",
            mock_mode=mock_mode
        )
    ]

    return Council(config, members)


# =============================================================================
# Example Councilors from Book
# =============================================================================

# Example councilors as defined in the book chapter
BOOK_COUNCILORS = [
    Councilor(
        id="customer_advocate",
        name="Customer Advocate",
        perspective="Prioritizes user experience and customer value",
        expertise=["UX", "customer feedback", "market needs"],
        system_prompt="""You are the Customer Advocate on a product council.

Your role is to represent customer interests in all decisions. You:
- Argue for features that solve real customer problems
- Push back on complexity that hurts usability
- Cite customer feedback and research
- Consider diverse user segments

You are NOT a pushover. Advocate strongly for customers."""
    ),
    Councilor(
        id="tech_realist",
        name="Technical Realist",
        perspective="Focuses on implementation feasibility and technical debt",
        expertise=["engineering", "architecture", "scalability"],
        system_prompt="""You are the Technical Realist on a product council.

Your role is to ground discussions in technical reality. You:
- Assess implementation complexity honestly
- Warn about technical debt and maintenance burden
- Propose technically sound alternatives
- Consider security and performance implications

You're realistic, not pessimistic. Support good ideas, flag genuine concerns."""
    ),
    Councilor(
        id="business_strategist",
        name="Business Strategist",
        perspective="Considers market position and business impact",
        expertise=["strategy", "competition", "revenue"],
        system_prompt="""You are the Business Strategist on a product council.

Your role is to ensure decisions align with business goals. You:
- Consider competitive positioning
- Evaluate revenue and cost implications
- Think about market timing
- Balance short-term and long-term value

Good business requires happy customers and sustainable engineering."""
    ),
]


# Security councilor example from book
SECURITY_COUNCILOR = Councilor(
    id="security_reviewer",
    name="Security Analyst",
    perspective="security-first evaluation",
    expertise=["threat modeling", "secure architecture", "compliance"],
    system_prompt="""You are the Security Analyst on this council.

Your job is to identify and advocate for security considerations. You:
- Assume attackers are sophisticated and persistent
- Identify attack surfaces in every proposal
- Require specific mitigations, not vague assurances
- Consider both technical and human factors
- Reference relevant compliance requirements

You are NOT paranoid or obstructionist. Security enables business by making risk
visible and manageable. Support proposals that address your concerns adequately.

When critiquing, always propose specific improvements rather than just rejecting.
When approving, clearly state what security properties you verified.

Your recommendations should be proportional to the actual risk level."""
)


def create_product_council(mock_mode: bool = False) -> Council:
    """Create a product council using the book's example councilors.

    This demonstrates the councilor definitions from the book chapter.

    Args:
        mock_mode: If True, use simulated responses (no API key required)
    """
    config = CouncilConfig(
        name="Product Council",
        description="Reviews product feature decisions",
        voting_method=VotingMechanism.MAJORITY,
        max_deliberation_rounds=3,
        require_human_above_impact="high"
    )

    # Create members from book councilors
    members = [
        CouncilMember(
            member_id=c.id,
            name=c.name,
            expertise=ExpertiseArea.BUSINESS,  # Default for product decisions
            persona=c.system_prompt,
            mock_mode=mock_mode
        )
        for c in BOOK_COUNCILORS
    ]

    return Council(config, members)


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate council deliberation"""

    # Create technical council with mock mode for demo (no API key needed)
    council = create_technical_council(mock_mode=True)

    # Question for deliberation
    question = """
    Should we migrate our monolithic application to microservices architecture?

    Current state:
    - 500K lines of code
    - 50 developers
    - 99.9% uptime requirement
    - Major feature releases every 2 weeks
    """

    context = {
        "current_architecture": "monolith",
        "team_size": 50,
        "annual_budget": "$5M",
        "risk_tolerance": "medium"
    }

    try:
        print("=" * 60)
        print("Council Deliberation Demo")
        print("=" * 60)
        print(f"\nQuestion: {question[:100]}...")
        print(f"\nCouncil: {council.config.name}")
        print(f"Members: {[m.name for m in council.members]}")
        print(f"Voting Method: {council.config.voting_method.value}")

        # Run deliberation
        print("\nDeliberating...")
        decision = await council.deliberate(
            question=question,
            context=context,
            impact_level="high"
        )

        print("\n" + "=" * 60)
        print("DECISION")
        print("=" * 60)
        print(f"\nDecision ID: {decision.decision_id}")
        print(f"Outcome: {decision.decision}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Requires Human Approval: {decision.requires_human_approval}")

        print(f"\nDeliberation Rounds: {len(decision.deliberation_rounds)}")
        for round_info in decision.deliberation_rounds:
            print(f"  Round {round_info.round_number}: {round_info.outcome}")
            print(f"    Proposals: {len(round_info.proposals)}")
            print(f"    Critiques: {len(round_info.critiques)}")

        if decision.dissenting_opinions:
            print(f"\nDissenting Opinions: {len(decision.dissenting_opinions)}")
            for dissent in decision.dissenting_opinions:
                print(f"  - {dissent['member_id']}: {dissent['concerns'][:2]}")
    finally:
        await council.close()


if __name__ == "__main__":
    asyncio.run(main())
