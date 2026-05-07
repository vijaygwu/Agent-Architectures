"""
Chapter 11: Policy Fundamentals
===============================

Implements policy enforcement for AI agents including:
- Policy definition with conditions and actions
- Rate limiting
- Content filtering
- Budget/quota policies
- Policy gateway for centralized enforcement
- Human-in-the-loop approval workflows
- Policy testing and simulation
- Policy observability

Based on Chapter 11 of the Agent Architectures book.
"""

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from secrets import token_urlsafe
from typing import Any, Callable, Protocol

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from pydantic import BaseModel

# Import structured logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
try:
    from utils import configure_logging
    logger = configure_logging(level="INFO", json_output=True, logger_name="gateway")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gateway")

# Optional dependency check
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Prometheus metrics support (optional)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True

    # Define Prometheus metrics
    REQUEST_COUNT = Counter(
        'gateway_requests_total',
        'Total requests processed by the gateway',
        ['method', 'endpoint', 'status']
    )
    REQUEST_LATENCY = Histogram(
        'gateway_request_latency_seconds',
        'Request latency in seconds',
        ['endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    ACTIVE_TASKS = Gauge(
        'gateway_active_tasks',
        'Number of currently active tasks'
    )
    POLICY_DECISIONS = Counter(
        'gateway_policy_decisions_total',
        'Policy decisions made by the gateway',
        ['policy_name', 'action']
    )
    CIRCUIT_BREAKER_STATE = Gauge(
        'gateway_circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half-open)',
        ['breaker_name']
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None
    CONTENT_TYPE_LATEST = None


# =============================================================================
# FastAPI App and Health Check Endpoints
# =============================================================================

app = FastAPI(title="Policy Gateway", version="1.0.0") if FASTAPI_AVAILABLE else None

# Track startup time
_startup_time = datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]


if app:
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Liveness probe - is the service running?"""
        now = datetime.now(timezone.utc)
        return HealthResponse(
            status="healthy",
            timestamp=now.isoformat(),
            version="1.0.0",
            uptime_seconds=(now - _startup_time).total_seconds()
        )

    @app.get("/ready", response_model=ReadinessResponse)
    async def readiness_check() -> ReadinessResponse:
        """Readiness probe - is the service ready to accept traffic?"""
        checks = {}

        # Check if required dependencies are available
        checks["anthropic_sdk"] = ANTHROPIC_AVAILABLE
        checks["config_loaded"] = True  # Add actual config check if needed

        return ReadinessResponse(
            ready=all(checks.values()),
            checks=checks
        )

    @app.get("/metrics")
    async def metrics(format: str = "json"):
        """
        Metrics endpoint for monitoring.

        Args:
            format: Output format - 'json' (default) or 'prometheus'

        Returns:
            Metrics in requested format
        """
        if format == "prometheus" and PROMETHEUS_AVAILABLE:
            from fastapi.responses import Response
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )

        return {
            "requests_total": getattr(app.state, 'request_count', 0),
            "errors_total": getattr(app.state, 'error_count', 0),
            "uptime_seconds": (datetime.now(timezone.utc) - _startup_time).total_seconds(),
            "prometheus_available": PROMETHEUS_AVAILABLE
        }

    @app.get("/metrics/prometheus")
    async def prometheus_metrics():
        """
        Prometheus-compatible metrics endpoint.

        Returns metrics in Prometheus text format for scraping.
        Install prometheus-client: pip install prometheus-client
        """
        if not PROMETHEUS_AVAILABLE:
            return {
                "error": "prometheus_client not installed",
                "install": "pip install prometheus-client"
            }

        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# =============================================================================
# Policy Definition (Section 11.4)
# =============================================================================

class PolicyAction(Enum):
    """Actions a policy rule can trigger"""
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMIT = "rate_limit"
    REQUIRE_APPROVAL = "require_approval"
    LOG = "log"


@dataclass
class PolicyCondition:
    """A condition that must be met for the policy to apply."""
    field: str              # e.g., "agent.role", "request.resource"
    operator: str           # e.g., "equals", "contains", "greater_than"
    value: Any

    def evaluate(self, context: dict) -> bool:
        """Evaluate condition against context."""
        actual = self._get_field(context, self.field)

        if self.operator == "equals":
            return actual == self.value
        elif self.operator == "not_equals":
            return actual != self.value
        elif self.operator == "contains":
            return actual is not None and self.value in actual
        elif self.operator == "greater_than":
            return actual is not None and actual > self.value
        elif self.operator == "less_than":
            return actual is not None and actual < self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "matches":
            return bool(re.match(self.value, str(actual)))

        return False

    def _get_field(self, context: dict, field: str) -> Any:
        """Navigate nested fields like 'agent.role'."""
        parts = field.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
        return value


@dataclass
class Policy:
    """A policy that controls agent behavior."""
    id: str
    name: str
    description: str
    conditions: list[PolicyCondition]
    action: PolicyAction
    action_params: dict = field(default_factory=dict)
    priority: int = 0  # Higher = evaluated first
    enabled: bool = True

    def applies(self, context: dict) -> bool:
        """Check if this policy applies to the given context."""
        if not self.enabled:
            return False
        return all(c.evaluate(context) for c in self.conditions)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action": self.action.value,
            "priority": self.priority
        }


# =============================================================================
# Rate Limiting (Section 11.5.1)
# =============================================================================

class RateLimitPolicy:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        scope: str = "global"  # "global", "per_agent", "per_resource"
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.scope = scope
        self.requests: dict[str, list[datetime]] = defaultdict(list)

    def check(self, context: dict) -> tuple[bool, dict]:
        """Check if request is within rate limit."""
        key = self._get_key(context)
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_seconds)

        # Clean old requests
        self.requests[key] = [
            t for t in self.requests[key]
            if t > window_start
        ]

        if len(self.requests[key]) >= self.max_requests:
            retry_after = (self.requests[key][0] - window_start).total_seconds()
            return False, {
                "reason": "rate_limit_exceeded",
                "limit": self.max_requests,
                "window_seconds": self.window_seconds,
                "retry_after": retry_after
            }

        self.requests[key].append(now)
        return True, {"remaining": self.max_requests - len(self.requests[key])}

    def _get_key(self, context: dict) -> str:
        if self.scope == "global":
            return "global"
        elif self.scope == "per_agent":
            return context.get("agent_id", "unknown")
        elif self.scope == "per_resource":
            return context.get("resource", "unknown")
        return "global"


# =============================================================================
# Content Filtering (Section 11.5.2)
# =============================================================================

@dataclass
class ContentRule:
    name: str
    pattern: str  # Regex pattern
    action: str   # "block" or "redact"
    violation_message: str
    contexts: list[str] = None  # None = all contexts

    def applies(self, context: dict) -> bool:
        if self.contexts is None:
            return True
        return context.get("type") in self.contexts

    def matches(self, text: str) -> bool:
        return bool(re.search(self.pattern, text, re.IGNORECASE))

    def filter(self, text: str) -> str:
        if self.action == "redact":
            return re.sub(self.pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text


class ContentPolicy:
    def __init__(self):
        self.rules: list[ContentRule] = []

    def add_rule(self, rule: "ContentRule"):
        self.rules.append(rule)

    def check(self, content: Any, context: dict) -> tuple[bool, list[str]]:
        """Check content against all rules."""
        violations = []

        text = self._extract_text(content)

        for rule in self.rules:
            if rule.applies(context) and rule.matches(text):
                violations.append(rule.violation_message)

        return len(violations) == 0, violations

    def filter(self, content: Any, context: dict) -> Any:
        """Filter/redact violating content."""
        text = self._extract_text(content)

        for rule in self.rules:
            if rule.applies(context):
                text = rule.filter(text)

        return text

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content)
        return str(content)


# =============================================================================
# Budget/Quota Policies (Section 11.5.3)
# =============================================================================

@dataclass
class BudgetPolicy:
    max_amount: float
    period: str  # "request", "hour", "day", "month"
    resource: str  # "tokens", "dollars", "api_calls"

    def __post_init__(self):
        self.usage: dict[str, float] = defaultdict(float)
        self.period_start: dict[str, datetime] = {}

    def check(self, agent_id: str, amount: float) -> tuple[bool, dict]:
        """Check if usage would exceed budget."""
        key = f"{agent_id}:{self.period}"

        # Reset if period expired
        if self._period_expired(key):
            self.usage[key] = 0
            self.period_start[key] = datetime.now(timezone.utc)

        current = self.usage[key]
        if current + amount > self.max_amount:
            return False, {
                "reason": "budget_exceeded",
                "current": current,
                "requested": amount,
                "limit": self.max_amount,
                "resource": self.resource
            }

        return True, {
            "current": current,
            "remaining": self.max_amount - current - amount
        }

    def record(self, agent_id: str, amount: float):
        """Record usage."""
        key = f"{agent_id}:{self.period}"
        self.usage[key] += amount

    def _period_expired(self, key: str) -> bool:
        if key not in self.period_start:
            return True

        start = self.period_start[key]
        now = datetime.now(timezone.utc)

        if self.period == "request":
            return True
        elif self.period == "hour":
            return (now - start).total_seconds() > 3600
        elif self.period == "day":
            return (now - start).days >= 1
        elif self.period == "month":
            return (now - start).days >= 30

        return True


# =============================================================================
# The Policy Gateway (Section 11.6)
# =============================================================================

class Verdict(Enum):
    """Final decision after evaluating all policies"""
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMITED = "rate_limited"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class PolicyDecision:
    verdict: Verdict
    matched_policies: list[str]
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PolicyGateway:
    """
    Central policy enforcement point.

    All agent requests flow through the gateway,
    which evaluates applicable policies and makes decisions.
    """

    def __init__(self):
        self.policies: list[Policy] = []
        self.rate_limiters: dict[str, RateLimitPolicy] = {}
        self.content_policy: ContentPolicy = ContentPolicy()
        self.budget_policies: dict[str, BudgetPolicy] = {}

    def add_policy(self, policy: Policy):
        """Add a policy to the gateway."""
        self.policies.append(policy)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def add_rate_limit(self, name: str, policy: RateLimitPolicy):
        """Add a rate limit policy."""
        self.rate_limiters[name] = policy

    def add_budget(self, name: str, policy: BudgetPolicy):
        """Add a budget policy."""
        self.budget_policies[name] = policy

    def evaluate(self, request: dict) -> PolicyDecision:
        """Evaluate a request against all policies."""
        context = self._build_context(request)
        matched = []
        reasons = []

        # Check general policies (highest priority first)
        for policy in self.policies:
            if policy.applies(context):
                matched.append(policy.id)

                if policy.action == PolicyAction.DENY:
                    return PolicyDecision(
                        verdict=Verdict.DENY,
                        matched_policies=matched,
                        reasons=[policy.description]
                    )
                elif policy.action == PolicyAction.REQUIRE_APPROVAL:
                    return PolicyDecision(
                        verdict=Verdict.NEEDS_APPROVAL,
                        matched_policies=matched,
                        reasons=[policy.description]
                    )

        # Check rate limits
        for name, limiter in self.rate_limiters.items():
            allowed, info = limiter.check(context)
            if not allowed:
                return PolicyDecision(
                    verdict=Verdict.RATE_LIMITED,
                    matched_policies=[name],
                    reasons=[info.get("reason", "Rate limit exceeded")],
                    metadata=info
                )

        # Check budgets
        agent_id = context.get("agent_id")
        estimated_cost = request.get("estimated_cost", 0)

        for name, budget in self.budget_policies.items():
            allowed, info = budget.check(agent_id, estimated_cost)
            if not allowed:
                return PolicyDecision(
                    verdict=Verdict.DENY,
                    matched_policies=[name],
                    reasons=["Budget exceeded"],
                    metadata=info
                )

        # Check content if present
        if "content" in request:
            allowed, violations = self.content_policy.check(
                request["content"],
                context
            )
            if not allowed:
                return PolicyDecision(
                    verdict=Verdict.DENY,
                    matched_policies=["content_policy"],
                    reasons=violations
                )

        return PolicyDecision(
            verdict=Verdict.ALLOW,
            matched_policies=matched,
            reasons=["All policies passed"]
        )

    def _build_context(self, request: dict) -> dict:
        """Build evaluation context from request."""
        return {
            "agent_id": request.get("agent_id"),
            "agent_role": request.get("agent_role"),
            "action": request.get("action"),
            "resource": request.get("resource"),
            "type": request.get("type"),
            "timestamp": datetime.now(timezone.utc),
            "request": request
        }


# =============================================================================
# Policy Enforced Agent (Section 11.7)
# =============================================================================

class PolicyViolationError(Exception):
    pass


class RateLimitedError(Exception):
    pass


class PolicyEnforcedAgent:
    def __init__(
        self,
        agent_id: str,
        gateway: PolicyGateway,
        llm_client: Any
    ):
        self.agent_id = agent_id
        self.gateway = gateway
        self.llm = llm_client

    async def execute(self, action: str, resource: str, content: Any = None) -> Any:
        """Execute an action through the policy gateway."""

        # Build request
        request = {
            "agent_id": self.agent_id,
            "action": action,
            "resource": resource,
            "content": content,
            "estimated_cost": self._estimate_cost(action, content),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Evaluate policies
        decision = self.gateway.evaluate(request)

        if decision.verdict == Verdict.DENY:
            raise PolicyViolationError(
                f"Action denied: {', '.join(decision.reasons)}"
            ) from None

        if decision.verdict == Verdict.RATE_LIMITED:
            retry_after = decision.metadata.get("retry_after", 60)
            raise RateLimitedError(
                f"Rate limited. Retry after {retry_after}s"
            )

        if decision.verdict == Verdict.NEEDS_APPROVAL:
            # Queue for human approval
            return await self._request_approval(request, decision)

        # Execute the action
        result = await self._do_execute(action, resource, content)

        # Record usage for budget tracking
        actual_cost = self._calculate_actual_cost(result)
        for budget in self.gateway.budget_policies.values():
            budget.record(self.agent_id, actual_cost)

        return result

    def _estimate_cost(self, action: str, content: Any) -> float:
        """Estimate cost before execution."""
        if action == "llm_call":
            # Estimate based on content length
            tokens = len(str(content).split()) * 1.3
            return tokens * 0.00003  # Example pricing
        return 0

    def _calculate_actual_cost(self, result: Any) -> float:
        """Calculate actual cost after execution."""
        if hasattr(result, "usage"):
            return result.usage.total_tokens * 0.00003
        return 0

    async def _do_execute(self, action: str, resource: str, content: Any) -> Any:
        """Execute the actual action (placeholder)."""
        # In a real implementation, this would dispatch to actual handlers
        return {"status": "success", "action": action, "resource": resource}

    async def _request_approval(self, request: dict, decision: PolicyDecision) -> Any:
        """Request human approval (placeholder)."""
        return {"status": "pending_approval", "request": request}


# =============================================================================
# Human-in-the-Loop Approval Workflows (Section 11.9)
# =============================================================================

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    id: str
    agent_id: str
    action: str
    resource: str
    content: Any
    reason: str  # Why approval is needed
    policy_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = None
    decided_by: str = None
    decided_at: datetime = None
    decision_reason: str = None


class ApprovalWorkflow:
    """Manages human approval for agent actions."""

    def __init__(self, default_timeout_hours: int = 24):
        self.pending: dict[str, ApprovalRequest] = {}
        self.default_timeout = default_timeout_hours
        self.notification_handlers: list[Callable] = []

    async def request_approval(
        self,
        agent_id: str,
        action: str,
        resource: str,
        content: Any,
        reason: str,
        policy_id: str,
        timeout_hours: int = None
    ) -> ApprovalRequest:
        """Create an approval request and notify approvers."""

        request_id = f"approval_{token_urlsafe(8)}"
        timeout = timeout_hours or self.default_timeout

        request = ApprovalRequest(
            id=request_id,
            agent_id=agent_id,
            action=action,
            resource=resource,
            content=content,
            reason=reason,
            policy_id=policy_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=timeout)
        )

        self.pending[request_id] = request

        # Notify approvers
        for handler in self.notification_handlers:
            await handler(request)

        return request

    async def wait_for_decision(
        self,
        request_id: str,
        poll_interval: int = 5
    ) -> ApprovalRequest:
        """Wait for a decision on an approval request."""

        while True:
            request = self.pending.get(request_id)
            if not request:
                raise ValueError(f"Unknown request: {request_id}")

            # Check expiration
            if datetime.now(timezone.utc) > request.expires_at:
                request.status = ApprovalStatus.EXPIRED
                return request

            # Check if decided
            if request.status != ApprovalStatus.PENDING:
                return request

            await asyncio.sleep(poll_interval)

    def approve(self, request_id: str, approver: str, reason: str = None):
        """Approve a pending request."""
        request = self.pending.get(request_id)
        if request and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.APPROVED
            request.decided_by = approver
            request.decided_at = datetime.now(timezone.utc)
            request.decision_reason = reason

    def deny(self, request_id: str, approver: str, reason: str):
        """Deny a pending request."""
        request = self.pending.get(request_id)
        if request and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.DENIED
            request.decided_by = approver
            request.decided_at = datetime.now(timezone.utc)
            request.decision_reason = reason


# =============================================================================
# Tiered Approval Thresholds (Section 11.9.1)
# =============================================================================

class ApprovalTier(Enum):
    AUTO = 0        # No approval needed
    MANAGER = 1     # Single manager approval
    DIRECTOR = 2    # Director-level approval
    EXECUTIVE = 3   # Executive approval


@dataclass
class TieredApprovalPolicy:
    """Policy that requires different approval levels."""

    thresholds: dict[str, list[tuple[float, ApprovalTier]]]

    def get_required_tier(self, action: str, amount: float) -> ApprovalTier:
        """Determine required approval tier."""

        if action not in self.thresholds:
            return ApprovalTier.AUTO

        for threshold, tier in sorted(self.thresholds[action], reverse=True):
            if amount >= threshold:
                return tier

        return ApprovalTier.AUTO


# =============================================================================
# Policy Testing and Simulation (Section 11.10)
# =============================================================================

@dataclass
class SimulationReport:
    total_requests: int
    results: dict
    policy_hit_rate: dict[str, float]


@dataclass
class PolicyDiff:
    total_tested: int
    behavioral_changes: int
    changes: list[dict]


class PolicySimulator:
    """Test policy configurations without affecting production."""

    def __init__(self, gateway: PolicyGateway):
        self.gateway = gateway
        self.simulation_log: list[dict] = []

    def simulate(self, requests: list[dict]) -> SimulationReport:
        """Simulate a batch of requests against current policies."""

        results = {
            "allowed": 0,
            "denied": 0,
            "rate_limited": 0,
            "needs_approval": 0
        }

        for request in requests:
            decision = self.gateway.evaluate(request)

            self.simulation_log.append({
                "request": request,
                "decision": decision.verdict.value,
                "matched_policies": decision.matched_policies
            })

            results[decision.verdict.value] += 1

        return SimulationReport(
            total_requests=len(requests),
            results=results,
            policy_hit_rate=self._calculate_hit_rates()
        )

    def _calculate_hit_rates(self) -> dict[str, float]:
        """Calculate how often each policy matched."""
        hits = defaultdict(int)

        for entry in self.simulation_log:
            for policy_id in entry["matched_policies"]:
                hits[policy_id] += 1

        total = len(self.simulation_log)
        return {k: v/total for k, v in hits.items()}

    def diff_policies(
        self,
        current_gateway: PolicyGateway,
        proposed_gateway: PolicyGateway,
        test_requests: list[dict]
    ) -> PolicyDiff:
        """Compare behavior of two policy configurations."""

        changes = []

        for request in test_requests:
            current_decision = current_gateway.evaluate(request)
            proposed_decision = proposed_gateway.evaluate(request)

            if current_decision.verdict != proposed_decision.verdict:
                changes.append({
                    "request": request,
                    "current": current_decision.verdict.value,
                    "proposed": proposed_decision.verdict.value
                })

        return PolicyDiff(
            total_tested=len(test_requests),
            behavioral_changes=len(changes),
            changes=changes
        )


# =============================================================================
# Policy Observability (Section 11.11)
# =============================================================================

class MetricsClient(Protocol):
    """Protocol for metrics collection."""
    def increment(self, name: str, tags: dict = None) -> None: ...
    def histogram(self, name: str, value: float) -> None: ...


class TraceClient(Protocol):
    """Protocol for distributed tracing."""
    def start_span(self, name: str, attributes: dict = None) -> Any: ...


class PolicyObserver:
    """Collects metrics and traces for policy enforcement."""

    def __init__(self, metrics_client: MetricsClient, trace_client: TraceClient):
        self.metrics = metrics_client
        self.tracer = trace_client

    def observe_decision(
        self,
        request: dict,
        decision: PolicyDecision,
        duration_ms: float
    ):
        """Record metrics for a policy decision."""

        # Count decisions by verdict
        self.metrics.increment(
            "policy.decisions.total",
            tags={
                "verdict": decision.verdict.value,
                "agent_id": request.get("agent_id", "unknown")
            }
        )

        # Record latency
        self.metrics.histogram(
            "policy.evaluation.duration_ms",
            duration_ms
        )

        # Record policy matches
        for policy_id in decision.matched_policies:
            self.metrics.increment(
                "policy.matches",
                tags={"policy_id": policy_id}
            )

    def create_trace(self, request: dict):
        """Create a trace span for policy evaluation."""
        return self.tracer.start_span(
            "policy.evaluate",
            attributes={
                "agent_id": request.get("agent_id"),
                "action": request.get("action"),
                "resource": request.get("resource")
            }
        )


# =============================================================================
# Common Policy Patterns (Section 11.12)
# =============================================================================

class EscalatingPolicy:
    """Policy that gets stricter after violations."""

    def __init__(
        self,
        base_limit: int,
        violation_penalty: float = 0.5,
        recovery_rate: float = 0.1
    ):
        self.base_limit = base_limit
        self.penalty = violation_penalty
        self.recovery = recovery_rate
        self.agent_limits: dict[str, float] = {}

    def get_limit(self, agent_id: str) -> int:
        """Get current limit for agent."""
        multiplier = self.agent_limits.get(agent_id, 1.0)
        return int(self.base_limit * multiplier)

    def record_violation(self, agent_id: str):
        """Reduce limit after violation."""
        current = self.agent_limits.get(agent_id, 1.0)
        self.agent_limits[agent_id] = max(0.1, current * self.penalty)

    def record_success(self, agent_id: str):
        """Slowly restore limit after successful behavior."""
        current = self.agent_limits.get(agent_id, 1.0)
        self.agent_limits[agent_id] = min(1.0, current + self.recovery)


@dataclass
class PolicyModifier:
    name: str
    condition: Callable[[dict], bool]
    modification: Callable[[Policy], Policy]

    def applies(self, context: dict) -> bool:
        return self.condition(context)

    def modify(self, policy: Policy) -> Policy:
        return self.modification(policy)


class ContextSensitivePolicy:
    """Policy that adjusts based on context."""

    def __init__(self, base_policy: Policy):
        self.base = base_policy
        self.modifiers: list[PolicyModifier] = []

    def evaluate(self, request: dict, context: dict) -> PolicyDecision:
        """Evaluate with context modifications."""

        modified_policy = self.base

        for modifier in self.modifiers:
            if modifier.applies(context):
                modified_policy = modifier.modify(modified_policy)

        # Build a simple decision based on policy evaluation
        if modified_policy.applies(context):
            if modified_policy.action == PolicyAction.DENY:
                return PolicyDecision(
                    verdict=Verdict.DENY,
                    matched_policies=[modified_policy.id],
                    reasons=[modified_policy.description]
                )
            elif modified_policy.action == PolicyAction.REQUIRE_APPROVAL:
                return PolicyDecision(
                    verdict=Verdict.NEEDS_APPROVAL,
                    matched_policies=[modified_policy.id],
                    reasons=[modified_policy.description]
                )

        return PolicyDecision(
            verdict=Verdict.ALLOW,
            matched_policies=[modified_policy.id],
            reasons=["Policy passed"]
        )

    def add_modifier(self, modifier: "PolicyModifier"):
        self.modifiers.append(modifier)


# =============================================================================
# Example Usage (Section 11.8)
# =============================================================================

async def main():
    # Create gateway
    gateway = PolicyGateway()

    # Add general policies
    gateway.add_policy(Policy(
        id="block_external_pii",
        name="Block PII in External Calls",
        description="Prevent PII from being sent to external services",
        conditions=[
            PolicyCondition("type", "equals", "external_api")
        ],
        action=PolicyAction.LOG,  # Will combine with content policy
        priority=100
    ))

    gateway.add_policy(Policy(
        id="require_approval_high_value",
        name="Require Approval for High-Value Actions",
        description="Actions over $100 need human approval",
        conditions=[
            PolicyCondition("estimated_cost", "greater_than", 100)
        ],
        action=PolicyAction.REQUIRE_APPROVAL,
        priority=90
    ))

    # Add rate limits
    gateway.add_rate_limit(
        "api_calls",
        RateLimitPolicy(max_requests=100, window_seconds=60, scope="per_agent")
    )

    gateway.add_rate_limit(
        "expensive_ops",
        RateLimitPolicy(max_requests=10, window_seconds=3600, scope="per_agent")
    )

    # Add budget
    gateway.add_budget(
        "daily_spend",
        BudgetPolicy(max_amount=500, period="day", resource="dollars")
    )

    # Add content rules
    gateway.content_policy.add_rule(ContentRule(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        action="block",
        violation_message="SSN detected in content"
    ))

    # Example: Block PII in external requests
    content_policy = ContentPolicy()
    content_policy.add_rule(ContentRule(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        action="block",
        violation_message="Social Security Number detected",
        contexts=["external_api"]
    ))
    content_policy.add_rule(ContentRule(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        action="redact",
        violation_message="Email address detected"
    ))
    content_policy.add_rule(ContentRule(
        name="credit_card",
        pattern=r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        action="block",
        violation_message="Credit card number detected"
    ))

    # Example: 100 requests per minute per agent
    rate_policy = RateLimitPolicy(
        max_requests=100,
        window_seconds=60,
        scope="per_agent"
    )

    # Example: $100/day per agent
    budget_policy = BudgetPolicy(
        max_amount=100.0,
        period="day",
        resource="dollars"
    )

    # Example: Spending approval tiers
    spending_policy = TieredApprovalPolicy(
        thresholds={
            "purchase": [
                (0, ApprovalTier.AUTO),      # Under $100 - auto-approve
                (100, ApprovalTier.MANAGER),  # $100-$1000 - manager
                (1000, ApprovalTier.DIRECTOR), # $1000-$10000 - director
                (10000, ApprovalTier.EXECUTIVE) # Over $10000 - executive
            ],
            "data_export": [
                (0, ApprovalTier.MANAGER),    # Any data export needs manager
                (1000, ApprovalTier.DIRECTOR)  # Over 1000 records - director
            ]
        }
    )

    # Create a mock LLM client for testing
    class MockLLM:
        pass

    llm = MockLLM()

    # Create agent with policy enforcement
    agent = PolicyEnforcedAgent(
        agent_id="agent_123",
        gateway=gateway,
        llm_client=llm
    )

    # Test various requests
    print("=" * 60)
    print("Policy Gateway Demonstration")
    print("=" * 60)

    try:
        # Normal request - should pass
        print("\nTest 1: Normal read request")
        result = await agent.execute("read", "documents/report.pdf")
        print(f"  Result: {result}")

        # Request with PII - should be blocked
        print("\nTest 2: Request with PII (should be blocked)")
        await agent.execute(
            "external_api_call",
            "https://api.example.com",
            content="Customer SSN: 123-45-6789"
        )
    except PolicyViolationError as e:
        print(f"  Blocked: {e}")
    except RateLimitedError as e:
        print(f"  Rate limited: {e}")

    # Test rate limiting
    print("\nTest 3: Rate limit test")
    context = {"agent_id": "test_agent"}
    for i in range(5):
        allowed, info = rate_policy.check(context)
        print(f"  Request {i+1}: {'allowed' if allowed else 'denied'} - {info}")

    # Test budget policy
    print("\nTest 4: Budget policy test")
    allowed, info = budget_policy.check("agent_123", 50.0)
    print(f"  $50 request: {'allowed' if allowed else 'denied'} - {info}")
    budget_policy.record("agent_123", 50.0)

    allowed, info = budget_policy.check("agent_123", 60.0)
    print(f"  $60 request: {'allowed' if allowed else 'denied'} - {info}")

    # Test approval tiers
    print("\nTest 5: Approval tier test")
    tier = spending_policy.get_required_tier("purchase", 50)
    print(f"  $50 purchase requires: {tier.name}")
    tier = spending_policy.get_required_tier("purchase", 500)
    print(f"  $500 purchase requires: {tier.name}")
    tier = spending_policy.get_required_tier("purchase", 5000)
    print(f"  $5000 purchase requires: {tier.name}")

    print("\n" + "=" * 60)
    print("Demonstration complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
