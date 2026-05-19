"""
Chapter 8: The Guardian Pattern
================================
Implementation of safety oversight agents that validate actions before execution.

The guardian pattern provides:
1. Policy-based action validation
2. Safety checks before execution
3. Graduated autonomy based on trust
4. Audit trail for all decisions

Reference: Deutsche Telekom case study - 95% resolution time improvement
with guardian agents validating network changes before deployment.

Production Notes:
- Deploy guardians as separate services for fault isolation
- Use centralized policy management for consistent enforcement
- Implement guardian health monitoring with automatic failover
- Store audit logs in append-only storage for compliance
"""

__all__ = [
    "GuardianConfig",
    "MetricsExporter",
    "SecurityMetrics",
    "InMemoryMetricsExporter",
    "ValidationError",
    "GuardianDecision",
    "ValidationResult",
    "ActionRequest",
    "Guardian",
    "GuardianPipeline",
    "GuardedExecutor",
    "ContentPolicy",
    "ContentGuardian",
    "ActionGuardian",
    "CostGuardian",
    "SecurityGuardian",
    "CircuitState",
    "CircuitConfig",
    "CircuitBreaker",
    "CircuitBreakerGuardian",
    "create_defense_in_depth_pipeline",
    "ResilientGuardianPipeline",
    "FailurePolicy",
    "create_production_pipeline",
    "EscalationTicket",
    "EscalationManager",
    "GuardedExecutorWithEscalation",
    "GuardianMonitor",
    "create_guardian_dashboard",
]

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

# Configure logging for audit trail
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardian")


# =============================================================================
# Guardian Configuration (Environment-Configurable)
# =============================================================================

@dataclass
class GuardianConfig:
    """Externally configurable thresholds for guardian alerting and circuit breakers.

    All values can be overridden via environment variables:
        GUARDIAN_ALERT_THRESHOLD: Escalation spike threshold for alerts
        GUARDIAN_MAX_VIOLATIONS: Max violations per minute before alerting
        GUARDIAN_CB_THRESHOLD: Circuit breaker failure threshold
        GUARDIAN_ESCALATION_TIMEOUT: Escalation timeout in seconds
    """
    alert_threshold: int = int(os.environ.get("GUARDIAN_ALERT_THRESHOLD", "10"))
    max_violations_per_minute: int = int(os.environ.get("GUARDIAN_MAX_VIOLATIONS", "100"))
    circuit_breaker_threshold: int = int(os.environ.get("GUARDIAN_CB_THRESHOLD", "5"))
    escalation_timeout: float = float(os.environ.get("GUARDIAN_ESCALATION_TIMEOUT", "300.0"))

    def __post_init__(self):
        """Validate configuration values."""
        if self.alert_threshold <= 0:
            raise ValueError("Alert threshold must be positive")
        if self.max_violations_per_minute <= 0:
            raise ValueError("Max violations per minute must be positive")
        if self.circuit_breaker_threshold <= 0:
            raise ValueError("Circuit breaker threshold must be positive")
        if self.escalation_timeout <= 0:
            raise ValueError("Escalation timeout must be positive")


# =============================================================================
# Metrics Export Protocol and Implementation
# =============================================================================

class MetricsExporter(Protocol):
    """Protocol for exporting security metrics to external systems.

    Implement this protocol to integrate with your monitoring stack:
    - Prometheus: Use prometheus_client library
    - DataDog: Use datadog library
    - SIEM systems: Format events appropriately

    Example implementation for Prometheus:
        class PrometheusMetricsExporter:
            def __init__(self):
                from prometheus_client import Counter, Histogram
                self.events_total = Counter(
                    'security_events_total',
                    'Total security events',
                    ['severity', 'event_type']
                )
                self.violations_by_type = Counter(
                    'security_violations_total',
                    'Violations by type',
                    ['violation_type']
                )

            async def record_security_event(self, event: dict) -> None:
                self.events_total.labels(
                    severity=event['severity'],
                    event_type=event['action_type']
                ).inc()
    """

    async def record_security_event(self, event: dict) -> None:
        """Record a security event for metrics/monitoring.

        Args:
            event: Security event dict with keys:
                - timestamp: ISO format timestamp
                - severity: 'low', 'medium', 'high', 'critical'
                - agent_id: ID of the agent that triggered the event
                - action_type: Type of action attempted
                - violations: List of violation descriptions
                - request_id: Unique request identifier
                - parameters_hash: SHA256 hash of request parameters
        """
        ...

    async def increment_counter(self, name: str, labels: dict[str, str]) -> None:
        """Increment a named counter with labels."""
        ...

    async def record_histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Record a value in a histogram metric."""
        ...


@dataclass
class SecurityMetrics:
    """Aggregated security metrics for reporting."""
    security_events_total: int = 0
    violations_by_type: dict[str, int] = field(default_factory=dict)
    actions_blocked: int = 0
    actions_approved: int = 0
    escalations: int = 0
    events_by_severity: dict[str, int] = field(default_factory=lambda: {
        "low": 0, "medium": 0, "high": 0, "critical": 0
    })

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for export."""
        return {
            "security_events_total": self.security_events_total,
            "violations_by_type": self.violations_by_type,
            "actions_blocked": self.actions_blocked,
            "actions_approved": self.actions_approved,
            "escalations": self.escalations,
            "events_by_severity": self.events_by_severity
        }

    def to_siem_format(self) -> list[dict]:
        """Export metrics in SIEM-compatible format (CEF-like)."""
        return [
            {
                "event_type": "security_metrics_summary",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": self.to_dict(),
                "format_version": "1.0"
            }
        ]


class InMemoryMetricsExporter:
    """In-memory metrics exporter for testing and development.

    Usage:
        exporter = InMemoryMetricsExporter()
        guardian = SecurityGuardian("security", config, metrics_exporter=exporter)

        # After some operations...
        metrics = exporter.get_metrics()
        print(f"Total events: {metrics.security_events_total}")
        print(f"Blocked actions: {metrics.actions_blocked}")
    """

    def __init__(self):
        self.metrics = SecurityMetrics()
        self.events: list[dict] = []
        self._lock = asyncio.Lock()

    async def record_security_event(self, event: dict) -> None:
        """Record a security event."""
        async with self._lock:
            self.events.append(event)
            self.metrics.security_events_total += 1

            severity = event.get("severity", "medium")
            if severity in self.metrics.events_by_severity:
                self.metrics.events_by_severity[severity] += 1

            for violation in event.get("violations", []):
                # Extract violation type from violation message
                violation_type = self._extract_violation_type(violation)
                self.metrics.violations_by_type[violation_type] = (
                    self.metrics.violations_by_type.get(violation_type, 0) + 1
                )

    async def increment_counter(self, name: str, labels: dict[str, str]) -> None:
        """Increment a named counter."""
        async with self._lock:
            if name == "actions_blocked":
                self.metrics.actions_blocked += 1
            elif name == "actions_approved":
                self.metrics.actions_approved += 1
            elif name == "escalations":
                self.metrics.escalations += 1

    async def record_histogram(self, name: str, value: float, labels: dict[str, str]) -> None:
        """Record histogram value (no-op for in-memory exporter)."""
        pass

    def _extract_violation_type(self, violation: str) -> str:
        """Extract a violation type category from violation message."""
        violation_lower = violation.lower()
        if "injection" in violation_lower:
            return "injection"
        elif "rate limit" in violation_lower:
            return "rate_limit"
        elif "permission" in violation_lower:
            return "permission"
        elif "sensitive" in violation_lower:
            return "sensitive_data"
        elif "traversal" in violation_lower:
            return "path_traversal"
        else:
            return "other"

    def get_metrics(self) -> SecurityMetrics:
        """Get current metrics snapshot."""
        return self.metrics

    def get_events(self) -> list[dict]:
        """Get all recorded events."""
        return self.events.copy()

    def reset(self) -> None:
        """Reset all metrics and events."""
        self.metrics = SecurityMetrics()
        self.events = []


# =============================================================================
# Exceptions
# =============================================================================

class ValidationError(Exception):
    """Raised when guardian validation fails."""
    pass


# =============================================================================
# Core Enums and Data Models
# =============================================================================

class GuardianDecision(Enum):
    """Possible decisions from a guardian validation."""
    APPROVE = "approve"
    DENY = "deny"
    MODIFY = "modify"
    ESCALATE = "escalate"


@dataclass
class ValidationResult:
    """Result of guardian validation."""
    decision: GuardianDecision
    reason: str
    modified_action: dict | None = None
    violations: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_approved(self) -> bool:
        return self.decision in (GuardianDecision.APPROVE, GuardianDecision.MODIFY)


@dataclass
class ActionRequest:
    """An action proposed by an agent for validation."""
    action_type: str
    parameters: dict
    agent_id: str
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = field(default_factory=lambda: hashlib.sha256(
        str(datetime.now(timezone.utc).timestamp()).encode()
    ).hexdigest()[:16])


# =============================================================================
# Base Guardian Interface
# =============================================================================

class Guardian(ABC):
    """Base class for all guardians."""

    # Default max audit log entries to prevent unbounded memory growth
    DEFAULT_MAX_AUDIT_ENTRIES = 10000

    def __init__(self, guardian_id: str, config: dict = None):
        self.id = guardian_id
        self.config = config or {}
        max_entries = self.config.get("max_audit_entries", self.DEFAULT_MAX_AUDIT_ENTRIES)
        self.audit_log: deque[dict] = deque(maxlen=max_entries)

    @abstractmethod
    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Validate an action request."""
        pass

    def log_decision(self, action: ActionRequest, result: ValidationResult):
        """Record decision for audit trail."""
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "guardian_id": self.id,
            "request_id": action.request_id,
            "action_type": action.action_type,
            "decision": result.decision.value,
            "reason": result.reason,
            "violations": result.violations
        })


# =============================================================================
# Guardian Pipeline
# =============================================================================

class GuardianPipeline:
    """Chains multiple guardians with configurable behavior."""

    def __init__(self, guardians: list[Guardian], mode: str = "all"):
        self.guardians = guardians
        self.mode = mode  # "all" = all must approve, "any" = any approval suffices

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Run action through all guardians."""
        results = []
        modified_action = action.parameters.copy()

        for guardian in self.guardians:
            # Pass potentially modified action to next guardian
            action_copy = ActionRequest(
                action_type=action.action_type,
                parameters=modified_action,
                agent_id=action.agent_id,
                context=action.context,
                timestamp=action.timestamp,
                request_id=action.request_id
            )

            result = await guardian.validate(action_copy)
            guardian.log_decision(action_copy, result)
            results.append(result)

            # Handle escalation immediately
            if result.decision == GuardianDecision.ESCALATE:
                return ValidationResult(
                    decision=GuardianDecision.ESCALATE,
                    reason=f"Guardian {guardian.id} requires human review: {result.reason}",
                    violations=result.violations
                )

            # In "all" mode, any denial stops the pipeline
            if self.mode == "all" and result.decision == GuardianDecision.DENY:
                return result

            # Apply modifications
            if result.decision == GuardianDecision.MODIFY and result.modified_action:
                modified_action = result.modified_action

        # Aggregate results
        if self.mode == "all":
            # All approved (or modified)
            all_violations = []
            for r in results:
                all_violations.extend(r.violations)

            return ValidationResult(
                decision=GuardianDecision.APPROVE if not any(
                    r.decision == GuardianDecision.MODIFY for r in results
                ) else GuardianDecision.MODIFY,
                reason="All guardians approved",
                modified_action=modified_action if modified_action != action.parameters else None,
                violations=all_violations
            )
        else:  # "any" mode
            approvals = [r for r in results if r.is_approved]
            if approvals:
                return approvals[0]
            return results[-1]  # Return last denial


class GuardedExecutor:
    """Executes actions only after guardian approval."""

    def __init__(self, pipeline: GuardianPipeline, executor: Callable):
        self.pipeline = pipeline
        self.executor = executor
        self.execution_log: list[dict] = []

    async def execute(self, action: ActionRequest) -> dict:
        """Validate and execute an action."""
        # Pre-execution validation
        result = await self.pipeline.validate(action)

        if result.decision == GuardianDecision.DENY:
            self.log_execution(action, result, executed=False)
            return {
                "success": False,
                "reason": result.reason,
                "violations": result.violations
            }

        if result.decision == GuardianDecision.ESCALATE:
            self.log_execution(action, result, executed=False, escalated=True)
            return {
                "success": False,
                "escalated": True,
                "reason": result.reason,
                "requires_human_review": True
            }

        # Execute with potentially modified parameters
        params = result.modified_action if result.modified_action else action.parameters

        try:
            execution_result = await self.executor(action.action_type, params)
            self.log_execution(action, result, executed=True,
                               execution_result=execution_result)
            return {
                "success": True,
                "result": execution_result,
                "modified": result.decision == GuardianDecision.MODIFY
            }
        except Exception as e:
            self.log_execution(action, result, executed=True, error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    def log_execution(self, action: ActionRequest, validation: ValidationResult,
                      executed: bool, **kwargs):
        self.execution_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": action.request_id,
            "action_type": action.action_type,
            "validation_decision": validation.decision.value,
            "executed": executed,
            **kwargs
        })


# =============================================================================
# Content Guardian
# =============================================================================

@dataclass
class ContentPolicy:
    """A content validation policy."""
    name: str
    description: str
    check_function: Callable[[str], tuple[bool, str]]
    severity: str = "medium"  # low, medium, high, critical


class ContentGuardian(Guardian):
    """Validates text content against policies."""

    def __init__(self, guardian_id: str, llm_client=None, policies: list[ContentPolicy] = None):
        if not guardian_id or not guardian_id.strip():
            raise ValueError("Guardian ID must not be empty")
        super().__init__(guardian_id)
        self.llm = llm_client
        self.policies = policies or self.default_policies()

    def default_policies(self) -> list[ContentPolicy]:
        return [
            ContentPolicy(
                name="no_pii",
                description="Content must not contain personally identifiable information",
                check_function=self.check_pii,
                severity="high"
            ),
            ContentPolicy(
                name="no_harmful",
                description="Content must not be harmful, hateful, or dangerous",
                check_function=self.check_harmful,
                severity="critical"
            ),
            ContentPolicy(
                name="professional_tone",
                description="Content should maintain professional tone",
                check_function=self.check_tone,
                severity="low"
            ),
        ]

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Validate content in the action."""
        content = self.extract_content(action)

        if not content:
            return ValidationResult(
                decision=GuardianDecision.APPROVE,
                reason="No content to validate"
            )

        violations = []

        # Rule-based checks
        for policy in self.policies:
            passed, message = policy.check_function(content)
            if not passed:
                violations.append({
                    "policy": policy.name,
                    "severity": policy.severity,
                    "message": message
                })

        # LLM-based content analysis for nuanced issues
        if self.llm:
            llm_analysis = await self.llm_content_check(content)
            if llm_analysis["issues"]:
                violations.extend(llm_analysis["issues"])

        # Determine decision based on violations
        if any(v["severity"] == "critical" for v in violations):
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason="Critical content policy violation",
                violations=[v["message"] for v in violations]
            )

        if any(v["severity"] == "high" for v in violations):
            return ValidationResult(
                decision=GuardianDecision.ESCALATE,
                reason="High severity content issue requires review",
                violations=[v["message"] for v in violations]
            )

        if violations:
            # Try to fix minor issues
            if self.llm:
                fixed_content = await self.fix_content(content, violations)
                if fixed_content:
                    modified_params = action.parameters.copy()
                    self.set_content(modified_params, fixed_content)
                    return ValidationResult(
                        decision=GuardianDecision.MODIFY,
                        reason="Minor issues auto-corrected",
                        modified_action=modified_params,
                        violations=[v["message"] for v in violations]
                    )

        return ValidationResult(
            decision=GuardianDecision.APPROVE,
            reason="Content passes all policy checks"
        )

    def extract_content(self, action: ActionRequest) -> str | None:
        """Extract text content from various action types."""
        params = action.parameters

        # Common content field names
        for field_name in ["content", "body", "text", "message", "description"]:
            if field_name in params:
                return params[field_name]

        return None

    def set_content(self, params: dict, content: str):
        """Set content back into parameters."""
        for field_name in ["content", "body", "text", "message", "description"]:
            if field_name in params:
                params[field_name] = content
                return

    def check_pii(self, content: str) -> tuple[bool, str]:
        """Check for personally identifiable information."""
        patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }

        for pii_type, pattern in patterns.items():
            if re.search(pattern, content):
                return False, f"Contains potential {pii_type}"

        return True, ""

    def check_harmful(self, content: str) -> tuple[bool, str]:
        """Check for harmful content patterns."""
        harmful_patterns = [
            (r"\b(kill|murder|attack)\s+(yourself|himself|herself|themselves)\b",
             "self-harm"),
            (r"\bhow\s+to\s+(make|build)\s+(bomb|weapon|explosive)\b",
             "violence"),
        ]

        for pattern, category in harmful_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, f"Potentially harmful content: {category}"

        return True, ""

    def check_tone(self, content: str) -> tuple[bool, str]:
        """Check for unprofessional tone."""
        profanity_indicators = ["damn", "hell", "crap"]  # Simplified

        content_lower = content.lower()
        for word in profanity_indicators:
            if word in content_lower:
                return False, "Unprofessional language detected"

        return True, ""

    async def llm_content_check(self, content: str) -> dict:
        """Use LLM for nuanced content analysis."""
        prompt = f"""Analyze this content for potential issues:

Content:
{content[:2000]}

Check for:
1. Harmful or dangerous instructions
2. Discriminatory or biased language
3. Misleading or false information
4. Inappropriate for professional context

Respond with JSON:
{{
    "safe": true/false,
    "issues": [
        {{"severity": "low/medium/high/critical", "message": "description"}}
    ],
    "explanation": "brief explanation"
}}"""

        response = await self.llm.complete([
            {"role": "system", "content": "You are a content safety analyst. Be thorough but avoid false positives."},
            {"role": "user", "content": prompt}
        ])

        try:
            result = json.loads(response.content)
            # Validate expected structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            result.setdefault("safe", True)
            result.setdefault("issues", [])
            result.setdefault("explanation", "")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            # Defensive fallback: assume safe but log the parsing failure
            logger.warning(f"Content analysis response parsing failed: {type(e).__name__}: {e}")
            return {"safe": True, "issues": [], "explanation": f"Response parsing failed: {e}"}

    async def fix_content(self, content: str, violations: list) -> str | None:
        """Attempt to fix minor content issues."""
        fixable_severities = ["low", "medium"]

        if not all(v["severity"] in fixable_severities for v in violations):
            return None

        prompt = f"""Fix these issues in the content while preserving meaning:

Content:
{content}

Issues to fix:
{json.dumps(violations, indent=2)}

Return only the fixed content, no explanation."""

        response = await self.llm.complete([
            {"role": "system", "content": "You fix content issues minimally and precisely."},
            {"role": "user", "content": prompt}
        ])

        return response.content


# =============================================================================
# Action Guardian
# =============================================================================

class ActionGuardian(Guardian):
    """Validates actions against operational policies."""

    def __init__(self, guardian_id: str, action_policies: dict[str, dict]):
        if not guardian_id or not guardian_id.strip():
            raise ValueError("Guardian ID must not be empty")
        super().__init__(guardian_id)
        self.action_policies = action_policies

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Validate action against its specific policy."""
        policy = self.action_policies.get(action.action_type)

        if not policy:
            # Unknown action type - deny by default for safety
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason=f"Unknown action type: {action.action_type}",
                violations=["Action type not in allowlist"]
            )

        violations = []

        # Check required parameters
        required = policy.get("required_params", [])
        for param in required:
            if param not in action.parameters:
                violations.append(f"Missing required parameter: {param}")

        # Check parameter constraints
        constraints = policy.get("constraints", {})
        for param, constraint in constraints.items():
            if param in action.parameters:
                value = action.parameters[param]
                violation = self.check_constraint(param, value, constraint)
                if violation:
                    violations.append(violation)

        # Check forbidden patterns
        forbidden = policy.get("forbidden_patterns", [])
        for pattern in forbidden:
            if self.matches_pattern(action.parameters, pattern):
                violations.append(f"Forbidden pattern detected: {pattern['description']}")

        # Check agent permissions
        allowed_agents = policy.get("allowed_agents", [])
        if allowed_agents and action.agent_id not in allowed_agents:
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason=f"Agent {action.agent_id} not authorized for {action.action_type}",
                violations=["Unauthorized agent"]
            )

        if violations:
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason="Action policy violations",
                violations=violations
            )

        return ValidationResult(
            decision=GuardianDecision.APPROVE,
            reason="Action passes all policy checks"
        )

    def check_constraint(self, param: str, value: Any, constraint: dict) -> str | None:
        """Check a value against a constraint."""
        if "type" in constraint:
            expected_type = constraint["type"]
            if expected_type == "string" and not isinstance(value, str):
                return f"{param} must be a string"
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return f"{param} must be a number"
            elif expected_type == "list" and not isinstance(value, list):
                return f"{param} must be a list"

        if "min" in constraint and value < constraint["min"]:
            return f"{param} must be at least {constraint['min']}"

        if "max" in constraint and value > constraint["max"]:
            return f"{param} must be at most {constraint['max']}"

        if "pattern" in constraint:
            if not re.match(constraint["pattern"], str(value)):
                return f"{param} does not match required pattern"

        if "enum" in constraint and value not in constraint["enum"]:
            return f"{param} must be one of {constraint['enum']}"

        return None

    def matches_pattern(self, params: dict, pattern: dict) -> bool:
        """Check if parameters match a forbidden pattern."""
        conditions = pattern.get("conditions", {})
        for key, expected in conditions.items():
            if key not in params:
                return False
            if params[key] != expected:
                return False
        return True


# =============================================================================
# Cost Guardian
# =============================================================================

class CostGuardian(Guardian):
    """Monitors and limits costs across the system."""

    def __init__(self, guardian_id: str, config: dict):
        if not guardian_id or not guardian_id.strip():
            raise ValueError("Guardian ID must not be empty")
        super().__init__(guardian_id, config)
        self.budgets = config.get("budgets", {})
        # Validate budget values are positive
        for budget_key, budget_value in self.budgets.items():
            if isinstance(budget_value, (int, float)) and budget_value < 0:
                raise ValueError(f"Budget value for '{budget_key}' must be non-negative")
        # Bound spending records to prevent unbounded memory growth
        max_records = config.get("max_spending_records", 10000)
        self.spending: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_records)
        )
        self.cost_calculator = config.get("cost_calculator", self.default_cost_calculator)

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Check if action is within budget limits.

        Three zones per limit:
          cost <= soft_mult * limit         -> approve (well under)
          soft_mult * limit < cost <= limit -> escalate (warning zone)
          cost > limit                      -> deny (hard breach)

        The same zones apply to per-action, per-agent, and global limits;
        we choose the most severe outcome across all three.
        """
        # Estimate cost
        estimated_cost = await self.estimate_cost(action)

        if estimated_cost == 0:
            return ValidationResult(
                decision=GuardianDecision.APPROVE,
                reason="No cost associated with action"
            )

        soft_mult = self.budgets.get("soft_limit_multiplier", 0.8)

        def classify(cost_after: float, limit: float) -> str:
            if cost_after > limit:
                return "breach"
            if cost_after > soft_mult * limit:
                return "warn"
            return "ok"

        violations: list[str] = []
        warnings: list[str] = []

        def record(zone: str, msg: str) -> None:
            (violations if zone == "breach" else warnings).append(msg)

        # Per-action limit
        key = f"per_action_{action.action_type}"
        default = self.budgets.get("per_action_default", float("inf"))
        action_limit = self.budgets.get(key, default)
        zone = classify(estimated_cost, action_limit)
        if zone != "ok":
            record(zone, (f"Per-action cost ${estimated_cost:.2f} "
                          f"is at {zone} of limit ${action_limit:.2f}"))

        # Per-agent limit (hourly)
        agent_spent = self.get_spending(action.agent_id, timedelta(hours=1))
        agent_limit = self.budgets.get("per_agent_hourly", float("inf"))
        zone = classify(agent_spent + estimated_cost, agent_limit)
        if zone != "ok":
            total = agent_spent + estimated_cost
            record(zone, (f"Agent {action.agent_id} would be at {zone} of "
                          f"hourly limit: ${total:.2f} vs ${agent_limit:.2f}"))

        # Global daily limit
        global_spent = self.get_spending("global", timedelta(days=1))
        global_limit = self.budgets.get("global_daily", float("inf"))
        zone = classify(global_spent + estimated_cost, global_limit)
        if zone != "ok":
            total = global_spent + estimated_cost
            record(zone, (f"Global daily spend would be at {zone}: "
                          f"${total:.2f} vs ${global_limit:.2f}"))

        if violations:
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason="Budget limit exceeded",
                violations=violations + warnings,
                metadata={"estimated_cost": estimated_cost},
            )
        if warnings:
            return ValidationResult(
                decision=GuardianDecision.ESCALATE,
                reason="Approaching budget limits",
                violations=warnings,
                metadata={"estimated_cost": estimated_cost},
            )

        # Record the anticipated spending
        self.record_spending(action.agent_id, estimated_cost, action)

        return ValidationResult(
            decision=GuardianDecision.APPROVE,
            reason=f"Within budget (estimated: ${estimated_cost:.2f})",
            metadata={"estimated_cost": estimated_cost}
        )

    async def estimate_cost(self, action: ActionRequest) -> float:
        """Estimate the cost of an action."""
        return self.cost_calculator(action)

    def default_cost_calculator(self, action: ActionRequest) -> float:
        """Default cost estimation based on action type.

        Note: Model prices below are approximate and may be outdated.
        In production, fetch current prices from provider APIs or config.
        """
        costs = {
            "llm_call": 0.01,
            "api_request": 0.001,
            "database_query": 0.0001,
            "send_email": 0.002,
            "web_search": 0.005,
        }

        base_cost = costs.get(action.action_type, 0)

        # Adjust for parameters
        if "model" in action.parameters:
            model_multipliers = {
                "gpt-4o": 5,
                "gpt-4o-mini": 1,
                "gpt-3.5-turbo": 1,
                "claude-opus-4": 10,
                "claude-sonnet-4": 3,
                "claude-haiku-3": 0.5
            }
            base_cost *= model_multipliers.get(action.parameters["model"], 1)

        if "tokens" in action.parameters:
            base_cost *= action.parameters["tokens"] / 1000

        return base_cost

    def record_spending(self, agent_id: str, amount: float, action: ActionRequest):
        """Record spending for tracking."""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "amount": amount,
            "action_type": action.action_type,
            "request_id": action.request_id
        }
        self.spending[agent_id].append(record)
        self.spending["global"].append(record)

    def get_spending(self, agent_id: str, window: timedelta) -> float:
        """Get total spending within a time window."""
        cutoff = datetime.now(timezone.utc) - window
        records = self.spending.get(agent_id, [])
        return sum(r["amount"] for r in records if r["timestamp"] > cutoff)

    def get_budget_report(self) -> dict:
        """Generate a budget utilization report."""
        now = datetime.now(timezone.utc)

        return {
            "global_daily": {
                "spent": self.get_spending("global", timedelta(days=1)),
                "limit": self.budgets.get("global_daily", float("inf")),
            },
            "by_agent": {
                agent_id: {
                    "hourly": self.get_spending(agent_id, timedelta(hours=1)),
                    "daily": self.get_spending(agent_id, timedelta(days=1)),
                }
                for agent_id in set(self.spending.keys()) - {"global"}
            },
            "timestamp": now.isoformat()
        }


# =============================================================================
# Security Guardian
# =============================================================================

class SecurityGuardian(Guardian):
    """Enforces security policies and detects threats.

    Supports optional metrics export for SIEM integration and observability.

    Args:
        guardian_id: Unique identifier for this guardian
        config: Configuration dict with keys:
            - permissions: Dict mapping agent_id to list of allowed actions
            - blocked_patterns: List of {name, regex} patterns to block
            - rate_limits: Dict mapping action_type to {window_seconds, max_requests}
            - max_request_history: Max entries in request history (default 1000)
        metrics_exporter: Optional MetricsExporter for external monitoring

    Example:
        from guardian import SecurityGuardian, InMemoryMetricsExporter

        exporter = InMemoryMetricsExporter()
        guardian = SecurityGuardian(
            "security",
            config={...},
            metrics_exporter=exporter
        )

        # Later, check metrics
        metrics = exporter.get_metrics()
        print(f"Blocked: {metrics.actions_blocked}")
    """

    def __init__(
        self,
        guardian_id: str,
        config: dict,
        metrics_exporter: MetricsExporter | None = None
    ):
        super().__init__(guardian_id, config)
        self.permissions = config.get("permissions", {})
        self.blocked_patterns = config.get("blocked_patterns", [])
        self.rate_limits = config.get("rate_limits", {})
        # Validate rate limits are positive
        for action_type, limit_config in self.rate_limits.items():
            if isinstance(limit_config, dict):
                window = limit_config.get("window_seconds")
                max_req = limit_config.get("max_requests")
                if window is not None and window <= 0:
                    raise ValueError(f"Rate limit window_seconds for '{action_type}' must be positive")
                if max_req is not None and max_req <= 0:
                    raise ValueError(f"Rate limit max_requests for '{action_type}' must be positive")
        # Bound request history to prevent unbounded memory growth
        max_history = config.get("max_request_history", 1000)
        self.request_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # Optional metrics exporter for SIEM/monitoring integration
        self._metrics_exporter = metrics_exporter

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Comprehensive security validation."""
        violations = []

        # Permission check
        perm_result = self.check_permissions(action)
        if perm_result:
            violations.append(perm_result)

        # Input sanitization
        sanitization_result = self.check_inputs(action)
        violations.extend(sanitization_result)

        # Rate limiting
        rate_result = self.check_rate_limit(action)
        if rate_result:
            violations.append(rate_result)

        # Injection detection
        injection_result = self.detect_injection(action)
        if injection_result:
            violations.append(injection_result)

        # Sensitive data access
        sensitive_result = self.check_sensitive_access(action)
        if sensitive_result:
            violations.append(sensitive_result)

        if violations:
            severity = self.assess_severity(violations)
            if severity == "critical":
                # Log security event
                await self.log_security_event(action, violations, "critical")
                return ValidationResult(
                    decision=GuardianDecision.DENY,
                    reason="Security policy violation",
                    violations=violations
                )
            elif severity == "high":
                return ValidationResult(
                    decision=GuardianDecision.ESCALATE,
                    reason="Security concern requires review",
                    violations=violations
                )

        return ValidationResult(
            decision=GuardianDecision.APPROVE,
            reason="Security checks passed"
        )

    def check_permissions(self, action: ActionRequest) -> str | None:
        """Verify agent has permission for this action."""
        agent_perms = self.permissions.get(action.agent_id, [])
        required_perm = f"{action.action_type}:execute"

        if "*" in agent_perms:
            return None  # Superuser

        if required_perm not in agent_perms and action.action_type not in agent_perms:
            return f"Agent lacks permission: {required_perm}"

        return None

    def check_inputs(self, action: ActionRequest) -> list[str]:
        """Sanitize and validate inputs."""
        violations = []

        for key, value in action.parameters.items():
            if isinstance(value, str):
                # Check for dangerous patterns
                for pattern in self.blocked_patterns:
                    if re.search(pattern["regex"], value, re.IGNORECASE):
                        violations.append(f"Blocked pattern in {key}: {pattern['name']}")

        return violations

    def check_rate_limit(self, action: ActionRequest) -> str | None:
        """Enforce rate limits."""
        key = f"{action.agent_id}:{action.action_type}"
        limit_config = self.rate_limits.get(action.action_type,
                                            self.rate_limits.get("default", {}))

        if not limit_config:
            return None

        window = timedelta(seconds=limit_config.get("window_seconds", 60))
        max_requests = limit_config.get("max_requests", 100)

        now = datetime.now(timezone.utc)
        cutoff = now - window

        # Clean old requests while preserving deque maxlen
        max_history = self.config.get("max_request_history", 1000)
        filtered = [t for t in self.request_history[key] if t > cutoff]
        self.request_history[key] = deque(filtered, maxlen=max_history)

        if len(self.request_history[key]) >= max_requests:
            return f"Rate limit exceeded: {max_requests} requests per {window.seconds}s"

        self.request_history[key].append(now)
        return None

    def detect_injection(self, action: ActionRequest) -> str | None:
        """Detect potential injection attacks."""
        injection_patterns = [
            (r";\s*DROP\s+TABLE", "SQL injection attempt"),
            (r"<script[^>]*>", "XSS attempt"),
            (r"\$\{.*\}", "Template injection attempt"),
            (r"__import__|eval|exec", "Code injection attempt"),
            (r"\.\./\.\./", "Path traversal attempt"),
        ]

        params_str = json.dumps(action.parameters)

        for pattern, name in injection_patterns:
            if re.search(pattern, params_str, re.IGNORECASE):
                return f"Potential {name} detected"

        return None

    def check_sensitive_access(self, action: ActionRequest) -> str | None:
        """Check for unauthorized sensitive data access."""
        sensitive_fields = ["password", "secret", "token", "api_key", "private_key"]

        # Check if action reads sensitive data
        if action.action_type in ["read_file", "database_query", "api_call"]:
            params_str = json.dumps(action.parameters).lower()
            for field_name in sensitive_fields:
                if field_name in params_str:
                    return f"Accessing sensitive field: {field_name}"

        return None

    def assess_severity(self, violations: list[str]) -> str:
        """Assess overall severity of violations."""
        critical_keywords = ["injection", "traversal", "unauthorized"]
        high_keywords = ["rate limit", "sensitive", "blocked"]

        for v in violations:
            v_lower = v.lower()
            if any(k in v_lower for k in critical_keywords):
                return "critical"

        for v in violations:
            v_lower = v.lower()
            if any(k in v_lower for k in high_keywords):
                return "high"

        return "medium"

    async def log_security_event(self, action: ActionRequest,
                                 violations: list[str], severity: str):
        """Log security events for monitoring and export to metrics system.

        Records the event to:
        1. Local audit_log (always)
        2. External metrics exporter if configured (SIEM, Prometheus, etc.)

        Args:
            action: The action request that triggered the security event
            violations: List of violation descriptions
            severity: Event severity ('low', 'medium', 'high', 'critical')
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "agent_id": action.agent_id,
            "action_type": action.action_type,
            "violations": violations,
            "request_id": action.request_id,
            "parameters_hash": hashlib.sha256(
                json.dumps(action.parameters, sort_keys=True).encode()
            ).hexdigest()
        }

        # Always log to local audit log
        self.audit_log.append(event)

        # Export to external metrics/SIEM system if configured
        if self._metrics_exporter:
            try:
                await self._metrics_exporter.record_security_event(event)
                # Also increment the blocked actions counter
                await self._metrics_exporter.increment_counter(
                    "actions_blocked",
                    {"severity": severity, "action_type": action.action_type}
                )
            except Exception as e:
                # Don't let metrics export failure affect security operations
                logger.warning(f"Failed to export security metrics: {e}")

    async def record_approval(self, action: ActionRequest):
        """Record an approved action for metrics tracking.

        Call this when an action passes all security checks.
        """
        if self._metrics_exporter:
            try:
                await self._metrics_exporter.increment_counter(
                    "actions_approved",
                    {"action_type": action.action_type, "agent_id": action.agent_id}
                )
            except Exception as e:
                logger.warning(f"Failed to export approval metrics: {e}")

    def get_metrics_summary(self) -> dict:
        """Get a summary of security metrics from the audit log.

        Returns aggregated metrics suitable for dashboards or reporting.
        For real-time metrics, use a MetricsExporter implementation.
        """
        events = list(self.audit_log)
        summary = {
            "total_events": len(events),
            "events_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "violations_by_type": {},
            "top_agents": {},
            "top_action_types": {}
        }

        for event in events:
            # Count by severity
            severity = event.get("severity", "medium")
            if severity in summary["events_by_severity"]:
                summary["events_by_severity"][severity] += 1

            # Count violations by type
            for violation in event.get("violations", []):
                vtype = self._categorize_violation(violation)
                summary["violations_by_type"][vtype] = (
                    summary["violations_by_type"].get(vtype, 0) + 1
                )

            # Count by agent
            agent_id = event.get("agent_id", "unknown")
            summary["top_agents"][agent_id] = (
                summary["top_agents"].get(agent_id, 0) + 1
            )

            # Count by action type
            action_type = event.get("action_type", "unknown")
            summary["top_action_types"][action_type] = (
                summary["top_action_types"].get(action_type, 0) + 1
            )

        return summary

    def _categorize_violation(self, violation: str) -> str:
        """Categorize a violation message into a type."""
        violation_lower = violation.lower()
        if "injection" in violation_lower:
            return "injection"
        elif "rate limit" in violation_lower:
            return "rate_limit"
        elif "permission" in violation_lower:
            return "permission"
        elif "sensitive" in violation_lower:
            return "sensitive_data"
        elif "traversal" in violation_lower:
            return "path_traversal"
        elif "blocked pattern" in violation_lower:
            return "blocked_pattern"
        else:
            return "other"

    def export_events_siem_format(self, since: datetime | None = None) -> list[dict]:
        """Export security events in SIEM-compatible format.

        Returns events formatted for ingestion by common SIEM systems.
        Format follows CEF (Common Event Format) conventions.

        Args:
            since: Optional datetime to filter events after this time

        Returns:
            List of SIEM-formatted event dicts

        Example:
            events = guardian.export_events_siem_format(
                since=datetime.now(timezone.utc) - timedelta(hours=1)
            )
            for event in events:
                siem_client.send(event)
        """
        events = []
        for audit_event in self.audit_log:
            # Parse timestamp for filtering
            event_time = datetime.fromisoformat(audit_event["timestamp"])
            if since and event_time < since:
                continue

            # Map severity to CEF severity (0-10 scale)
            severity_map = {"low": 2, "medium": 5, "high": 7, "critical": 10}
            cef_severity = severity_map.get(audit_event.get("severity", "medium"), 5)

            siem_event = {
                # CEF header fields
                "cef_version": "0",
                "device_vendor": "AgentBook",
                "device_product": "Guardian",
                "device_version": "1.0",
                "signature_id": f"GUARD-{audit_event.get('action_type', 'UNKNOWN')}",
                "name": f"Security Event: {audit_event.get('action_type', 'unknown')}",
                "severity": cef_severity,

                # Extension fields
                "timestamp": audit_event["timestamp"],
                "source_user_id": audit_event.get("agent_id", "unknown"),
                "request_id": audit_event.get("request_id", ""),
                "action": audit_event.get("action_type", "unknown"),
                "outcome": "blocked",
                "reason": "; ".join(audit_event.get("violations", [])),
                "parameters_hash": audit_event.get("parameters_hash", ""),

                # Custom fields
                "violation_count": len(audit_event.get("violations", [])),
                "guardian_id": self.id
            }
            events.append(siem_event)

        return events


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """States for circuit breaker."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking all requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 3      # Successes to close from half-open
    timeout_seconds: float = 60.0   # Time before trying half-open

    def __post_init__(self):
        """Validate configuration values."""
        if self.failure_threshold <= 0:
            raise ValueError("Failure threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("Success threshold must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout seconds must be positive")


class CircuitBreaker:
    """Circuit breaker for system protection.

    Note: For concurrent async environments, wrap state mutations
    with asyncio.Lock to ensure thread safety.
    """

    def __init__(self, name: str, config: CircuitConfig = None):
        if not name or not name.strip():
            raise ValueError("Circuit breaker name must not be empty")
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.state_change_callbacks: list[Callable] = []
        self._lock = asyncio.Lock()  # Thread safety for concurrent access

    async def can_execute(self) -> bool:
        """Check if execution is allowed (async for lock safety)."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self.last_failure_time:
                    elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        self.transition_to(CircuitState.HALF_OPEN)
                        return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return True

            return False

    async def record_success(self):
        """Record a successful execution (async for lock safety)."""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.transition_to(CircuitState.CLOSED)

    async def record_failure(self):
        """Record a failed execution (async for lock safety)."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            self.success_count = 0

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.transition_to(CircuitState.OPEN)

            elif self.state == CircuitState.HALF_OPEN:
                self.transition_to(CircuitState.OPEN)

    def transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state

        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.OPEN:
            self.success_count = 0  # Reset for next recovery attempt

        for callback in self.state_change_callbacks:
            callback(self.name, old_state, new_state)

    def force_open(self):
        """Emergency kill switch - force circuit open."""
        self.transition_to(CircuitState.OPEN)
        self.last_failure_time = datetime.now(timezone.utc) + timedelta(hours=24)  # Stay open

    def force_close(self):
        """Manual override - force circuit closed."""
        self.transition_to(CircuitState.CLOSED)


class CircuitBreakerGuardian(Guardian):
    """Guardian that implements circuit breaker pattern."""

    def __init__(self, guardian_id: str, breakers: dict[str, CircuitBreaker] = None,
                 config: GuardianConfig = None):
        if not guardian_id or not guardian_id.strip():
            raise ValueError("Guardian ID must not be empty")
        super().__init__(guardian_id)
        self.guardian_config = config or GuardianConfig()
        self.breakers = breakers or {}
        self.global_breaker = CircuitBreaker("global", CircuitConfig(
            failure_threshold=self.guardian_config.circuit_breaker_threshold,
            timeout_seconds=self.guardian_config.escalation_timeout
        ))

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Check circuit breakers before allowing action."""
        # Check global breaker
        if not await self.global_breaker.can_execute():
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason="System circuit breaker is open - operations suspended",
                violations=["Global circuit breaker tripped"]
            )

        # Check action-specific breaker
        breaker = self.breakers.get(action.action_type)
        if breaker and not await breaker.can_execute():
            return ValidationResult(
                decision=GuardianDecision.DENY,
                reason=f"Circuit breaker for {action.action_type} is open",
                violations=[f"{action.action_type} circuit breaker tripped"]
            )

        return ValidationResult(
            decision=GuardianDecision.APPROVE,
            reason="Circuit breakers allow execution",
            metadata={"circuit_state": self.global_breaker.state.value}
        )

    async def record_outcome(self, action: ActionRequest, success: bool):
        """Record execution outcome to update circuit breakers."""
        breaker = self.breakers.get(action.action_type)

        if success:
            await self.global_breaker.record_success()
            if breaker:
                await breaker.record_success()
        else:
            await self.global_breaker.record_failure()
            if breaker:
                await breaker.record_failure()

    def emergency_stop(self):
        """Emergency shutdown - open all breakers."""
        self.global_breaker.force_open()
        for breaker in self.breakers.values():
            breaker.force_open()


# =============================================================================
# Defense in Depth Pipeline Factory
# =============================================================================

def create_defense_in_depth_pipeline(llm_client, config: dict) -> GuardianPipeline:
    """Create a multi-layered guardian pipeline."""

    # Layer 1: Circuit breakers (fastest, system-level)
    circuit_guardian = CircuitBreakerGuardian(
        "circuit_breaker",
        breakers={
            "llm_call": CircuitBreaker("llm", CircuitConfig(failure_threshold=3)),
            "api_request": CircuitBreaker("api", CircuitConfig(failure_threshold=5)),
            "database_query": CircuitBreaker("db", CircuitConfig(failure_threshold=3)),
        }
    )

    # Layer 2: Security (authentication, authorization, injection)
    security_guardian = SecurityGuardian(
        "security",
        config={
            "permissions": config.get("agent_permissions", {}),
            "blocked_patterns": [
                {"name": "sql_injection", "regex": r";\s*(DROP|DELETE|UPDATE|INSERT)"},
                {"name": "command_injection", "regex": r"[;&|`$]"},
                {"name": "path_traversal", "regex": r"\.\.[\\/]"},
            ],
            "rate_limits": {
                "default": {"window_seconds": 60, "max_requests": 100},
                "llm_call": {"window_seconds": 60, "max_requests": 20},
            }
        }
    )

    # Layer 3: Cost controls
    cost_guardian = CostGuardian(
        "cost",
        config={
            "budgets": {
                "per_action_default": 1.00,
                "per_action_llm_call": 0.50,
                "per_agent_hourly": 10.00,
                "global_daily": 100.00,
                "soft_limit_multiplier": 0.8
            }
        }
    )

    # Layer 4: Action-specific policies
    action_guardian = ActionGuardian(
        "action_policy",
        action_policies=config.get("action_policies", {})
    )

    # Layer 5: Content moderation (slowest, most thorough)
    content_guardian = ContentGuardian(
        "content",
        llm_client=llm_client
    )

    return GuardianPipeline(
        guardians=[
            circuit_guardian,
            security_guardian,
            cost_guardian,
            action_guardian,
            content_guardian
        ],
        mode="all"
    )


# =============================================================================
# Resilient Pipeline with Failure Handling
# =============================================================================

class ResilientGuardianPipeline(GuardianPipeline):
    """Guardian pipeline with failure handling."""

    def __init__(self, guardians: list[Guardian], mode: str = "all",
                 fallback_decision: GuardianDecision = GuardianDecision.DENY):
        super().__init__(guardians, mode)
        self.fallback_decision = fallback_decision
        self.guardian_health: dict[str, bool] = {g.id: True for g in guardians}

    async def validate(self, action: ActionRequest) -> ValidationResult:
        """Validate with individual guardian failure handling."""
        results = []

        for guardian in self.guardians:
            if not self.guardian_health[guardian.id]:
                # Skip unhealthy guardians
                continue

            try:
                result = await asyncio.wait_for(
                    guardian.validate(action),
                    timeout=5.0  # Individual guardian timeout
                )
                results.append(result)
                guardian.log_decision(action, result)

            except asyncio.TimeoutError:
                self.mark_unhealthy(guardian.id)
                results.append(self.create_fallback_result(
                    f"Guardian {guardian.id} timed out"
                ))

            except Exception as e:
                self.mark_unhealthy(guardian.id)
                results.append(self.create_fallback_result(
                    f"Guardian {guardian.id} error: {str(e)}"
                ))

        return self.aggregate_results(results, action)

    async def shutdown(self):
        """Cancel all pending health check tasks on shutdown."""
        if hasattr(self, '_health_check_tasks'):
            for task in self._health_check_tasks:
                task.cancel()
            await asyncio.gather(*self._health_check_tasks, return_exceptions=True)
            self._health_check_tasks.clear()

    def mark_unhealthy(self, guardian_id: str):
        """Mark a guardian as unhealthy."""
        self.guardian_health[guardian_id] = False
        # Schedule health check, tracking task to avoid warnings
        if not hasattr(self, '_health_check_tasks'):
            self._health_check_tasks = []
        task = asyncio.create_task(self.health_check_later(guardian_id))
        self._health_check_tasks.append(task)

    async def health_check_later(self, guardian_id: str, delay: float = 60.0):
        """Re-enable guardian after delay if healthy."""
        await asyncio.sleep(delay)

        guardian = next((g for g in self.guardians if g.id == guardian_id), None)
        if not guardian:
            return

        # Simple health check action
        test_action = ActionRequest(
            action_type="health_check",
            parameters={},
            agent_id="system"
        )

        try:
            await asyncio.wait_for(
                guardian.validate(test_action),
                timeout=2.0
            )
            self.guardian_health[guardian_id] = True
        except (asyncio.TimeoutError, asyncio.CancelledError, ValidationError):
            # Still unhealthy, check again later with tracked task
            task = asyncio.create_task(self.health_check_later(guardian_id, delay * 2))
            self._health_check_tasks.append(task)
        except Exception as e:
            # Log unexpected errors but don't crash the health check loop
            logger.warning(f"Unexpected error in health check for {guardian_id}: {e}")
            task = asyncio.create_task(self.health_check_later(guardian_id, delay * 2))
            self._health_check_tasks.append(task)

    def create_fallback_result(self, reason: str) -> ValidationResult:
        """Create a fallback result when guardian fails."""
        return ValidationResult(
            decision=self.fallback_decision,
            reason=reason,
            violations=["Guardian unavailable"],
            confidence=0.0  # Low confidence in fallback
        )

    def aggregate_results(self, results: list[ValidationResult],
                          action: ActionRequest) -> ValidationResult:
        """Aggregate results from multiple guardians."""
        if not results:
            return self.create_fallback_result("No guardians available")

        # Check for escalations first
        for result in results:
            if result.decision == GuardianDecision.ESCALATE:
                return result

        # In "all" mode, any denial stops
        if self.mode == "all":
            for result in results:
                if result.decision == GuardianDecision.DENY:
                    return result

        # All approved
        all_violations = []
        for r in results:
            all_violations.extend(r.violations)

        modified = any(r.decision == GuardianDecision.MODIFY for r in results)

        return ValidationResult(
            decision=GuardianDecision.MODIFY if modified else GuardianDecision.APPROVE,
            reason="All guardians approved",
            violations=all_violations
        )


# =============================================================================
# Failure Policy Enum
# =============================================================================

class FailurePolicy(Enum):
    """How to handle guardian failures."""
    FAIL_CLOSED = "fail_closed"  # Deny on guardian failure (safer)
    FAIL_OPEN = "fail_open"      # Allow on guardian failure (more available)
    ESCALATE = "escalate"        # Escalate on guardian failure


def create_production_pipeline(
    guardians: list[Guardian],
    failure_policy: FailurePolicy = FailurePolicy.FAIL_CLOSED
) -> ResilientGuardianPipeline:
    """Create a production-capable pipeline with appropriate failure handling."""

    fallback_decisions = {
        FailurePolicy.FAIL_CLOSED: GuardianDecision.DENY,
        FailurePolicy.FAIL_OPEN: GuardianDecision.APPROVE,
        FailurePolicy.ESCALATE: GuardianDecision.ESCALATE,
    }

    return ResilientGuardianPipeline(
        guardians=guardians,
        mode="all",
        fallback_decision=fallback_decisions[failure_policy]
    )


# =============================================================================
# Human-in-the-Loop Integration
# =============================================================================

@dataclass
class EscalationTicket:
    """A ticket for human review."""
    ticket_id: str
    action: ActionRequest
    guardian_id: str
    reason: str
    violations: list[str]
    created_at: datetime
    status: str = "pending"  # pending, approved, denied, expired
    reviewed_by: str | None = None
    review_notes: str | None = None


class EscalationManager:
    """Manages escalated actions requiring human review."""

    def __init__(self, notification_service, timeout_hours: int = 24):
        if timeout_hours <= 0:
            raise ValueError("Timeout hours must be positive")
        self.tickets: dict[str, EscalationTicket] = {}
        self.notification_service = notification_service
        self.timeout_hours = timeout_hours
        self.pending_actions: dict[str, asyncio.Future] = {}
        self._timeout_tasks: dict[str, asyncio.Task] = {}

    async def escalate(self, action: ActionRequest, guardian_id: str,
                       result: ValidationResult) -> EscalationTicket:
        """Create an escalation ticket and wait for resolution."""
        ticket = EscalationTicket(
            ticket_id=f"ESC-{action.request_id}",
            action=action,
            guardian_id=guardian_id,
            reason=result.reason,
            violations=result.violations,
            created_at=datetime.now(timezone.utc)
        )

        self.tickets[ticket.ticket_id] = ticket

        # Notify reviewers
        await self.notification_service.notify_escalation(ticket)

        # Create a future for the resolution
        future = asyncio.get_running_loop().create_future()
        self.pending_actions[ticket.ticket_id] = future

        # Set timeout with tracked task reference
        self._timeout_tasks[ticket.ticket_id] = asyncio.create_task(
            self.timeout_ticket(ticket.ticket_id)
        )

        return ticket

    async def await_resolution(self, ticket_id: str) -> ValidationResult:
        """Wait for human resolution of an escalation."""
        if ticket_id not in self.pending_actions:
            raise ValueError(f"Unknown ticket: {ticket_id}")

        decision = await self.pending_actions[ticket_id]
        return decision

    async def resolve(self, ticket_id: str, approved: bool,
                      reviewer: str, notes: str = "") -> ValidationResult:
        """Human resolves an escalation."""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            raise ValueError(f"Unknown ticket: {ticket_id}")

        if ticket.status != "pending":
            raise ValueError(f"Ticket already resolved: {ticket.status}")

        ticket.status = "approved" if approved else "denied"
        ticket.reviewed_by = reviewer
        ticket.review_notes = notes

        result = ValidationResult(
            decision=GuardianDecision.APPROVE if approved else GuardianDecision.DENY,
            reason=f"Human review by {reviewer}: {notes}",
            violations=ticket.violations,
            metadata={"escalation_ticket": ticket_id, "reviewer": reviewer}
        )

        # Resolve the waiting future
        if ticket_id in self.pending_actions:
            self.pending_actions[ticket_id].set_result(result)

        return result

    async def timeout_ticket(self, ticket_id: str):
        """Timeout an unresolved ticket."""
        try:
            await asyncio.sleep(self.timeout_hours * 3600)

            ticket = self.tickets.get(ticket_id)
            if ticket and ticket.status == "pending":
                ticket.status = "expired"

                result = ValidationResult(
                    decision=GuardianDecision.DENY,
                    reason="Escalation timed out without review",
                    violations=["Review timeout"]
                )

                if ticket_id in self.pending_actions:
                    self.pending_actions[ticket_id].set_result(result)
        finally:
            # Clean up task reference to prevent memory growth
            self._timeout_tasks.pop(ticket_id, None)


class GuardedExecutorWithEscalation(GuardedExecutor):
    """Executor that handles escalations to humans."""

    def __init__(self, pipeline: GuardianPipeline, executor: Callable,
                 escalation_manager: EscalationManager):
        super().__init__(pipeline, executor)
        self.escalation_manager = escalation_manager

    async def execute(self, action: ActionRequest) -> dict:
        result = await self.pipeline.validate(action)

        if result.decision == GuardianDecision.ESCALATE:
            # Create escalation and wait
            ticket = await self.escalation_manager.escalate(
                action,
                "pipeline",
                result
            )

            # Wait for human decision (or timeout)
            human_result = await self.escalation_manager.await_resolution(
                ticket.ticket_id
            )

            if human_result.decision == GuardianDecision.APPROVE:
                # Human approved, proceed with execution
                return await self.execute_approved(action)
            else:
                return {
                    "success": False,
                    "reason": human_result.reason,
                    "escalation_ticket": ticket.ticket_id
                }

        # Normal flow for non-escalated actions
        return await super().execute(action)

    async def execute_approved(self, action: ActionRequest) -> dict:
        """Execute an action that was approved after escalation."""
        try:
            execution_result = await self.executor(action.action_type, action.parameters)
            return {
                "success": True,
                "result": execution_result,
                "escalated": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "escalated": True
            }


# =============================================================================
# Monitoring
# =============================================================================

class GuardianMonitor:
    """Monitors guardian activity and health."""

    def __init__(self, guardians: list[Guardian], config: GuardianConfig = None):
        self.guardians = {g.id: g for g in guardians}
        self.metrics: dict[str, list] = defaultdict(list)
        self.config = config or GuardianConfig()

    def collect_metrics(self) -> dict:
        """Collect metrics from all guardians."""
        metrics = {}

        for guardian_id, guardian in self.guardians.items():
            logs = guardian.audit_log
            recent = [log for log in logs if self.is_recent(log["timestamp"])]

            metrics[guardian_id] = {
                "total_validations": len(recent),
                "approvals": sum(1 for log in recent if log["decision"] == "approve"),
                "denials": sum(1 for log in recent if log["decision"] == "deny"),
                "escalations": sum(1 for log in recent if log["decision"] == "escalate"),
                "denial_rate": self.calculate_rate(recent, "deny"),
                "top_violations": self.get_top_violations(recent),
            }

        return metrics

    def is_recent(self, timestamp: str, hours: int = 1) -> bool:
        dt = datetime.fromisoformat(timestamp)
        return (datetime.now(timezone.utc) - dt).total_seconds() < hours * 3600

    def calculate_rate(self, logs: list, decision: str) -> float:
        if not logs:
            return 0.0
        return sum(1 for log in logs if log["decision"] == decision) / len(logs)

    def get_top_violations(self, logs: list, top_n: int = 5) -> list:
        from collections import Counter
        violations = []
        for log in logs:
            violations.extend(log.get("violations", []))
        return Counter(violations).most_common(top_n)

    def generate_alert(self, metrics: dict) -> list[str]:
        """Generate alerts based on metrics."""
        alerts = []

        for guardian_id, m in metrics.items():
            # High denial rate
            if m["denial_rate"] > 0.3:
                alerts.append(
                    f"High denial rate ({m['denial_rate']:.1%}) for {guardian_id}"
                )

            # Spike in escalations
            if m["escalations"] > self.config.alert_threshold:
                alerts.append(
                    f"Escalation spike ({m['escalations']}) for {guardian_id}"
                )

        return alerts


def create_guardian_dashboard(monitor: GuardianMonitor) -> dict:
    """Create a dashboard view of guardian status."""
    metrics = monitor.collect_metrics()
    alerts = monitor.generate_alert(metrics)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_health": "healthy" if not alerts else "degraded",
        "alerts": alerts,
        "guardian_metrics": metrics,
        "summary": {
            "total_validations": sum(m["total_validations"] for m in metrics.values()),
            "total_denials": sum(m["denials"] for m in metrics.values()),
            "total_escalations": sum(m["escalations"] for m in metrics.values()),
        }
    }


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate guardian pattern."""

    print("=" * 60)
    print("Guardian Pattern Demo")
    print("=" * 60)

    # Create a simple executor
    async def execute_action(action_type: str, params: dict) -> dict:
        return {"status": "success", "action": action_type}

    # Create pipeline with defense in depth
    pipeline = create_defense_in_depth_pipeline(
        llm_client=None,  # No LLM for demo
        config={
            "agent_permissions": {
                "customer_service_agent": [
                    "lookup_customer",
                    "process_refund",
                    "send_email",
                    "escalate_to_human"
                ]
            },
            "action_policies": {
                "process_refund": {
                    "required_params": ["order_id", "amount", "reason"],
                    "constraints": {
                        "amount": {"type": "number", "min": 0, "max": 500},
                        "reason": {"type": "string", "enum": [
                            "defective", "wrong_item", "not_as_described",
                            "customer_request", "duplicate_charge"
                        ]}
                    },
                    "allowed_agents": ["customer_service_agent"]
                }
            }
        }
    )

    executor = GuardedExecutor(pipeline, execute_action)

    # Test cases
    test_cases = [
        # Normal refund request
        ActionRequest(
            action_type="process_refund",
            parameters={
                "order_id": "ORD-12345678",
                "amount": 49.99,
                "reason": "defective"
            },
            agent_id="customer_service_agent"
        ),
        # Excessive amount
        ActionRequest(
            action_type="process_refund",
            parameters={
                "order_id": "ORD-12345678",
                "amount": 1500.00,
                "reason": "customer_request"
            },
            agent_id="customer_service_agent"
        ),
        # Unauthorized agent
        ActionRequest(
            action_type="process_refund",
            parameters={
                "order_id": "ORD-12345678",
                "amount": 50.00,
                "reason": "defective"
            },
            agent_id="analytics_agent"
        ),
    ]

    print("\nValidating actions...")
    for action in test_cases:
        result = await executor.execute(action)
        status = "SUCCESS" if result.get("success") else "BLOCKED"
        print(f"\n[{status}] {action.action_type} by {action.agent_id}")
        if not result.get("success"):
            print(f"  Reason: {result.get('reason', 'Unknown')}")
            if result.get("violations"):
                for v in result["violations"]:
                    print(f"  Violation: {v}")


if __name__ == "__main__":
    asyncio.run(main())
