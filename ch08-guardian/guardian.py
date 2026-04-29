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
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import uuid


# =============================================================================
# Severity and Policy Types
# =============================================================================

class Severity(int, Enum):
    """Severity levels for policy violations"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ActionCategory(str, Enum):
    """Categories of actions that can be validated"""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    EXECUTE_CODE = "execute_code"
    EXTERNAL_API = "external_api"
    FINANCIAL = "financial"
    USER_COMMUNICATION = "user_communication"
    SYSTEM_CONFIG = "system_config"
    NETWORK = "network"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Action:
    """An action proposed by an agent"""
    id: str
    agent_id: str
    category: ActionCategory
    operation: str
    target: str
    parameters: dict
    context: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "category": self.category.value,
            "operation": self.operation,
            "target": self.target,
            "parameters": self.parameters,
            "context": self.context,
            "timestamp": self.timestamp
        }


@dataclass
class PolicyViolation:
    """A detected policy violation"""
    policy_name: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "severity": self.severity.name,
            "message": self.message,
            "details": self.details
        }


@dataclass
class ValidationResult:
    """Result of action validation"""
    approved: bool
    action_id: str
    violations: list[PolicyViolation] = field(default_factory=list)
    warnings: list[PolicyViolation] = field(default_factory=list)
    reason: str | None = None
    conditions: list[str] = field(default_factory=list)
    expires_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "action_id": self.action_id,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "reason": self.reason,
            "conditions": self.conditions,
            "expires_at": self.expires_at
        }


@dataclass
class TrustScore:
    """Trust score for an agent"""
    agent_id: str
    score: float  # 0.0 to 1.0
    successful_actions: int
    failed_actions: int
    violations: int
    last_updated: str

    @property
    def autonomy_level(self) -> int:
        """Calculate autonomy level (1-5) based on trust score"""
        if self.score >= 0.9:
            return 5  # Full autonomy
        elif self.score >= 0.75:
            return 4  # High autonomy
        elif self.score >= 0.5:
            return 3  # Medium autonomy
        elif self.score >= 0.25:
            return 2  # Low autonomy
        else:
            return 1  # Minimal autonomy, human approval required


# =============================================================================
# Policy Definitions
# =============================================================================

class Policy:
    """Base class for policies"""

    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        """Check if action violates this policy"""
        raise NotImplementedError


class RateLimitPolicy(Policy):
    """Limits the rate of actions by an agent"""

    def __init__(
        self,
        name: str = "rate_limit",
        max_actions: int = 100,
        window_seconds: int = 60
    ):
        super().__init__(name, f"Max {max_actions} actions per {window_seconds}s")
        self.max_actions = max_actions
        self.window_seconds = window_seconds
        self._action_times: dict[str, list[datetime]] = {}

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        agent_id = action.agent_id
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Get recent actions for this agent
        if agent_id not in self._action_times:
            self._action_times[agent_id] = []

        # Filter to window
        self._action_times[agent_id] = [
            t for t in self._action_times[agent_id] if t > window_start
        ]

        # Check limit
        if len(self._action_times[agent_id]) >= self.max_actions:
            return PolicyViolation(
                policy_name=self.name,
                severity=Severity.MEDIUM,
                message=f"Rate limit exceeded: {self.max_actions} actions per {self.window_seconds}s",
                details={"current_count": len(self._action_times[agent_id])}
            )

        # Record this action
        self._action_times[agent_id].append(now)
        return None


class SensitiveDataPolicy(Policy):
    """Prevents access to sensitive data without proper authorization"""

    SENSITIVE_PATTERNS = [
        "password", "secret", "token", "key", "credential",
        "ssn", "social_security", "credit_card", "bank_account"
    ]

    def __init__(self):
        super().__init__(
            "sensitive_data",
            "Blocks access to sensitive data without authorization"
        )

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        # Check target for sensitive patterns
        target_lower = action.target.lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in target_lower:
                # Check if agent has explicit authorization
                authorized = context.get("sensitive_data_authorized", False)
                if not authorized:
                    return PolicyViolation(
                        policy_name=self.name,
                        severity=Severity.HIGH,
                        message=f"Access to sensitive data requires authorization",
                        details={"pattern_matched": pattern, "target": action.target}
                    )

        # Check parameters
        params_str = json.dumps(action.parameters).lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in params_str:
                return PolicyViolation(
                    policy_name=self.name,
                    severity=Severity.HIGH,
                    message=f"Sensitive data detected in parameters",
                    details={"pattern_matched": pattern}
                )

        return None


class ExternalAPIPolicy(Policy):
    """Validates external API calls"""

    def __init__(self, allowed_domains: list[str]):
        super().__init__(
            "external_api",
            "Restricts external API calls to approved domains"
        )
        self.allowed_domains = allowed_domains

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        if action.category != ActionCategory.EXTERNAL_API:
            return None

        target = action.target.lower()

        # Check if domain is allowed
        domain_allowed = any(
            domain in target for domain in self.allowed_domains
        )

        if not domain_allowed:
            return PolicyViolation(
                policy_name=self.name,
                severity=Severity.HIGH,
                message=f"External API call to unapproved domain",
                details={
                    "target": action.target,
                    "allowed_domains": self.allowed_domains
                }
            )

        return None


class FinancialLimitPolicy(Policy):
    """Enforces limits on financial transactions"""

    def __init__(
        self,
        max_single_transaction: float = 10000.0,
        max_daily_total: float = 50000.0
    ):
        super().__init__(
            "financial_limit",
            f"Limits transactions to ${max_single_transaction} single, ${max_daily_total} daily"
        )
        self.max_single = max_single_transaction
        self.max_daily = max_daily_total
        self._daily_totals: dict[str, float] = {}

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        if action.category != ActionCategory.FINANCIAL:
            return None

        amount = action.parameters.get("amount", 0)

        # Check single transaction limit
        if amount > self.max_single:
            return PolicyViolation(
                policy_name=self.name,
                severity=Severity.CRITICAL,
                message=f"Transaction exceeds single limit of ${self.max_single}",
                details={"amount": amount, "limit": self.max_single}
            )

        # Check daily total
        agent_id = action.agent_id
        today = datetime.utcnow().date().isoformat()
        key = f"{agent_id}:{today}"

        current_daily = self._daily_totals.get(key, 0)
        if current_daily + amount > self.max_daily:
            return PolicyViolation(
                policy_name=self.name,
                severity=Severity.CRITICAL,
                message=f"Transaction would exceed daily limit of ${self.max_daily}",
                details={
                    "amount": amount,
                    "daily_total": current_daily,
                    "limit": self.max_daily
                }
            )

        # Update daily total
        self._daily_totals[key] = current_daily + amount
        return None


class TwoOutOfThreePolicy(Policy):
    """
    Implements the "two out of three" rule:
    Agents shouldn't simultaneously:
    1. Read sensitive data
    2. Execute code
    3. Work autonomously

    Reference: Google Cloud Next 2026 security guidance
    """

    def __init__(self):
        super().__init__(
            "two_out_of_three",
            "Prevents agents from having all three: sensitive data + code exec + autonomy"
        )

    async def check(self, action: Action, context: dict) -> PolicyViolation | None:
        # Count how many of the three are true
        has_sensitive_data = context.get("has_sensitive_data_access", False)
        can_execute_code = action.category == ActionCategory.EXECUTE_CODE
        is_autonomous = context.get("autonomy_level", 1) >= 4

        active_count = sum([has_sensitive_data, can_execute_code, is_autonomous])

        if active_count >= 3:
            return PolicyViolation(
                policy_name=self.name,
                severity=Severity.CRITICAL,
                message="Violates two-out-of-three rule: cannot have sensitive data + code execution + full autonomy",
                details={
                    "has_sensitive_data": has_sensitive_data,
                    "can_execute_code": can_execute_code,
                    "is_autonomous": is_autonomous
                }
            )

        return None


# =============================================================================
# Guardian Agent
# =============================================================================

class GuardianAgent:
    """
    Safety oversight agent that validates actions before execution.

    The guardian:
    1. Checks all proposed actions against policies
    2. Maintains trust scores for agents
    3. Escalates critical violations to humans
    4. Provides audit trail for all decisions
    """

    def __init__(
        self,
        policies: list[Policy],
        escalation_handler: Callable[[str, Action, list[PolicyViolation]], Awaitable[bool]] | None = None
    ):
        self.policies = policies
        self.escalation_handler = escalation_handler
        self._action_log: list[dict] = []
        self._trust_scores: dict[str, TrustScore] = {}

    async def validate_action(
        self,
        action: Action,
        context: dict | None = None
    ) -> ValidationResult:
        """
        Validate a proposed action against all policies.

        Returns ValidationResult indicating if action is approved.
        """
        context = context or {}
        violations = []
        warnings = []

        # Add trust score to context
        trust = self._get_trust_score(action.agent_id)
        context["autonomy_level"] = trust.autonomy_level

        # Check all policies
        for policy in self.policies:
            if not policy.enabled:
                continue

            violation = await policy.check(action, context)
            if violation:
                if violation.severity >= Severity.HIGH:
                    violations.append(violation)
                else:
                    warnings.append(violation)

        # Log the validation
        self._log_action(action, violations, warnings)

        # Determine approval
        if violations:
            max_severity = max(v.severity for v in violations)

            # Critical violations always blocked
            if max_severity >= Severity.CRITICAL:
                self._update_trust_score(action.agent_id, success=False, violation=True)
                return ValidationResult(
                    approved=False,
                    action_id=action.id,
                    violations=violations,
                    warnings=warnings,
                    reason=f"Blocked: {violations[0].message}"
                )

            # High severity - try escalation
            if max_severity >= Severity.HIGH:
                if self.escalation_handler:
                    human_approved = await self.escalation_handler(
                        action.agent_id, action, violations
                    )
                    if human_approved:
                        return ValidationResult(
                            approved=True,
                            action_id=action.id,
                            violations=[],
                            warnings=violations + warnings,
                            reason="Approved by human after escalation",
                            conditions=["Human oversight confirmed"]
                        )

                self._update_trust_score(action.agent_id, success=False, violation=True)
                return ValidationResult(
                    approved=False,
                    action_id=action.id,
                    violations=violations,
                    warnings=warnings,
                    reason="Blocked pending human approval"
                )

        # Approved (possibly with warnings)
        self._update_trust_score(action.agent_id, success=True, violation=False)

        return ValidationResult(
            approved=True,
            action_id=action.id,
            violations=[],
            warnings=warnings,
            reason="Approved" + (" with warnings" if warnings else ""),
            expires_at=(datetime.utcnow() + timedelta(hours=1)).isoformat()
        )

    def _get_trust_score(self, agent_id: str) -> TrustScore:
        """Get or create trust score for agent"""
        if agent_id not in self._trust_scores:
            self._trust_scores[agent_id] = TrustScore(
                agent_id=agent_id,
                score=0.5,  # Start with medium trust
                successful_actions=0,
                failed_actions=0,
                violations=0,
                last_updated=datetime.utcnow().isoformat()
            )
        return self._trust_scores[agent_id]

    def _update_trust_score(
        self,
        agent_id: str,
        success: bool,
        violation: bool
    ):
        """Update trust score based on action outcome"""
        trust = self._get_trust_score(agent_id)

        if success:
            trust.successful_actions += 1
            # Gradually increase trust
            trust.score = min(1.0, trust.score + 0.01)
        else:
            trust.failed_actions += 1
            # Decrease trust more for violations
            if violation:
                trust.violations += 1
                trust.score = max(0.0, trust.score - 0.1)
            else:
                trust.score = max(0.0, trust.score - 0.02)

        trust.last_updated = datetime.utcnow().isoformat()

    def _log_action(
        self,
        action: Action,
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation]
    ):
        """Log action for audit trail"""
        self._action_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.to_dict(),
            "violations": [v.to_dict() for v in violations],
            "warnings": [w.to_dict() for w in warnings],
            "approved": len(violations) == 0
        })

    def get_audit_log(
        self,
        agent_id: str | None = None,
        since: datetime | None = None
    ) -> list[dict]:
        """Get audit log, optionally filtered"""
        logs = self._action_log

        if agent_id:
            logs = [l for l in logs if l["action"]["agent_id"] == agent_id]

        if since:
            since_str = since.isoformat()
            logs = [l for l in logs if l["timestamp"] >= since_str]

        return logs

    def get_trust_score(self, agent_id: str) -> TrustScore:
        """Get current trust score for agent"""
        return self._get_trust_score(agent_id)


# =============================================================================
# Guarded Agent Wrapper
# =============================================================================

class GuardedAgent:
    """
    Wrapper that adds guardian validation to any agent.

    All actions from the wrapped agent pass through the guardian
    before execution.
    """

    def __init__(
        self,
        agent: Any,
        guardian: GuardianAgent,
        agent_id: str
    ):
        self.agent = agent
        self.guardian = guardian
        self.agent_id = agent_id
        self._pending_approvals: dict[str, Action] = {}

    async def execute(
        self,
        operation: str,
        target: str,
        parameters: dict,
        category: ActionCategory = ActionCategory.READ_DATA,
        context: dict | None = None
    ) -> Any:
        """
        Execute an action through guardian validation.

        Raises ActionBlockedError if guardian rejects the action.
        """
        # Create action
        action = Action(
            id=f"action_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            category=category,
            operation=operation,
            target=target,
            parameters=parameters,
            context=context or {}
        )

        # Validate with guardian
        result = await self.guardian.validate_action(action, context)

        if not result.approved:
            raise ActionBlockedError(
                action=action,
                result=result
            )

        # Execute the actual action
        return await self._execute_internal(operation, target, parameters)

    async def _execute_internal(
        self,
        operation: str,
        target: str,
        parameters: dict
    ) -> Any:
        """Execute the action on the wrapped agent"""
        # This would call the actual agent's method
        # For demo, return success
        return {
            "status": "success",
            "operation": operation,
            "target": target
        }


class ActionBlockedError(Exception):
    """Raised when guardian blocks an action"""

    def __init__(self, action: Action, result: ValidationResult):
        self.action = action
        self.result = result
        super().__init__(f"Action blocked: {result.reason}")


# =============================================================================
# Guardian Factory
# =============================================================================

def create_standard_guardian(
    allowed_api_domains: list[str] | None = None,
    max_transaction: float = 10000.0,
    escalation_handler: Callable | None = None
) -> GuardianAgent:
    """
    Create a guardian with standard enterprise policies.

    This is the recommended starting point for most deployments.
    """
    policies = [
        RateLimitPolicy(max_actions=100, window_seconds=60),
        SensitiveDataPolicy(),
        ExternalAPIPolicy(allowed_api_domains or ["api.company.com"]),
        FinancialLimitPolicy(max_single_transaction=max_transaction),
        TwoOutOfThreePolicy()
    ]

    return GuardianAgent(policies, escalation_handler)


# =============================================================================
# Usage Example
# =============================================================================

async def example_escalation_handler(
    agent_id: str,
    action: Action,
    violations: list[PolicyViolation]
) -> bool:
    """Example escalation handler - in production, this would notify humans"""
    print(f"\n[ESCALATION] Agent {agent_id} action requires approval")
    print(f"  Action: {action.operation} on {action.target}")
    print(f"  Violations: {[v.message for v in violations]}")
    # In production: send to approval queue, return True if approved
    return False  # Default deny


async def main():
    """Demonstrate guardian pattern"""

    print("=" * 60)
    print("Guardian Pattern Demo")
    print("=" * 60)

    # Create guardian
    guardian = create_standard_guardian(
        allowed_api_domains=["api.company.com", "api.partner.com"],
        max_transaction=5000.0,
        escalation_handler=example_escalation_handler
    )

    # Test actions
    test_actions = [
        Action(
            id="action_001",
            agent_id="agent_a",
            category=ActionCategory.READ_DATA,
            operation="read",
            target="users/profile",
            parameters={"user_id": "123"}
        ),
        Action(
            id="action_002",
            agent_id="agent_a",
            category=ActionCategory.READ_DATA,
            operation="read",
            target="users/password_hash",  # Sensitive!
            parameters={"user_id": "123"}
        ),
        Action(
            id="action_003",
            agent_id="agent_a",
            category=ActionCategory.EXTERNAL_API,
            operation="post",
            target="https://malicious-site.com/data",
            parameters={"data": "sensitive"}
        ),
        Action(
            id="action_004",
            agent_id="agent_a",
            category=ActionCategory.FINANCIAL,
            operation="transfer",
            target="bank/transfer",
            parameters={"amount": 10000, "recipient": "vendor"}
        ),
        Action(
            id="action_005",
            agent_id="agent_a",
            category=ActionCategory.EXTERNAL_API,
            operation="get",
            target="https://api.company.com/data",
            parameters={}
        )
    ]

    print("\nValidating actions...")
    for action in test_actions:
        result = await guardian.validate_action(action)
        status = "APPROVED" if result.approved else "BLOCKED"
        print(f"\n[{status}] {action.operation} {action.target}")
        if result.violations:
            for v in result.violations:
                print(f"  Violation: {v.message}")
        if result.warnings:
            for w in result.warnings:
                print(f"  Warning: {w.message}")

    # Show trust score
    print("\n" + "=" * 60)
    print("Trust Score")
    print("=" * 60)
    trust = guardian.get_trust_score("agent_a")
    print(f"Agent: {trust.agent_id}")
    print(f"Score: {trust.score:.2f}")
    print(f"Autonomy Level: {trust.autonomy_level}/5")
    print(f"Successful Actions: {trust.successful_actions}")
    print(f"Failed Actions: {trust.failed_actions}")
    print(f"Violations: {trust.violations}")

    # Show audit log
    print("\n" + "=" * 60)
    print("Audit Log (last 3)")
    print("=" * 60)
    for log in guardian.get_audit_log()[-3:]:
        status = "APPROVED" if log["approved"] else "BLOCKED"
        print(f"[{log['timestamp']}] {status}: {log['action']['operation']} {log['action']['target']}")


if __name__ == "__main__":
    asyncio.run(main())
