"""
Chapter 11: Policy Gateway Implementation
=========================================

Implements a policy enforcement gateway for AI agents that provides:
- Request/response filtering
- Rate limiting and quotas
- Policy-based access control
- Audit logging
- Content moderation

Based on enterprise API gateway patterns adapted for AI agents.
"""

import asyncio
import time
import re
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict
from datetime import datetime, timedelta


class PolicyDecision(Enum):
    """Possible policy evaluation outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"  # Allow with modifications
    AUDIT = "audit"    # Allow but log for review


class PolicyPriority(Enum):
    """Policy evaluation priority."""
    CRITICAL = 0   # Security-critical, evaluated first
    HIGH = 1       # Important business rules
    MEDIUM = 2     # Standard policies
    LOW = 3        # Advisory policies


@dataclass
class PolicyContext:
    """Context for policy evaluation."""
    agent_id: str
    action: str
    resource: str
    payload: dict
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.sha256(
                f"{self.agent_id}:{self.action}:{self.timestamp}".encode()
            ).hexdigest()[:16]


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    decision: PolicyDecision
    policy_id: str
    reason: str
    modifications: Optional[dict] = None
    required_approvers: Optional[list[str]] = None
    audit_data: Optional[dict] = None


@dataclass
class AuditEntry:
    """Audit log entry."""
    request_id: str
    agent_id: str
    action: str
    resource: str
    decision: PolicyDecision
    policy_id: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    payload_hash: str = ""
    response_hash: str = ""
    latency_ms: float = 0.0


class Policy(ABC):
    """Base class for all policies."""

    def __init__(self, policy_id: str, priority: PolicyPriority = PolicyPriority.MEDIUM):
        self.policy_id = policy_id
        self.priority = priority
        self.enabled = True
        self.stats = {"evaluations": 0, "denials": 0, "approvals": 0}

    @abstractmethod
    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate the policy against the given context."""
        pass

    @abstractmethod
    def applies_to(self, context: PolicyContext) -> bool:
        """Check if this policy applies to the given context."""
        pass


class RateLimitPolicy(Policy):
    """
    Rate limiting policy using token bucket algorithm.
    """

    def __init__(self,
                 policy_id: str,
                 requests_per_minute: int = 60,
                 burst_size: int = 10):
        super().__init__(policy_id, PolicyPriority.HIGH)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.buckets: dict[str, dict] = {}  # agent_id -> bucket state
        self._lock = asyncio.Lock()

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.stats["evaluations"] += 1

        async with self._lock:
            bucket = self._get_or_create_bucket(context.agent_id)

            # Refill tokens based on time passed
            now = time.time()
            time_passed = now - bucket["last_refill"]
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now

            # Check if request can proceed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                self.stats["approvals"] += 1
                return PolicyResult(
                    decision=PolicyDecision.ALLOW,
                    policy_id=self.policy_id,
                    reason="Rate limit check passed"
                )
            else:
                self.stats["denials"] += 1
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    policy_id=self.policy_id,
                    reason=f"Rate limit exceeded: {self.requests_per_minute}/min",
                    audit_data={"tokens_remaining": bucket["tokens"]}
                )

    def _get_or_create_bucket(self, agent_id: str) -> dict:
        if agent_id not in self.buckets:
            self.buckets[agent_id] = {
                "tokens": self.burst_size,
                "last_refill": time.time()
            }
        return self.buckets[agent_id]

    def applies_to(self, context: PolicyContext) -> bool:
        return True  # Applies to all requests


class QuotaPolicy(Policy):
    """
    Quota policy for limiting resource usage over time.
    """

    def __init__(self,
                 policy_id: str,
                 daily_limit: int = 1000,
                 monthly_limit: int = 25000):
        super().__init__(policy_id, PolicyPriority.HIGH)
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.usage: dict[str, dict] = {}  # agent_id -> usage tracking

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.stats["evaluations"] += 1

        usage = self._get_or_create_usage(context.agent_id)
        self._reset_if_needed(usage)

        # Check limits
        if usage["daily"] >= self.daily_limit:
            self.stats["denials"] += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                policy_id=self.policy_id,
                reason=f"Daily quota exceeded: {self.daily_limit}",
                audit_data={"daily_usage": usage["daily"]}
            )

        if usage["monthly"] >= self.monthly_limit:
            self.stats["denials"] += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                policy_id=self.policy_id,
                reason=f"Monthly quota exceeded: {self.monthly_limit}",
                audit_data={"monthly_usage": usage["monthly"]}
            )

        # Increment usage
        usage["daily"] += 1
        usage["monthly"] += 1

        self.stats["approvals"] += 1
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            policy_id=self.policy_id,
            reason="Quota check passed",
            audit_data={
                "daily_remaining": self.daily_limit - usage["daily"],
                "monthly_remaining": self.monthly_limit - usage["monthly"]
            }
        )

    def _get_or_create_usage(self, agent_id: str) -> dict:
        if agent_id not in self.usage:
            now = datetime.now()
            self.usage[agent_id] = {
                "daily": 0,
                "monthly": 0,
                "daily_reset": now.date(),
                "monthly_reset": now.replace(day=1).date()
            }
        return self.usage[agent_id]

    def _reset_if_needed(self, usage: dict):
        now = datetime.now()
        if now.date() > usage["daily_reset"]:
            usage["daily"] = 0
            usage["daily_reset"] = now.date()
        if now.date() >= (usage["monthly_reset"] + timedelta(days=32)).replace(day=1):
            usage["monthly"] = 0
            usage["monthly_reset"] = now.replace(day=1).date()

    def applies_to(self, context: PolicyContext) -> bool:
        return True


class ContentPolicy(Policy):
    """
    Content filtering policy for input/output moderation.
    """

    def __init__(self,
                 policy_id: str,
                 blocked_patterns: list[str] = None,
                 pii_detection: bool = True):
        super().__init__(policy_id, PolicyPriority.CRITICAL)
        self.blocked_patterns = [re.compile(p, re.IGNORECASE)
                                 for p in (blocked_patterns or [])]
        self.pii_detection = pii_detection

        # Common PII patterns
        self.pii_patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "api_key": re.compile(r'\b[A-Za-z0-9]{32,}\b')  # Generic API key pattern
        }

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.stats["evaluations"] += 1

        content = json.dumps(context.payload)

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                self.stats["denials"] += 1
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    policy_id=self.policy_id,
                    reason=f"Content contains blocked pattern: {pattern.pattern}"
                )

        # Check for PII
        if self.pii_detection:
            pii_found = []
            for pii_type, pattern in self.pii_patterns.items():
                if pattern.search(content):
                    pii_found.append(pii_type)

            if pii_found:
                self.stats["denials"] += 1
                return PolicyResult(
                    decision=PolicyDecision.MODIFY,
                    policy_id=self.policy_id,
                    reason=f"PII detected: {', '.join(pii_found)}",
                    modifications={"redact_pii": pii_found},
                    audit_data={"pii_types": pii_found}
                )

        self.stats["approvals"] += 1
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            policy_id=self.policy_id,
            reason="Content policy check passed"
        )

    def applies_to(self, context: PolicyContext) -> bool:
        return True


class AccessControlPolicy(Policy):
    """
    Role-based access control policy.
    """

    def __init__(self,
                 policy_id: str,
                 permissions: dict[str, dict[str, list[str]]] = None):
        super().__init__(policy_id, PolicyPriority.CRITICAL)
        # permissions: {role: {resource_pattern: [allowed_actions]}}
        self.permissions = permissions or {}
        self.agent_roles: dict[str, list[str]] = {}

    def assign_role(self, agent_id: str, roles: list[str]):
        """Assign roles to an agent."""
        self.agent_roles[agent_id] = roles

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.stats["evaluations"] += 1

        roles = self.agent_roles.get(context.agent_id, ["default"])

        for role in roles:
            if role in self.permissions:
                role_perms = self.permissions[role]
                for resource_pattern, allowed_actions in role_perms.items():
                    if self._match_resource(context.resource, resource_pattern):
                        if context.action in allowed_actions or "*" in allowed_actions:
                            self.stats["approvals"] += 1
                            return PolicyResult(
                                decision=PolicyDecision.ALLOW,
                                policy_id=self.policy_id,
                                reason=f"Access granted via role: {role}"
                            )

        self.stats["denials"] += 1
        return PolicyResult(
            decision=PolicyDecision.DENY,
            policy_id=self.policy_id,
            reason=f"Access denied: no permission for {context.action} on {context.resource}",
            audit_data={"agent_roles": roles}
        )

    def _match_resource(self, resource: str, pattern: str) -> bool:
        """Match resource against pattern (supports wildcards)."""
        if pattern == "*":
            return True
        if pattern.endswith("/*"):
            return resource.startswith(pattern[:-1])
        return resource == pattern

    def applies_to(self, context: PolicyContext) -> bool:
        return True


class ApprovalPolicy(Policy):
    """
    Policy requiring human approval for sensitive actions.
    """

    def __init__(self,
                 policy_id: str,
                 sensitive_actions: list[str] = None,
                 sensitive_resources: list[str] = None,
                 approvers: list[str] = None):
        super().__init__(policy_id, PolicyPriority.HIGH)
        self.sensitive_actions = sensitive_actions or ["delete", "modify_permissions", "transfer"]
        self.sensitive_resources = sensitive_resources or ["production/*", "finance/*"]
        self.approvers = approvers or ["admin"]

    async def evaluate(self, context: PolicyContext) -> PolicyResult:
        self.stats["evaluations"] += 1

        # Check if action is sensitive
        action_sensitive = context.action in self.sensitive_actions

        # Check if resource is sensitive
        resource_sensitive = any(
            self._match_resource(context.resource, pattern)
            for pattern in self.sensitive_resources
        )

        if action_sensitive or resource_sensitive:
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                policy_id=self.policy_id,
                reason=f"Action requires approval: {'sensitive action' if action_sensitive else 'sensitive resource'}",
                required_approvers=self.approvers,
                audit_data={
                    "action_sensitive": action_sensitive,
                    "resource_sensitive": resource_sensitive
                }
            )

        self.stats["approvals"] += 1
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            policy_id=self.policy_id,
            reason="No approval required"
        )

    def _match_resource(self, resource: str, pattern: str) -> bool:
        if pattern.endswith("/*"):
            return resource.startswith(pattern[:-1])
        return resource == pattern

    def applies_to(self, context: PolicyContext) -> bool:
        return True


class AuditLogger:
    """
    Audit logging system for gateway operations.
    """

    def __init__(self, max_entries: int = 10000):
        self.entries: list[AuditEntry] = []
        self.max_entries = max_entries
        self._lock = asyncio.Lock()

    async def log(self, entry: AuditEntry):
        """Log an audit entry."""
        async with self._lock:
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

    async def query(self,
                    agent_id: Optional[str] = None,
                    action: Optional[str] = None,
                    decision: Optional[PolicyDecision] = None,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None,
                    limit: int = 100) -> list[AuditEntry]:
        """Query audit logs."""
        results = []
        for entry in reversed(self.entries):
            if agent_id and entry.agent_id != agent_id:
                continue
            if action and entry.action != action:
                continue
            if decision and entry.decision != decision:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    async def get_stats(self, window_seconds: int = 3600) -> dict:
        """Get statistics for the specified time window."""
        cutoff = time.time() - window_seconds
        recent = [e for e in self.entries if e.timestamp >= cutoff]

        return {
            "total_requests": len(recent),
            "decisions": {
                d.value: sum(1 for e in recent if e.decision == d)
                for d in PolicyDecision
            },
            "by_agent": defaultdict(int, {
                e.agent_id: sum(1 for r in recent if r.agent_id == e.agent_id)
                for e in recent
            }),
            "avg_latency_ms": sum(e.latency_ms for e in recent) / len(recent) if recent else 0
        }


class PolicyGateway:
    """
    Main gateway that enforces policies on agent requests.
    """

    def __init__(self):
        self.policies: list[Policy] = []
        self.audit_logger = AuditLogger()
        self.pending_approvals: dict[str, dict] = {}  # request_id -> approval state

    def add_policy(self, policy: Policy):
        """Add a policy to the gateway."""
        self.policies.append(policy)
        # Sort by priority
        self.policies.sort(key=lambda p: p.priority.value)

    def remove_policy(self, policy_id: str):
        """Remove a policy from the gateway."""
        self.policies = [p for p in self.policies if p.policy_id != policy_id]

    async def process_request(self,
                               context: PolicyContext,
                               handler: Callable) -> dict:
        """
        Process a request through the gateway.

        Args:
            context: The policy evaluation context
            handler: The actual handler to call if policies allow

        Returns:
            Result dict with response or error
        """
        start_time = time.time()

        # Evaluate all applicable policies
        results = []
        for policy in self.policies:
            if policy.enabled and policy.applies_to(context):
                result = await policy.evaluate(context)
                results.append(result)

                # Short-circuit on DENY
                if result.decision == PolicyDecision.DENY:
                    await self._log_request(context, result, start_time)
                    return {
                        "allowed": False,
                        "error": result.reason,
                        "policy": result.policy_id,
                        "request_id": context.request_id
                    }

                # Handle approval requirement
                if result.decision == PolicyDecision.REQUIRE_APPROVAL:
                    approval_id = await self._create_approval_request(context, result)
                    await self._log_request(context, result, start_time)
                    return {
                        "allowed": False,
                        "pending_approval": True,
                        "approval_id": approval_id,
                        "required_approvers": result.required_approvers,
                        "request_id": context.request_id
                    }

        # Apply any modifications
        modified_payload = context.payload.copy()
        for result in results:
            if result.decision == PolicyDecision.MODIFY and result.modifications:
                modified_payload = await self._apply_modifications(
                    modified_payload, result.modifications
                )

        # Execute the handler
        try:
            if asyncio.iscoroutinefunction(handler):
                response = await handler(modified_payload)
            else:
                response = handler(modified_payload)

            # Log success
            audit_entry = AuditEntry(
                request_id=context.request_id,
                agent_id=context.agent_id,
                action=context.action,
                resource=context.resource,
                decision=PolicyDecision.ALLOW,
                policy_id="gateway",
                reason="All policies passed",
                latency_ms=(time.time() - start_time) * 1000,
                payload_hash=hashlib.sha256(json.dumps(context.payload).encode()).hexdigest()[:16],
                response_hash=hashlib.sha256(json.dumps(response).encode()).hexdigest()[:16] if response else ""
            )
            await self.audit_logger.log(audit_entry)

            return {
                "allowed": True,
                "response": response,
                "request_id": context.request_id,
                "latency_ms": audit_entry.latency_ms
            }

        except Exception as e:
            return {
                "allowed": True,
                "error": str(e),
                "request_id": context.request_id
            }

    async def _apply_modifications(self, payload: dict, modifications: dict) -> dict:
        """Apply policy-required modifications to payload."""
        result = payload.copy()

        if "redact_pii" in modifications:
            result = await self._redact_pii(result, modifications["redact_pii"])

        return result

    async def _redact_pii(self, payload: dict, pii_types: list[str]) -> dict:
        """Redact PII from payload."""
        content = json.dumps(payload)

        redaction_patterns = {
            "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),
            "credit_card": (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '****-****-****-****'),
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),
            "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]'),
            "api_key": (r'\b[A-Za-z0-9]{32,}\b', '[REDACTED_KEY]')
        }

        for pii_type in pii_types:
            if pii_type in redaction_patterns:
                pattern, replacement = redaction_patterns[pii_type]
                content = re.sub(pattern, replacement, content)

        return json.loads(content)

    async def _create_approval_request(self, context: PolicyContext,
                                        result: PolicyResult) -> str:
        """Create a pending approval request."""
        approval_id = f"approval-{context.request_id}"
        self.pending_approvals[approval_id] = {
            "context": context,
            "result": result,
            "created_at": time.time(),
            "status": "pending",
            "approvals": [],
            "rejections": []
        }
        return approval_id

    async def approve(self, approval_id: str, approver: str,
                      handler: Callable) -> dict:
        """Approve a pending request."""
        if approval_id not in self.pending_approvals:
            return {"error": "Approval request not found"}

        approval = self.pending_approvals[approval_id]
        result = approval["result"]

        if approver not in result.required_approvers:
            return {"error": f"Approver {approver} not authorized"}

        approval["approvals"].append({
            "approver": approver,
            "timestamp": time.time()
        })

        # Check if enough approvals
        if len(approval["approvals"]) >= 1:  # Can adjust threshold
            approval["status"] = "approved"
            # Execute the original request
            context = approval["context"]
            return await self.process_request(context, handler)

        return {"status": "pending", "approvals": len(approval["approvals"])}

    async def reject(self, approval_id: str, rejector: str, reason: str) -> dict:
        """Reject a pending approval request."""
        if approval_id not in self.pending_approvals:
            return {"error": "Approval request not found"}

        approval = self.pending_approvals[approval_id]
        approval["status"] = "rejected"
        approval["rejections"].append({
            "rejector": rejector,
            "reason": reason,
            "timestamp": time.time()
        })

        return {"status": "rejected", "reason": reason}

    async def _log_request(self, context: PolicyContext,
                           result: PolicyResult, start_time: float):
        """Log a denied or pending request."""
        entry = AuditEntry(
            request_id=context.request_id,
            agent_id=context.agent_id,
            action=context.action,
            resource=context.resource,
            decision=result.decision,
            policy_id=result.policy_id,
            reason=result.reason,
            latency_ms=(time.time() - start_time) * 1000,
            payload_hash=hashlib.sha256(json.dumps(context.payload).encode()).hexdigest()[:16]
        )
        await self.audit_logger.log(entry)

    def get_policy_stats(self) -> dict:
        """Get statistics for all policies."""
        return {
            policy.policy_id: policy.stats
            for policy in self.policies
        }


# =============================================================================
# Example Usage
# =============================================================================

async def example_handler(payload: dict) -> dict:
    """Example request handler."""
    await asyncio.sleep(0.05)  # Simulate processing
    return {
        "status": "success",
        "processed": payload.get("data", "unknown")
    }


async def main():
    """Demonstration of policy gateway."""
    print("=" * 60)
    print("Policy Gateway Demonstration")
    print("=" * 60)

    # Create gateway
    gateway = PolicyGateway()

    # Add policies
    gateway.add_policy(RateLimitPolicy(
        policy_id="rate-limit",
        requests_per_minute=10,
        burst_size=5
    ))

    gateway.add_policy(QuotaPolicy(
        policy_id="quota",
        daily_limit=100,
        monthly_limit=2000
    ))

    gateway.add_policy(ContentPolicy(
        policy_id="content-filter",
        blocked_patterns=[r"password\s*=\s*\S+", r"secret"],
        pii_detection=True
    ))

    access_policy = AccessControlPolicy(
        policy_id="access-control",
        permissions={
            "admin": {"*": ["*"]},
            "analyst": {"data/*": ["read", "analyze"], "reports/*": ["read", "write"]},
            "default": {"public/*": ["read"]}
        }
    )
    access_policy.assign_role("agent-admin", ["admin"])
    access_policy.assign_role("agent-analyst", ["analyst"])
    access_policy.assign_role("agent-guest", ["default"])
    gateway.add_policy(access_policy)

    gateway.add_policy(ApprovalPolicy(
        policy_id="approval",
        sensitive_actions=["delete"],
        sensitive_resources=["production/*"],
        approvers=["admin"]
    ))

    print(f"\nGateway configured with {len(gateway.policies)} policies")
    print()

    # Test cases
    test_cases = [
        {
            "name": "Normal request",
            "context": PolicyContext(
                agent_id="agent-analyst",
                action="read",
                resource="data/reports",
                payload={"query": "monthly stats"}
            )
        },
        {
            "name": "Request with PII",
            "context": PolicyContext(
                agent_id="agent-analyst",
                action="analyze",
                resource="data/customers",
                payload={"email": "user@example.com", "ssn": "123-45-6789"}
            )
        },
        {
            "name": "Unauthorized access",
            "context": PolicyContext(
                agent_id="agent-guest",
                action="write",
                resource="data/reports",
                payload={"data": "test"}
            )
        },
        {
            "name": "Sensitive action requiring approval",
            "context": PolicyContext(
                agent_id="agent-admin",
                action="delete",
                resource="production/database",
                payload={"table": "users"}
            )
        },
        {
            "name": "Blocked content",
            "context": PolicyContext(
                agent_id="agent-admin",
                action="write",
                resource="config/settings",
                payload={"data": "password = mysecretpass123"}
            )
        }
    ]

    for test in test_cases:
        print(f"Test: {test['name']}")
        result = await gateway.process_request(test["context"], example_handler)
        print(f"  Allowed: {result.get('allowed', False)}")
        if result.get('error'):
            print(f"  Error: {result['error']}")
        if result.get('pending_approval'):
            print(f"  Pending approval: {result['approval_id']}")
        if result.get('response'):
            print(f"  Response: {result['response']}")
        print()

    # Rate limit test
    print("Rate limit test (rapid requests):")
    for i in range(8):
        context = PolicyContext(
            agent_id="agent-analyst",
            action="read",
            resource="data/stats",
            payload={"query": f"test {i}"}
        )
        result = await gateway.process_request(context, example_handler)
        status = "allowed" if result.get("allowed") else "denied"
        print(f"  Request {i+1}: {status}")

    print()

    # Show statistics
    print("=" * 60)
    print("Policy Statistics")
    print("=" * 60)
    stats = gateway.get_policy_stats()
    for policy_id, policy_stats in stats.items():
        print(f"\n{policy_id}:")
        print(f"  Evaluations: {policy_stats['evaluations']}")
        print(f"  Approvals: {policy_stats['approvals']}")
        print(f"  Denials: {policy_stats['denials']}")

    # Audit log
    print("\n" + "=" * 60)
    print("Recent Audit Entries")
    print("=" * 60)
    audit_stats = await gateway.audit_logger.get_stats(window_seconds=3600)
    print(f"\nTotal requests: {audit_stats['total_requests']}")
    print(f"Decisions: {audit_stats['decisions']}")


if __name__ == "__main__":
    asyncio.run(main())
