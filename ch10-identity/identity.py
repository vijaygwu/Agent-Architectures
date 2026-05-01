"""
Chapter 10: Agent Identity Fundamentals
Identity Service Implementation
================================

A complete identity service for multi-agent systems.

This module provides:
1. AgentIdentity and role-based identity management
2. JWT token issuance and verification
3. Token lifecycle: issuance, validation, refresh, revocation
4. Scope-based authorization
5. Audit logging
6. Hierarchical identity and delegation patterns
7. Security best practices
"""

import asyncio
import hashlib
import jwt
import logging
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from secrets import token_urlsafe
from typing import Any
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("identity")


# =============================================================================
# Simple Token Authentication (Section: Simple Tokens)
# =============================================================================

@dataclass
class AgentToken:
    agent_id: str
    token: str
    created_at: datetime
    expires_at: datetime | None = None

class SimpleTokenAuth:
    def __init__(self):
        self.tokens: dict[str, AgentToken] = {}

    def create_token(self, agent_id: str, ttl_hours: int = 24) -> str:
        """Create a new token for an agent."""
        token = token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        self.tokens[token_hash] = AgentToken(
            agent_id=agent_id,
            token=token_hash,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=ttl_hours)
        )

        return token  # Return unhashed token to agent

    def authenticate(self, token: str) -> str | None:
        """Authenticate a token and return agent_id if valid."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        agent_token = self.tokens.get(token_hash)
        if not agent_token:
            return None

        if agent_token.expires_at and agent_token.expires_at < datetime.utcnow():
            return None

        return agent_token.agent_id

    def revoke(self, agent_id: str):
        """Revoke all tokens for an agent."""
        self.tokens = {
            k: v for k, v in self.tokens.items()
            if v.agent_id != agent_id
        }


# =============================================================================
# Scoped Token Authentication (Section: API Keys with Scopes)
# =============================================================================

@dataclass
class ScopedToken:
    agent_id: str
    token_hash: str
    scopes: set[str]  # e.g., {"read:customers", "write:orders"}
    created_at: datetime
    expires_at: datetime | None

class ScopedAuth:
    def __init__(self):
        self.tokens: dict[str, ScopedToken] = {}

    def create_token(self, agent_id: str, scopes: set[str]) -> str:
        token = token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        self.tokens[token_hash] = ScopedToken(
            agent_id=agent_id,
            token_hash=token_hash,
            scopes=scopes,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )

        return token

    def authorize(self, token: str, required_scope: str) -> bool:
        """Check if token has required scope."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        agent_token = self.tokens.get(token_hash)

        if not agent_token:
            return False

        if agent_token.expires_at and agent_token.expires_at < datetime.utcnow():
            return False

        return required_scope in agent_token.scopes


# =============================================================================
# JWT Authentication (Section: JWT - JSON Web Tokens)
# =============================================================================

class JWTAuth:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(self, agent_id: str, scopes: list[str],
                      ttl_hours: int = 24) -> str:
        """Create a JWT for an agent."""
        now = datetime.utcnow()
        payload = {
            "sub": agent_id,  # Subject (agent ID)
            "scopes": scopes,
            "iat": now,       # Issued at
            "exp": now + timedelta(hours=ttl_hours),  # Expiration
            "jti": token_urlsafe(16)  # Unique token ID
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict | None:
        """Verify and decode a JWT."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_agent_id(self, token: str) -> str | None:
        """Extract agent ID from token."""
        payload = self.verify_token(token)
        return payload.get("sub") if payload else None

    def has_scope(self, token: str, required_scope: str) -> bool:
        """Check if token has required scope."""
        payload = self.verify_token(token)
        if not payload:
            return False
        return required_scope in payload.get("scopes", [])


# =============================================================================
# Asymmetric JWT Signing (Section: Asymmetric JWT Signing)
# =============================================================================

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

class AsymmetricJWTAuth:
    def __init__(self, private_key_pem: bytes = None, public_key_pem: bytes = None):
        if private_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem, password=None
            )
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

        self.public_key = self.private_key.public_key() if self.private_key else (
            serialization.load_pem_public_key(public_key_pem)
        )

    def create_token(self, agent_id: str, scopes: list[str],
                      ttl_hours: int = 24) -> str:
        """Create a JWT signed with private key."""
        now = datetime.utcnow()
        payload = {
            "sub": agent_id,
            "scopes": scopes,
            "iat": now,
            "exp": now + timedelta(hours=ttl_hours),
            "jti": token_urlsafe(16)
        }

        return jwt.encode(payload, self.private_key, algorithm="RS256")

    def verify_token(self, token: str) -> dict | None:
        """Verify token with public key only."""
        try:
            return jwt.decode(token, self.public_key, algorithms=["RS256"])
        except jwt.InvalidTokenError:
            return None

    def get_public_key_pem(self) -> bytes:
        """Export public key for distribution."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )


# =============================================================================
# Identity Types (Section: Service Accounts vs Agent Accounts)
# =============================================================================

class IdentityType(Enum):
    SERVICE = "service"    # Long-lived, for system components
    AGENT = "agent"        # Potentially ephemeral, for AI agents
    USER = "user"          # Human users
    EXTERNAL = "external"  # Third-party integrations

@dataclass
class Identity:
    id: str
    type: IdentityType
    name: str
    parent_id: str | None = None  # For hierarchical relationships
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Agent Role and Identity (Section: A Complete Identity Service)
# =============================================================================

class AgentRole(Enum):
    """High-level roles for access control"""
    WORKER = "worker"
    ORCHESTRATOR = "orchestrator"
    GUARDIAN = "guardian"
    ADMIN = "admin"

@dataclass
class AgentIdentity:
    agent_id: str
    name: str
    role: AgentRole
    scopes: set[str]  # Fine-grained permissions (e.g., "data:read")
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_claims(self) -> dict:
        return {
            "sub": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "scopes": list(self.scopes)
        }

@dataclass
class TokenInfo:
    token_id: str
    agent_id: str
    issued_at: datetime
    expires_at: datetime
    revoked: bool = False


class IdentityService:
    """
    Manages agent identities and authentication tokens.

    Features:
    - Agent registration and management
    - JWT token issuance and verification
    - Token revocation
    - Scope-based authorization
    """

    def __init__(self, secret_key: str, issuer: str = "agent-system"):
        self.secret_key = secret_key
        self.issuer = issuer
        self.agents: dict[str, AgentIdentity] = {}
        self.tokens: dict[str, TokenInfo] = {}
        self.revoked_tokens: set[str] = set()
        logger.info(f"IdentityService initialized with issuer: {issuer}")

    # ----- Agent Management -----

    def register_agent(
        self,
        name: str,
        role: AgentRole,
        scopes: set[str] | None = None,
        metadata: dict | None = None
    ) -> AgentIdentity:
        """Register a new agent identity."""
        agent_id = f"agent_{token_urlsafe(8)}"

        # Default scopes based on role
        if scopes is None:
            scopes = self._default_scopes(role)

        identity = AgentIdentity(
            agent_id=agent_id,
            name=name,
            role=role,
            scopes=scopes,
            metadata=metadata or {}
        )

        self.agents[agent_id] = identity
        logger.info(f"Registered agent: {name} ({agent_id}) with role {role.value}")
        return identity

    def get_agent(self, agent_id: str) -> AgentIdentity | None:
        """Get an agent's identity."""
        return self.agents.get(agent_id)

    def update_scopes(self, agent_id: str, scopes: set[str]):
        """Update an agent's scopes."""
        if agent_id in self.agents:
            self.agents[agent_id].scopes = scopes
            logger.info(f"Updated scopes for agent {agent_id}: {scopes}")

    def delete_agent(self, agent_id: str):
        """Delete an agent and revoke all its tokens."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # Revoke all tokens for this agent
            for token_id, info in self.tokens.items():
                if info.agent_id == agent_id:
                    self.revoked_tokens.add(token_id)
            logger.info(f"Deleted agent {agent_id} and revoked all tokens")

    def _default_scopes(self, role: AgentRole) -> set[str]:
        """Default scopes for each role."""
        base = {"read:self"}

        if role == AgentRole.WORKER:
            return base | {"execute:tasks"}
        elif role == AgentRole.ORCHESTRATOR:
            return base | {"execute:tasks", "delegate:tasks", "read:workers"}
        elif role == AgentRole.GUARDIAN:
            return base | {"validate:actions", "read:all", "block:actions"}
        elif role == AgentRole.ADMIN:
            return base | {"admin:*"}

        return base

    # ----- Token Management -----

    def issue_token(
        self,
        agent_id: str,
        ttl_hours: int = 24,
        additional_claims: dict | None = None
    ) -> str | None:
        """Issue a JWT token for an agent."""
        identity = self.agents.get(agent_id)
        if not identity:
            logger.warning(f"Token issuance failed: agent {agent_id} not found")
            return None

        now = datetime.utcnow()
        token_id = token_urlsafe(16)

        payload = {
            **identity.to_claims(),
            "iss": self.issuer,
            "iat": now,
            "exp": now + timedelta(hours=ttl_hours),
            "jti": token_id
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        # Track token for revocation
        self.tokens[token_id] = TokenInfo(
            token_id=token_id,
            agent_id=agent_id,
            issued_at=now,
            expires_at=now + timedelta(hours=ttl_hours)
        )

        logger.info(f"Issued token for agent {agent_id}, expires in {ttl_hours} hours")
        return token

    def verify_token(self, token: str) -> dict | None:
        """Verify a token and return its claims."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                issuer=self.issuer
            )

            # Check if revoked
            token_id = payload.get("jti")
            if token_id in self.revoked_tokens:
                logger.warning(f"Token {token_id} has been revoked")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {e}")
            return None

    def revoke_token(self, token: str):
        """Revoke a specific token."""
        payload = self.verify_token(token)
        if payload and "jti" in payload:
            self.revoked_tokens.add(payload["jti"])
            logger.info(f"Revoked token {payload['jti']}")

    def revoke_all_tokens(self, agent_id: str):
        """Revoke all tokens for an agent."""
        count = 0
        for token_id, info in self.tokens.items():
            if info.agent_id == agent_id:
                self.revoked_tokens.add(token_id)
                count += 1
        logger.info(f"Revoked {count} tokens for agent {agent_id}")

    # ----- Authorization -----

    def authorize(self, token: str, required_scope: str) -> bool:
        """Check if a token has the required scope."""
        payload = self.verify_token(token)
        if not payload:
            return False

        scopes = set(payload.get("scopes", []))

        # Check for admin wildcard
        if "admin:*" in scopes:
            return True

        # Check for exact match or wildcard
        if required_scope in scopes:
            return True

        # Check for category wildcard (e.g., "read:*" matches "read:customers")
        category = required_scope.split(":")[0]
        if f"{category}:*" in scopes:
            return True

        return False

    def get_agent_from_token(self, token: str) -> AgentIdentity | None:
        """Get the agent identity from a token."""
        payload = self.verify_token(token)
        if not payload:
            return None
        return self.agents.get(payload.get("sub"))


# =============================================================================
# Authentication Middleware (Section: A Complete Identity Service)
# =============================================================================

class AuthenticationError(Exception):
    pass

class AuthorizationError(Exception):
    pass


class AuthMiddleware:
    """Middleware for authenticating agent requests."""

    def __init__(self, identity_service: IdentityService):
        self.identity_service = identity_service

    async def authenticate(self, request: dict) -> AgentIdentity | None:
        """Authenticate a request and return the agent identity."""
        # Get token from header
        auth_header = request.get("headers", {}).get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix
        return self.identity_service.get_agent_from_token(token)

    def require_scope(self, scope: str):
        """Decorator to require a specific scope."""
        def decorator(func):
            async def wrapper(request, *args, **kwargs):
                auth_header = request.get("headers", {}).get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    raise AuthenticationError("Missing authentication")

                token = auth_header[7:]
                if not self.identity_service.authorize(token, scope):
                    raise AuthorizationError(f"Missing required scope: {scope}")

                return await func(request, *args, **kwargs)
            return wrapper
        return decorator


# =============================================================================
# Agent-to-Agent Authentication (Section: Authentication Flows)
# =============================================================================

class AuthenticatedAgent:
    def __init__(self, identity: AgentIdentity, token: str):
        self.identity = identity
        self.token = token

    async def call_agent(self, target_url: str, request: dict) -> dict:
        """Call another agent with authentication."""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Agent-ID": self.identity.agent_id
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    target_url, json=request, headers=headers) as resp:
                return await resp.json()


# =============================================================================
# Token Refresh (Section: Token Refresh)
# =============================================================================

class TokenManager:
    def __init__(self, identity_service: IdentityService, agent_id: str):
        self.service = identity_service
        self.agent_id = agent_id
        self.current_token: str | None = None
        self.refresh_threshold = timedelta(hours=1)

    async def get_token(self) -> str:
        """Get a valid token, refreshing if needed."""
        if self.current_token:
            payload = self.service.verify_token(self.current_token)
            if payload:
                exp = datetime.fromtimestamp(payload["exp"])
                if exp - datetime.utcnow() > self.refresh_threshold:
                    return self.current_token

        # Need new token
        self.current_token = self.service.issue_token(self.agent_id)
        logger.info(f"Refreshed token for agent {self.agent_id}")
        return self.current_token

    async def refresh_loop(self):
        """Background task to keep token fresh."""
        while True:
            await asyncio.sleep(3600)  # Check every hour
            await self.get_token()


# =============================================================================
# Audit Logging (Section: Audit Logging)
# =============================================================================

# Type alias for storage interface
class Storage:
    def append(self, entry: dict):
        pass

@dataclass
class AuditEntry:
    timestamp: datetime
    agent_id: str
    action: str
    resource: str
    outcome: str  # "success", "denied", "error"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details
        }

class AuditLog:
    def __init__(self, storage: Storage | None = None):
        self.entries: list[AuditEntry] = []
        self.storage = storage

    def log(self, agent_id: str, action: str, resource: str,
            outcome: str, details: dict = None):
        """Log an agent action."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {}
        )

        self.entries.append(entry)
        logger.info(f"Audit: {agent_id} {action} {resource} -> {outcome}")

        if self.storage:
            self.storage.append(entry.to_dict())

    def query(self, agent_id: str = None, action: str = None,
              start_time: datetime = None,
              end_time: datetime = None) -> list[AuditEntry]:
        """Query audit log."""
        results = self.entries

        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if action:
            results = [e for e in results if e.action == action]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results


# =============================================================================
# Secure Agent Integration (Section: Integrating Identity with Agents)
# =============================================================================

class SecureAgent:
    def __init__(
        self,
        identity_service: IdentityService,
        audit_log: AuditLog,
        name: str,
        role: AgentRole
    ):
        self.identity_service = identity_service
        self.audit_log = audit_log

        # Register and get identity
        self.identity = identity_service.register_agent(name=name, role=role)
        self.token_manager = TokenManager(identity_service, self.identity.agent_id)

    async def execute_action(self, action: str, resource: str, **kwargs) -> Any:
        """Execute an action with authentication and audit."""
        token = await self.token_manager.get_token()

        # Check authorization
        required_scope = f"{action}:{resource.split('/')[0]}"
        if not self.identity_service.authorize(token, required_scope):
            self.audit_log.log(
                agent_id=self.identity.agent_id,
                action=action,
                resource=resource,
                outcome="denied",
                details={"reason": "insufficient_scope"}
            )
            raise AuthorizationError(f"Not authorized for {required_scope}")

        # Execute
        try:
            result = await self._do_action(action, resource, **kwargs)
            self.audit_log.log(
                agent_id=self.identity.agent_id,
                action=action,
                resource=resource,
                outcome="success"
            )
            return result
        except Exception as e:
            self.audit_log.log(
                agent_id=self.identity.agent_id,
                action=action,
                resource=resource,
                outcome="error",
                details={"error": str(e)}
            )
            raise

    async def _do_action(self, action: str, resource: str, **kwargs) -> Any:
        """Override in subclass to implement actual action."""
        raise NotImplementedError


# =============================================================================
# Hierarchical Identity (Section: Identity Inheritance)
# =============================================================================

class HierarchicalIdentityService(IdentityService):
    def spawn_child(
        self,
        parent_token: str,
        child_name: str,
        scope_subset: set[str] | None = None
    ) -> tuple[AgentIdentity, str]:
        """Create a child identity with inherited permissions."""

        parent_payload = self.verify_token(parent_token)
        if not parent_payload:
            raise AuthenticationError("Invalid parent token")

        parent_id = parent_payload["sub"]
        parent_scopes = set(parent_payload.get("scopes", []))

        # Child scopes are subset of parent scopes
        if scope_subset:
            child_scopes = parent_scopes & scope_subset
        else:
            child_scopes = parent_scopes

        # Create child identity
        child_id = f"{parent_id}:child_{token_urlsafe(8)}"

        child_identity = AgentIdentity(
            agent_id=child_id,
            name=child_name,
            role=AgentRole.WORKER,
            scopes=child_scopes,
            metadata={"parent_id": parent_id}
        )

        self.agents[child_id] = child_identity
        logger.info(f"Spawned child agent {child_id} from parent {parent_id}")

        # Issue token with parent reference
        child_token = self.issue_token(
            child_id,
            additional_claims={"parent": parent_id}
        )

        return child_identity, child_token


# =============================================================================
# Impersonation Service (Section: Impersonation)
# =============================================================================

class ImpersonationService:
    def __init__(self, identity_service: IdentityService):
        self.service = identity_service
        self.allowed_impersonation: dict[str, set[str]] = {}

    def allow_impersonation(self, impersonator: str, target: str):
        """Grant impersonation permission."""
        if impersonator not in self.allowed_impersonation:
            self.allowed_impersonation[impersonator] = set()
        self.allowed_impersonation[impersonator].add(target)
        logger.info(f"Allowed {impersonator} to impersonate {target}")

    def impersonate(
        self,
        impersonator_token: str,
        target_id: str
    ) -> str | None:
        """Create a token that acts as target identity."""

        impersonator_payload = self.service.verify_token(impersonator_token)
        if not impersonator_payload:
            return None

        impersonator_id = impersonator_payload["sub"]

        # Check permission
        if target_id not in self.allowed_impersonation.get(impersonator_id, set()):
            logger.warning(f"Impersonation denied: {impersonator_id} -> {target_id}")
            return None

        # Issue token for target with impersonation metadata
        target_identity = self.service.get_agent(target_id)
        if not target_identity:
            return None

        logger.info(f"Agent {impersonator_id} impersonating {target_id}")
        return self.service.issue_token(
            target_id,
            additional_claims={
                "act": {"sub": impersonator_id},  # Acting party claim
                "impersonated": True
            }
        )


# =============================================================================
# Constrained Delegation (Section: Constrained Delegation)
# =============================================================================

@dataclass
class DelegationGrant:
    delegator: str
    delegate: str
    scopes: set[str]
    constraints: dict
    expires_at: datetime

    def is_valid(self) -> bool:
        return datetime.utcnow() < self.expires_at

    def can_use_scope(self, scope: str) -> bool:
        return scope in self.scopes and self.is_valid()

class DelegationService:
    def __init__(self):
        self.grants: dict[str, DelegationGrant] = {}

    def create_grant(
        self,
        delegator_token: str,
        delegate_id: str,
        scopes: set[str],
        identity_service: IdentityService,
        constraints: dict = None,
        ttl_hours: int = 24
    ) -> DelegationGrant:
        """Create a delegation grant."""

        payload = identity_service.verify_token(delegator_token)
        if not payload:
            raise AuthenticationError("Invalid delegator token")

        delegator_scopes = set(payload.get("scopes", []))

        # Can only delegate scopes you have
        granted_scopes = scopes & delegator_scopes

        grant = DelegationGrant(
            delegator=payload["sub"],
            delegate=delegate_id,
            scopes=granted_scopes,
            constraints=constraints or {},
            expires_at=datetime.utcnow() + timedelta(hours=ttl_hours)
        )

        grant_id = f"grant_{token_urlsafe(8)}"
        self.grants[grant_id] = grant
        logger.info(f"Created delegation grant {grant_id}: {payload['sub']} -> {delegate_id}")

        return grant


# =============================================================================
# Security Best Practices
# =============================================================================

# Principle of Least Privilege (Section: Principle of Least Privilege)

class LeastPrivilegeAssigner:
    """Automatically determine minimum required scopes."""

    def __init__(self):
        self.action_to_scopes: dict[str, set[str]] = {
            "read_document": {"read:documents"},
            "search": {"read:search"},
            "write_document": {"write:documents"},
            "send_email": {"execute:email"},
            "call_api": {"execute:external_apis"},
        }

    def scopes_for_task(self, task_description: str) -> set[str]:
        """Determine scopes needed for a task."""
        needed = set()

        for action, scopes in self.action_to_scopes.items():
            if action in task_description.lower():
                needed |= scopes

        return needed

    def issue_task_token(
        self,
        identity_service: IdentityService,
        agent_id: str,
        task_description: str
    ) -> str:
        """Issue a token scoped to a specific task."""

        scopes = self.scopes_for_task(task_description)
        agent = identity_service.get_agent(agent_id)

        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # Intersect with agent's allowed scopes
        available_scopes = agent.scopes & scopes

        logger.info(f"Issuing task-scoped token for {agent_id}: {available_scopes}")
        return identity_service.issue_token(
            agent_id,
            ttl_hours=1,  # Short-lived
            additional_claims={
                "scopes": list(available_scopes),
                "task": task_description[:100]
            }
        )


# Token Security (Section: Token Security)

class SecureTokenStorage:
    """Store tokens securely."""

    def __init__(self, encryption_key: bytes):
        from cryptography.fernet import Fernet
        self.fernet = Fernet(encryption_key)
        self.tokens: dict[str, bytes] = {}

    def store(self, agent_id: str, token: str):
        """Encrypt and store a token."""
        encrypted = self.fernet.encrypt(token.encode())
        self.tokens[agent_id] = encrypted
        logger.info(f"Stored encrypted token for {agent_id}")

    def retrieve(self, agent_id: str) -> str | None:
        """Retrieve and decrypt a token."""
        encrypted = self.tokens.get(agent_id)
        if not encrypted:
            return None
        return self.fernet.decrypt(encrypted).decode()

    def clear(self, agent_id: str):
        """Remove a stored token."""
        if agent_id in self.tokens:
            del self.tokens[agent_id]
            logger.info(f"Cleared stored token for {agent_id}")


# Rotation and Revocation (Section: Rotation and Revocation)

class CredentialRotator:
    """Automatic credential rotation."""

    def __init__(
        self,
        identity_service: IdentityService,
        rotation_interval: timedelta = timedelta(hours=12)
    ):
        self.service = identity_service
        self.interval = rotation_interval
        self.last_rotation: dict[str, datetime] = {}

    async def ensure_fresh(self, agent_id: str) -> str:
        """Ensure agent has a fresh token."""
        last = self.last_rotation.get(agent_id)
        now = datetime.utcnow()

        if not last or (now - last) > self.interval:
            # Revoke old tokens
            self.service.revoke_all_tokens(agent_id)

            # Issue new token
            token = self.service.issue_token(agent_id)
            self.last_rotation[agent_id] = now
            logger.info(f"Rotated credentials for {agent_id}")
            return token

        return self.service.issue_token(agent_id)

    async def emergency_revoke(self, agent_id: str):
        """Emergency revocation on compromise detection."""
        self.service.revoke_all_tokens(agent_id)
        # Also log the incident
        logger.critical(f"Emergency revocation for {agent_id}")


# =============================================================================
# Demo / Main
# =============================================================================

async def main():
    """Demonstrate identity service usage."""

    # Initialize service
    service = IdentityService(secret_key="your-secret-key-here")

    # Register agents
    orchestrator = service.register_agent(
        name="Main Orchestrator",
        role=AgentRole.ORCHESTRATOR
    )
    print(f"Registered: {orchestrator.name} ({orchestrator.agent_id})")
    print(f"  Scopes: {orchestrator.scopes}")

    worker = service.register_agent(
        name="Research Worker",
        role=AgentRole.WORKER,
        scopes={"read:self", "execute:tasks", "read:documents"}
    )
    print(f"Registered: {worker.name} ({worker.agent_id})")

    guardian = service.register_agent(
        name="Safety Guardian",
        role=AgentRole.GUARDIAN
    )
    print(f"Registered: {guardian.name} ({guardian.agent_id})")

    # Issue tokens
    orch_token = service.issue_token(orchestrator.agent_id)
    worker_token = service.issue_token(worker.agent_id)

    # Verify and authorize
    orch_can_delegate = service.authorize(orch_token, 'delegate:tasks')
    print(f"\nOrchestrator can delegate: {orch_can_delegate}")
    wkr_can_delegate = service.authorize(worker_token, 'delegate:tasks')
    print(f"Worker can delegate: {wkr_can_delegate}")
    wkr_can_read = service.authorize(worker_token, 'read:documents')
    print(f"Worker can read documents: {wkr_can_read}")

    # Revoke token
    service.revoke_token(worker_token)
    still_valid = service.verify_token(worker_token) is not None
    print(f"\nAfter revocation, worker token valid: {still_valid}")

if __name__ == "__main__":
    asyncio.run(main())
