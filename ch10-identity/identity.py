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

__all__ = [
    "PerAgentRateLimiter",
    "TokenRateLimitExceeded",
    "AgentToken",
    "SimpleTokenAuth",
    "ScopedToken",
    "ScopedAuth",
    "JWTAuth",
    "AsymmetricJWTAuth",
    "IdentityType",
    "Identity",
    "AgentRole",
    "AgentIdentity",
    "TokenInfo",
    "IdentityService",
    "AuthenticationError",
    "AuthorizationError",
    "AuthMiddleware",
    "AuthenticatedAgent",
    "TokenManager",
    "AuditEntry",
    "AuditLog",
    "SecureAgent",
    "HierarchicalIdentityService",
    "ImpersonationService",
    "DelegationGrant",
    "DelegationService",
    "LeastPrivilegeAssigner",
    "SecureTokenStorage",
    "CredentialRotator",
]

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from secrets import token_urlsafe
from typing import Any
import json


# =============================================================================
# Rate Limiting for Token Issuance
# =============================================================================

class PerAgentRateLimiter:
    """Rate limiter that tracks limits per agent ID."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        """Initialize per-agent rate limiter.

        Args:
            max_requests: Maximum requests per agent per window (default: 10 per minute).
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._agent_requests: dict[str, list[float]] = {}

    def allow(self, agent_id: str) -> bool:
        """Check if a request is allowed for the given agent.

        Args:
            agent_id: The agent ID to check rate limit for.

        Returns:
            True if the request is allowed, False otherwise.
        """
        now = time.time()

        # Initialize if needed
        if agent_id not in self._agent_requests:
            self._agent_requests[agent_id] = []

        # Remove expired timestamps
        self._agent_requests[agent_id] = [
            t for t in self._agent_requests[agent_id]
            if now - t < self.window_seconds
        ]

        if len(self._agent_requests[agent_id]) < self.max_requests:
            self._agent_requests[agent_id].append(now)
            return True
        return False

    def reset(self, agent_id: str | None = None):
        """Reset rate limiter for a specific agent or all agents.

        Args:
            agent_id: If provided, reset only this agent. Otherwise reset all.
        """
        if agent_id:
            self._agent_requests.pop(agent_id, None)
        else:
            self._agent_requests.clear()


class TokenRateLimitExceeded(Exception):
    """Exception raised when token issuance rate limit is exceeded."""
    pass

# =============================================================================
# Metrics Collection
# =============================================================================

try:
    from common.metrics import MetricsCollector
    _identity_metrics = MetricsCollector(namespace="identity")
except ImportError:
    _identity_metrics = None

# =============================================================================
# Circuit Breaker for Identity Operations
# =============================================================================

try:
    from common.resilience import CircuitBreaker, CircuitBreakerOpen
    _identity_circuit_breaker = CircuitBreaker(
        name="identity_service",
        failure_threshold=5,
        recovery_timeout=60.0
    )
except ImportError:
    # Inline fallback implementation
    class CircuitBreakerOpen(Exception):
        """Exception raised when circuit breaker is open."""
        def __init__(self, breaker_name: str, time_until_retry: float):
            self.breaker_name = breaker_name
            self.time_until_retry = time_until_retry
            super().__init__(
                f"Circuit breaker '{breaker_name}' is OPEN. "
                f"Retry in {time_until_retry:.1f}s"
            )

    class CircuitBreaker:
        """Minimal fallback circuit breaker."""
        def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
            self.name = name
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._is_open = False

        def _check_state(self):
            if self._is_open:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._is_open = False
                    self._failure_count = 0
                else:
                    raise CircuitBreakerOpen(self.name, self.recovery_timeout - elapsed)

        def record_success(self):
            self._failure_count = 0
            self._is_open = False

        def record_failure(self):
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._is_open = True

        async def __aenter__(self):
            self._check_state()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.record_failure()
            else:
                self.record_success()
            return False

    _identity_circuit_breaker = CircuitBreaker(
        name="identity_service",
        failure_threshold=5,
        recovery_timeout=60.0
    )

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

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
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        )

        return token  # Return unhashed token to agent

    def authenticate(self, token: str) -> str | None:
        """Authenticate a token and return agent_id if valid."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        agent_token = self.tokens.get(token_hash)
        if not agent_token:
            return None

        if agent_token.expires_at and agent_token.expires_at < datetime.now(timezone.utc):
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
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )

        return token

    def authorize(self, token: str, required_scope: str) -> bool:
        """Check if token has required scope."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        agent_token = self.tokens.get(token_hash)

        if not agent_token:
            return False

        if agent_token.expires_at and agent_token.expires_at < datetime.now(timezone.utc):
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
        now = datetime.now(timezone.utc)
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

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    serialization = None
    rsa = None


class AsymmetricJWTAuth:
    def __init__(self, private_key_pem: bytes = None, public_key_pem: bytes = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required: pip install cryptography")
        if private_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem, password=None
            )
            self.public_key = self.private_key.public_key()
        elif public_key_pem:
            # Verify-only instance: load the supplied public key and
            # keep no private key, so externally issued tokens are
            # verified against the correct keypair.
            self.private_key = None
            self.public_key = serialization.load_pem_public_key(
                public_key_pem
            )
        else:
            # Generate new key pair (3072 bits per NIST SP 800-57)
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=3072
            )
            self.public_key = self.private_key.public_key()

    def create_token(self, agent_id: str, scopes: list[str],
                      ttl_hours: int = 24) -> str:
        """Create a JWT signed with private key."""
        if self.private_key is None:
            raise ValueError(
                "This instance is verify-only (constructed with a "
                "public key); it cannot create tokens"
            )
        now = datetime.now(timezone.utc)
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
    - Token revocation with automatic expiry cleanup
    - Scope-based authorization
    - Per-agent rate limiting on token issuance (10 tokens/min/agent)

    Requires: pip install PyJWT
    """

    def __init__(self, secret_key: str, issuer: str = "agent-system",
                 max_agents: int = 10000, max_tokens: int = 100000,
                 token_rate_limit: int = 10, rate_limit_window: float = 60.0):
        """Initialize the identity service.

        Args:
            secret_key: Secret key for JWT signing.
            issuer: Token issuer identifier.
            max_agents: Maximum number of registered agents.
            max_tokens: Maximum number of active tokens.
            token_rate_limit: Max token issuances per agent per window (default: 10).
            rate_limit_window: Rate limit window in seconds (default: 60.0).
        """
        if not JWT_AVAILABLE:
            raise ImportError(
                "PyJWT is required for IdentityService. Install with: pip install PyJWT"
            )
        self.secret_key = secret_key
        self.issuer = issuer
        self.agents: dict[str, AgentIdentity] = {}
        self.tokens: dict[str, TokenInfo] = {}
        self._max_agents = max_agents
        self._max_tokens = max_tokens
        # Store revoked tokens with their expiry time for cleanup
        self._revoked_tokens: dict[str, datetime] = {}
        # Per-agent rate limiter for token issuance
        self._token_rate_limiter = PerAgentRateLimiter(
            max_requests=token_rate_limit,
            window_seconds=rate_limit_window
        )
        logger.info(f"IdentityService initialized with issuer: {issuer}")

    @property
    def revoked_tokens(self) -> set[str]:
        """Get currently revoked tokens, auto-pruning expired ones."""
        now = datetime.now(timezone.utc)
        # Remove expired revocations (token would be invalid anyway)
        expired = [tid for tid, exp in self._revoked_tokens.items() if exp < now]
        for tid in expired:
            del self._revoked_tokens[tid]
        return set(self._revoked_tokens.keys())

    def _revoke_token_id(self, token_id: str, expires_at: datetime) -> None:
        """Internal method to revoke a token by ID."""
        self._revoked_tokens[token_id] = expires_at

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

        if len(self.agents) >= self._max_agents:
            raise ValueError(f"Maximum agents ({self._max_agents}) reached")
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
                    self._revoke_token_id(token_id, info.expires_at)
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
        additional_claims: dict | None = None,
        bypass_rate_limit: bool = False
    ) -> str | None:
        """Issue a JWT token for an agent.

        Args:
            agent_id: The agent ID to issue a token for.
            ttl_hours: Token time-to-live in hours (default: 24).
            additional_claims: Optional additional JWT claims.
            bypass_rate_limit: If True, skip rate limiting (for internal use).

        Returns:
            The issued JWT token, or None if agent not found.

        Raises:
            TokenRateLimitExceeded: If rate limit is exceeded for this agent.
        """
        # Check per-agent rate limit (10 tokens per minute per agent)
        if not bypass_rate_limit and not self._token_rate_limiter.allow(agent_id):
            logger.warning(f"Token issuance rate limit exceeded for agent {agent_id}")
            raise TokenRateLimitExceeded(
                f"Rate limit exceeded for agent {agent_id}. "
                "Maximum 10 tokens per minute allowed."
            )

        identity = self.agents.get(agent_id)
        if not identity:
            logger.warning(f"Token issuance failed: agent {agent_id} not found")
            return None

        now = datetime.now(timezone.utc)
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

        # Track token for revocation (with cleanup of expired tokens)
        if len(self.tokens) >= self._max_tokens:
            # Remove expired tokens
            expired = [tid for tid, info in self.tokens.items()
                      if info.expires_at < now]
            for tid in expired:
                del self.tokens[tid]
        self.tokens[token_id] = TokenInfo(
            token_id=token_id,
            agent_id=agent_id,
            issued_at=now,
            expires_at=now + timedelta(hours=ttl_hours)
        )

        logger.info(f"Issued token for agent {agent_id}, expires in {ttl_hours} hours")

        # Track metrics
        if _identity_metrics:
            _identity_metrics.increment("tokens_issued", labels={"role": identity.role.value})

        return token

    def verify_token(self, token: str) -> dict | None:
        """Verify a token and return its claims."""
        start_time = time.time()
        result = None
        status = "success"

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
                status = "revoked"
                return None

            result = payload
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: expired")
            status = "expired"
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {e}")
            status = "invalid"
            return None
        finally:
            # Track metrics
            if _identity_metrics:
                latency = time.time() - start_time
                _identity_metrics.increment("token_validations", labels={"status": status})
                _identity_metrics.observe("validation_latency", latency)

    def revoke_token(self, token: str):
        """Revoke a specific token."""
        try:
            # Decode without verification to get expiry even if token is invalid
            payload = jwt.decode(token, options={"verify_signature": False})
            if "jti" in payload:
                # Store with expiry time so we can clean up later
                exp = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
                self._revoked_tokens[payload["jti"]] = exp
                logger.info(f"Revoked token {payload['jti']}")

                # Track metrics
                if _identity_metrics:
                    _identity_metrics.increment("tokens_revoked")
        except jwt.InvalidTokenError:
            pass  # Invalid token, nothing to revoke

    def revoke_all_tokens(self, agent_id: str):
        """Revoke all tokens for an agent."""
        count = 0
        for token_id, info in self.tokens.items():
            if info.agent_id == agent_id:
                self._revoke_token_id(token_id, info.expires_at)
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
        """Authenticate a request and return the agent identity.

        Uses circuit breaker to fail fast if identity service is degraded.

        Raises:
            CircuitBreakerOpen: If identity service circuit breaker is open.
        """
        # Get token from header
        auth_header = request.get("headers", {}).get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Use circuit breaker for token validation
        async with _identity_circuit_breaker:
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
        self._shutdown = False

    async def get_token(self) -> str:
        """Get a valid token, refreshing if needed."""
        if self.current_token:
            payload = self.service.verify_token(self.current_token)
            if payload:
                exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
                if exp - datetime.now(timezone.utc) > self.refresh_threshold:
                    return self.current_token

        # Need new token
        self.current_token = self.service.issue_token(self.agent_id)
        logger.info(f"Refreshed token for agent {self.agent_id}")
        return self.current_token

    def stop(self):
        """Signal the refresh loop to stop."""
        self._shutdown = True

    async def refresh_loop(self):
        """Background task to keep token fresh."""
        while not self._shutdown:
            await asyncio.sleep(3600)  # Check every hour
            if not self._shutdown:
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
    # In-memory cap; durable retention requires a storage backend.
    MAX_IN_MEMORY_ENTRIES = 10000

    def __init__(self, storage: Storage | None = None):
        self.entries: deque = deque(maxlen=self.MAX_IN_MEMORY_ENTRIES)
        self.storage = storage
        if storage is None:
            logger.warning(
                "AuditLog has no durable storage backend; only the "
                f"most recent {self.MAX_IN_MEMORY_ENTRIES} entries "
                "are kept in memory"
            )

    def log(self, agent_id: str, action: str, resource: str,
            outcome: str, details: dict = None):
        """Log an agent action."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
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
        results = list(self.entries)

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
        return datetime.now(timezone.utc) < self.expires_at

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
            expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
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

try:
    from cryptography.fernet import Fernet
    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False
    Fernet = None


class SecureTokenStorage:
    """Store tokens securely."""

    def __init__(self, encryption_key: bytes):
        if not FERNET_AVAILABLE:
            raise ImportError("cryptography package required: pip install cryptography")
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
        now = datetime.now(timezone.utc)

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

async def main(demo_mode: bool = False):
    """Demonstrate identity service usage."""
    import os

    # Initialize service with secure key from environment
    secret_key = os.environ.get("IDENTITY_SECRET_KEY")
    if not secret_key:
        if demo_mode:
            secret_key = "DEMO-ONLY-DO-NOT-USE-IN-PRODUCTION"
            print("WARNING: Using demo secret key. Set IDENTITY_SECRET_KEY in production.")
        else:
            raise ValueError(
                "IDENTITY_SECRET_KEY environment variable required. "
                "Set it to a secure random string (at least 32 characters)."
            )
    service = IdentityService(secret_key=secret_key)

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
    # Run in demo mode by default for easier testing
    asyncio.run(main(demo_mode=True))
