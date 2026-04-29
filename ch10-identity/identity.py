"""
Chapter 10: Agent Identity Systems
===================================
Implementation of per-agent identity management for enterprise deployments.

This module provides:
1. Unique cryptographic identities for each agent
2. Credential lifecycle management
3. Permission-based access control
4. Audit trail for identity operations

Reference: Google Cloud Agent Identity - "assigns every agent a unique
cryptographic ID for complete traceability and auditing"
"""

import asyncio
import json
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


# =============================================================================
# Identity Types and Permissions
# =============================================================================

class PermissionScope(str, Enum):
    """Standard permission scopes for agents"""
    READ_DATA = "data:read"
    WRITE_DATA = "data:write"
    EXECUTE_CODE = "code:execute"
    INVOKE_TOOLS = "tools:invoke"
    INVOKE_AGENTS = "agents:invoke"
    MANAGE_SESSIONS = "sessions:manage"
    ACCESS_SECRETS = "secrets:access"
    ADMIN = "admin:*"


class IdentityStatus(str, Enum):
    """Status of an agent identity"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class AgentIdentity:
    """
    Unique identity for an agent.

    Each agent gets a cryptographically verifiable identity that:
    - Is globally unique
    - Has defined permissions
    - Has a lifecycle (creation, rotation, revocation)
    - Supports full audit trail
    """
    agent_id: str
    name: str
    owner: str
    permissions: list[PermissionScope]
    public_key_pem: str
    status: IdentityStatus
    created_at: str
    expires_at: str
    metadata: dict = field(default_factory=dict)
    last_used: str | None = None
    revoked_at: str | None = None
    revocation_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "owner": self.owner,
            "permissions": [p.value for p in self.permissions],
            "public_key_pem": self.public_key_pem,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "last_used": self.last_used,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason
        }

    def has_permission(self, scope: PermissionScope) -> bool:
        """Check if identity has a specific permission"""
        if PermissionScope.ADMIN in self.permissions:
            return True
        return scope in self.permissions

    def is_valid(self) -> bool:
        """Check if identity is currently valid"""
        if self.status != IdentityStatus.ACTIVE:
            return False
        # Handle 'Z' suffix for Python <3.11 compatibility
        expires_str = self.expires_at.replace('Z', '+00:00')
        if datetime.fromisoformat(expires_str) < datetime.now(timezone.utc):
            return False
        return True


@dataclass
class Credential:
    """Credential issued to an agent for authentication"""
    credential_id: str
    agent_id: str
    token_hash: str  # Hash of the actual token
    issued_at: str
    expires_at: str
    scopes: list[str]
    is_revoked: bool = False

    def to_dict(self) -> dict:
        return {
            "credential_id": self.credential_id,
            "agent_id": self.agent_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "scopes": self.scopes,
            "is_revoked": self.is_revoked
        }


@dataclass
class AuditEvent:
    """Audit event for identity operations"""
    event_id: str
    event_type: str
    agent_id: str
    actor: str
    timestamp: str
    details: dict
    ip_address: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "details": self.details,
            "ip_address": self.ip_address
        }


# =============================================================================
# Storage Interfaces
# =============================================================================

class IdentityStore:
    """In-memory identity store (replace with database in production)"""

    def __init__(self):
        self._identities: dict[str, AgentIdentity] = {}
        self._credentials: dict[str, Credential] = {}
        self._audit_log: list[AuditEvent] = []

    async def store_identity(self, identity: AgentIdentity):
        self._identities[identity.agent_id] = identity

    async def get_identity(self, agent_id: str) -> AgentIdentity | None:
        return self._identities.get(agent_id)

    async def update_identity(self, identity: AgentIdentity):
        self._identities[identity.agent_id] = identity

    async def list_identities(
        self,
        owner: str | None = None,
        status: IdentityStatus | None = None
    ) -> list[AgentIdentity]:
        results = list(self._identities.values())
        if owner:
            results = [i for i in results if i.owner == owner]
        if status:
            results = [i for i in results if i.status == status]
        return results

    async def store_credential(self, credential: Credential):
        self._credentials[credential.credential_id] = credential

    async def get_credential(self, credential_id: str) -> Credential | None:
        return self._credentials.get(credential_id)

    async def get_credentials_for_agent(self, agent_id: str) -> list[Credential]:
        return [c for c in self._credentials.values() if c.agent_id == agent_id]

    async def log_audit_event(self, event: AuditEvent):
        self._audit_log.append(event)

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        since: datetime | None = None
    ) -> list[AuditEvent]:
        events = self._audit_log
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if since:
            since_str = since.isoformat()
            events = [e for e in events if e.timestamp >= since_str]
        return events


class KeyVault:
    """Secure key storage (replace with HSM/KMS in production)"""

    def __init__(self):
        self._keys: dict[str, bytes] = {}

    async def store_private_key(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
        metadata: dict
    ):
        # Serialize and store (in production, use proper encryption)
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        self._keys[key_id] = pem

    async def get_private_key(self, key_id: str) -> rsa.RSAPrivateKey | None:
        pem = self._keys.get(key_id)
        if not pem:
            return None
        return serialization.load_pem_private_key(pem, password=None)

    async def delete_key(self, key_id: str):
        if key_id in self._keys:
            del self._keys[key_id]


# =============================================================================
# Identity Service
# =============================================================================

class AgentIdentityService:
    """
    Service for managing agent identities.

    Provides:
    1. Identity provisioning with unique credentials
    2. Token issuance and validation
    3. Credential rotation
    4. Revocation without affecting other agents
    5. Full audit trail
    """

    def __init__(
        self,
        store: IdentityStore,
        key_vault: KeyVault,
        jwt_secret: str,
        default_ttl_days: int = 90
    ):
        self.store = store
        self.key_vault = key_vault
        self.jwt_secret = jwt_secret
        self.default_ttl_days = default_ttl_days

    # -------------------------------------------------------------------------
    # Identity Lifecycle
    # -------------------------------------------------------------------------

    async def provision_identity(
        self,
        name: str,
        owner: str,
        permissions: list[PermissionScope],
        ttl_days: int | None = None,
        metadata: dict | None = None,
        actor: str = "system"
    ) -> tuple[AgentIdentity, str]:
        """
        Provision a new agent identity.

        Returns:
            Tuple of (AgentIdentity, initial_token)
        """
        # Generate unique agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"

        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Serialize public key
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        # Store private key in vault
        await self.key_vault.store_private_key(
            key_id=f"{agent_id}/private_key",
            private_key=private_key,
            metadata={"owner": owner, "name": name}
        )

        # Calculate expiration
        ttl = ttl_days or self.default_ttl_days
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=ttl)

        # Create identity
        identity = AgentIdentity(
            agent_id=agent_id,
            name=name,
            owner=owner,
            permissions=permissions,
            public_key_pem=public_key_pem,
            status=IdentityStatus.ACTIVE,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            metadata=metadata or {}
        )

        # Store identity
        await self.store.store_identity(identity)

        # Issue initial token
        token, _ = await self._issue_token(identity, actor)

        # Audit log
        await self._audit(
            event_type="identity_provisioned",
            agent_id=agent_id,
            actor=actor,
            details={
                "name": name,
                "owner": owner,
                "permissions": [p.value for p in permissions],
                "expires_at": expires_at.isoformat()
            }
        )

        return identity, token

    async def get_identity(self, agent_id: str) -> AgentIdentity | None:
        """Get an agent's identity"""
        return await self.store.get_identity(agent_id)

    async def suspend_identity(
        self,
        agent_id: str,
        reason: str,
        actor: str
    ) -> AgentIdentity:
        """Temporarily suspend an agent identity"""
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        identity.status = IdentityStatus.SUSPENDED
        await self.store.update_identity(identity)

        # Revoke all active credentials
        credentials = await self.store.get_credentials_for_agent(agent_id)
        for cred in credentials:
            cred.is_revoked = True
            await self.store.store_credential(cred)

        await self._audit(
            event_type="identity_suspended",
            agent_id=agent_id,
            actor=actor,
            details={"reason": reason}
        )

        return identity

    async def revoke_identity(
        self,
        agent_id: str,
        reason: str,
        actor: str
    ) -> AgentIdentity:
        """
        Permanently revoke an agent identity.

        This is irreversible - the agent will need a new identity.
        """
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        identity.status = IdentityStatus.REVOKED
        identity.revoked_at = datetime.now(timezone.utc).isoformat()
        identity.revocation_reason = reason
        await self.store.update_identity(identity)

        # Revoke all credentials
        credentials = await self.store.get_credentials_for_agent(agent_id)
        for cred in credentials:
            cred.is_revoked = True
            await self.store.store_credential(cred)

        # Delete private key
        await self.key_vault.delete_key(f"{agent_id}/private_key")

        await self._audit(
            event_type="identity_revoked",
            agent_id=agent_id,
            actor=actor,
            details={"reason": reason}
        )

        return identity

    async def reactivate_identity(
        self,
        agent_id: str,
        actor: str
    ) -> AgentIdentity:
        """Reactivate a suspended identity"""
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        if identity.status == IdentityStatus.REVOKED:
            raise ValueError("Cannot reactivate revoked identity")

        identity.status = IdentityStatus.ACTIVE
        await self.store.update_identity(identity)

        await self._audit(
            event_type="identity_reactivated",
            agent_id=agent_id,
            actor=actor,
            details={}
        )

        return identity

    # -------------------------------------------------------------------------
    # Token Management
    # -------------------------------------------------------------------------

    async def issue_token(
        self,
        agent_id: str,
        scopes: list[PermissionScope] | None = None,
        ttl_hours: int = 24,
        actor: str = "system"
    ) -> str:
        """Issue a new authentication token for an agent"""
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        if not identity.is_valid():
            raise ValueError(f"Identity is not valid: {identity.status.value}")

        # Use agent's permissions if no scopes specified
        if scopes is None:
            scopes = identity.permissions

        # Verify requested scopes are allowed
        for scope in scopes:
            if not identity.has_permission(scope):
                raise ValueError(f"Agent does not have permission: {scope.value}")

        token, credential = await self._issue_token(identity, actor, scopes, ttl_hours)

        await self._audit(
            event_type="token_issued",
            agent_id=agent_id,
            actor=actor,
            details={
                "credential_id": credential.credential_id,
                "scopes": [s.value for s in scopes],
                "expires_at": credential.expires_at
            }
        )

        return token

    async def _issue_token(
        self,
        identity: AgentIdentity,
        actor: str,
        scopes: list[PermissionScope] | None = None,
        ttl_hours: int = 24
    ) -> tuple[str, Credential]:
        """Internal token issuance"""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)

        # Generate token
        token_id = str(uuid.uuid4())
        payload = {
            "sub": identity.agent_id,
            "name": identity.name,
            "owner": identity.owner,
            "scopes": [s.value for s in (scopes or identity.permissions)],
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": token_id
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")

        # Store credential record
        credential = Credential(
            credential_id=token_id,
            agent_id=identity.agent_id,
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            issued_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            scopes=[s.value for s in (scopes or identity.permissions)]
        )
        await self.store.store_credential(credential)

        return token, credential

    async def validate_token(self, token: str) -> dict:
        """
        Validate an authentication token.

        Returns decoded payload if valid, raises exception otherwise.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")

        # Check credential not revoked
        credential = await self.store.get_credential(payload["jti"])
        if not credential:
            raise ValueError("Credential not found")
        if credential.is_revoked:
            raise ValueError("Credential has been revoked")

        # Check identity still valid
        identity = await self.store.get_identity(payload["sub"])
        if not identity or not identity.is_valid():
            raise ValueError("Agent identity is no longer valid")

        # Update last used
        identity.last_used = datetime.now(timezone.utc).isoformat()
        await self.store.update_identity(identity)

        return payload

    async def revoke_token(
        self,
        credential_id: str,
        actor: str
    ):
        """Revoke a specific token"""
        credential = await self.store.get_credential(credential_id)
        if not credential:
            raise ValueError(f"Credential not found: {credential_id}")

        credential.is_revoked = True
        await self.store.store_credential(credential)

        await self._audit(
            event_type="token_revoked",
            agent_id=credential.agent_id,
            actor=actor,
            details={"credential_id": credential_id}
        )

    async def rotate_credentials(
        self,
        agent_id: str,
        actor: str
    ) -> str:
        """
        Rotate credentials by revoking all existing and issuing new.

        This is useful for periodic security rotation or after
        a suspected compromise.
        """
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        # Revoke all existing credentials
        credentials = await self.store.get_credentials_for_agent(agent_id)
        for cred in credentials:
            cred.is_revoked = True
            await self.store.store_credential(cred)

        # Issue new token
        new_token = await self.issue_token(agent_id, actor=actor)

        await self._audit(
            event_type="credentials_rotated",
            agent_id=agent_id,
            actor=actor,
            details={"credentials_revoked": len(credentials)}
        )

        return new_token

    # -------------------------------------------------------------------------
    # Permission Management
    # -------------------------------------------------------------------------

    async def grant_permission(
        self,
        agent_id: str,
        permission: PermissionScope,
        actor: str
    ) -> AgentIdentity:
        """Grant additional permission to an agent"""
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        if permission not in identity.permissions:
            identity.permissions.append(permission)
            await self.store.update_identity(identity)

            await self._audit(
                event_type="permission_granted",
                agent_id=agent_id,
                actor=actor,
                details={"permission": permission.value}
            )

        return identity

    async def revoke_permission(
        self,
        agent_id: str,
        permission: PermissionScope,
        actor: str
    ) -> AgentIdentity:
        """Revoke a permission from an agent"""
        identity = await self.store.get_identity(agent_id)
        if not identity:
            raise ValueError(f"Identity not found: {agent_id}")

        if permission in identity.permissions:
            identity.permissions.remove(permission)
            await self.store.update_identity(identity)

            await self._audit(
                event_type="permission_revoked",
                agent_id=agent_id,
                actor=actor,
                details={"permission": permission.value}
            )

        return identity

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    async def _audit(
        self,
        event_type: str,
        agent_id: str,
        actor: str,
        details: dict,
        ip_address: str | None = None
    ):
        """Record audit event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            agent_id=agent_id,
            actor=actor,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
            ip_address=ip_address
        )
        await self.store.log_audit_event(event)

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        since: datetime | None = None
    ) -> list[AuditEvent]:
        """Get audit log for an agent or all agents"""
        return await self.store.get_audit_log(agent_id, since)


# =============================================================================
# Authentication Middleware
# =============================================================================

class AgentAuthenticator:
    """
    Middleware for authenticating agent requests.

    Use this in API endpoints to validate agent identity.
    """

    def __init__(self, identity_service: AgentIdentityService):
        self.identity_service = identity_service

    async def authenticate(self, token: str) -> AgentIdentity:
        """
        Authenticate a request using agent token.

        Returns the agent's identity if valid.
        """
        payload = await self.identity_service.validate_token(token)
        identity = await self.identity_service.get_identity(payload["sub"])
        if not identity:
            raise ValueError("Agent identity not found")
        return identity

    async def authorize(
        self,
        token: str,
        required_scope: PermissionScope
    ) -> AgentIdentity:
        """
        Authenticate and authorize for a specific scope.

        Returns identity if authorized, raises exception otherwise.
        """
        identity = await self.authenticate(token)

        if not identity.has_permission(required_scope):
            raise PermissionError(
                f"Agent {identity.agent_id} does not have permission: {required_scope.value}"
            )

        return identity


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate identity service usage"""

    print("=" * 60)
    print("Agent Identity Service Demo")
    print("=" * 60)

    # Initialize service
    store = IdentityStore()
    key_vault = KeyVault()

    # In production, load JWT secret from environment or secrets manager
    # NEVER hardcode secrets in production code
    import os
    jwt_secret = os.environ.get("JWT_SECRET", secrets.token_urlsafe(32))
    if "JWT_SECRET" not in os.environ:
        print("   WARNING: Using generated JWT secret. Set JWT_SECRET env var in production.")

    service = AgentIdentityService(
        store=store,
        key_vault=key_vault,
        jwt_secret=jwt_secret
    )
    auth = AgentAuthenticator(service)

    # 1. Provision a new agent
    print("\n1. Provisioning new agent identity...")
    identity, token = await service.provision_identity(
        name="procurement-agent",
        owner="supply-chain-team",
        permissions=[
            PermissionScope.READ_DATA,
            PermissionScope.WRITE_DATA,
            PermissionScope.INVOKE_TOOLS
        ],
        metadata={"environment": "production", "version": "1.0"}
    )
    print(f"   Agent ID: {identity.agent_id}")
    print(f"   Status: {identity.status.value}")
    print(f"   Permissions: {[p.value for p in identity.permissions]}")
    print(f"   Token: {token[:50]}...")

    # 2. Validate token
    print("\n2. Validating token...")
    try:
        validated_identity = await auth.authenticate(token)
        print(f"   Token valid for: {validated_identity.name}")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Authorize for specific scope
    print("\n3. Authorizing for scope...")
    try:
        await auth.authorize(token, PermissionScope.READ_DATA)
        print("   Authorized for data:read")

        await auth.authorize(token, PermissionScope.ACCESS_SECRETS)
        print("   Authorized for secrets:access")
    except PermissionError as e:
        print(f"   Permission denied: {e}")

    # 4. Grant new permission
    print("\n4. Granting new permission...")
    identity = await service.grant_permission(
        identity.agent_id,
        PermissionScope.INVOKE_AGENTS,
        actor="admin"
    )
    print(f"   New permissions: {[p.value for p in identity.permissions]}")

    # 5. Rotate credentials
    print("\n5. Rotating credentials...")
    new_token = await service.rotate_credentials(identity.agent_id, actor="security-bot")
    print(f"   New token: {new_token[:50]}...")

    # Verify old token is invalid
    try:
        await auth.authenticate(token)
        print("   Old token still valid (unexpected!)")
    except ValueError as e:
        print(f"   Old token correctly rejected: {e}")

    # 6. View audit log
    print("\n6. Audit Log:")
    events = await service.get_audit_log(identity.agent_id)
    for event in events[-5:]:
        print(f"   [{event.timestamp}] {event.event_type} by {event.actor}")

    # 7. Revoke identity
    print("\n7. Revoking identity...")
    identity = await service.revoke_identity(
        identity.agent_id,
        reason="Demo cleanup",
        actor="admin"
    )
    print(f"   Status: {identity.status.value}")
    print(f"   Revoked at: {identity.revoked_at}")


if __name__ == "__main__":
    asyncio.run(main())
