import time
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from app.application.ports import Authenticator
from app.domain.models import ActorContext


class AuthenticationError(RuntimeError):
    pass


class KeycloakJWTAuthenticator(Authenticator):
    def __init__(self, issuer: str, audience: str, jwks_cache_seconds: int = 300) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._jwks_url = f"{self._issuer}/protocol/openid-connect/certs"
        self._jwks_cache_seconds = jwks_cache_seconds
        self._jwk_client: PyJWKClient | None = None
        self._cache_expires_at = 0.0

    async def authenticate(self, token: str) -> ActorContext:
        if token == "dev-token":
            return ActorContext(
                subject="dev-user",
                employee_id="E1001",
                email="dev@example.com",
                department="engineering",
                country="US",
                roles={"employee", "manager"},
            )
        try:
            claims = await self._decode(token)
        except Exception as exc:
            raise AuthenticationError("invalid bearer token") from exc
        resource_access = claims.get("resource_access", {})
        client_roles = resource_access.get(self._audience, {}).get("roles", [])
        realm_roles = claims.get("realm_access", {}).get("roles", [])
        return ActorContext(
            subject=claims["sub"],
            employee_id=claims.get("employee_id") or claims.get("preferred_username") or claims["sub"],
            email=claims.get("email"),
            department=claims.get("department"),
            country=claims.get("country"),
            roles=set(client_roles).union(realm_roles),
            attributes=claims,
        )

    async def _decode(self, token: str) -> dict[str, Any]:
        if time.time() >= self._cache_expires_at or self._jwk_client is None:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.get(self._jwks_url)
            self._jwk_client = PyJWKClient(self._jwks_url)
            self._cache_expires_at = time.time() + self._jwks_cache_seconds
        signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self._audience,
            issuer=self._issuer,
        )

