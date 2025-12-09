"""
ATLAN Authentication & RBAC Module
Provides JWT-based auth with role-based access control.
"""

import os
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum

# JWT support (using PyJWT if available, fallback to simple tokens)
try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    print("PyJWT not installed. Using simple token auth. Install with: pip install PyJWT")

# Configuration
SECRET_KEY = os.getenv("ATLAN_SECRET_KEY", secrets.token_hex(32))
TOKEN_EXPIRY_HOURS = int(os.getenv("ATLAN_TOKEN_EXPIRY", "24"))
USERS_FILE = "atlan_users.json"


class Role(str, Enum):
    ADMIN = "admin"      # Full access: ingest, delete, configure, manage users
    USER = "user"        # Read-only: resonate, analytics
    API = "api"          # API access only: resonate endpoint


class User:
    def __init__(self, username: str, password_hash: str, role: Role,
                 email: str = "", created_at: str = None, active: bool = True):
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.email = email
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.active = active

    def to_dict(self):
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "email": self.email,
            "created_at": self.created_at,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            username=data["username"],
            password_hash=data["password_hash"],
            role=Role(data["role"]),
            email=data.get("email", ""),
            created_at=data.get("created_at"),
            active=data.get("active", True)
        )


class UserDatabase:
    """Simple file-based user storage for MVP. Replace with real DB in production."""

    def __init__(self, filepath: str = USERS_FILE):
        self.filepath = filepath
        self.users: dict[str, User] = {}
        self.load()

    def load(self):
        """Load users from file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    self.users = {k: User.from_dict(v) for k, v in data.items()}
                print(f"Loaded {len(self.users)} users from {self.filepath}")
            except Exception as e:
                print(f"Error loading users: {e}")
                self.users = {}
        else:
            self.users = {}

    def save(self):
        """Save users to file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump({k: v.to_dict() for k, v in self.users.items()}, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")

    def get(self, username: str) -> Optional[User]:
        return self.users.get(username)

    def create(self, username: str, password: str, role: Role, email: str = "") -> User:
        if username in self.users:
            raise ValueError(f"User '{username}' already exists")

        password_hash = self._hash_password(password)
        user = User(username=username, password_hash=password_hash, role=role, email=email)
        self.users[username] = user
        self.save()
        return user

    def verify_password(self, username: str, password: str) -> bool:
        user = self.get(username)
        if not user or not user.active:
            return False
        return self._hash_password(password) == user.password_hash

    def update_password(self, username: str, new_password: str):
        user = self.get(username)
        if user:
            user.password_hash = self._hash_password(new_password)
            self.save()

    def delete(self, username: str):
        if username in self.users:
            del self.users[username]
            self.save()

    def list_users(self) -> List[dict]:
        return [
            {"username": u.username, "role": u.role.value, "email": u.email,
             "active": u.active, "created_at": u.created_at}
            for u in self.users.values()
        ]

    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salted = f"{SECRET_KEY}:{password}"
        return hashlib.sha256(salted.encode()).hexdigest()


class TokenManager:
    """Manages JWT tokens or simple bearer tokens."""

    @staticmethod
    def create_token(username: str, role: Role) -> str:
        """Create access token."""
        if HAS_JWT:
            payload = {
                "sub": username,
                "role": role.value,
                "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
                "iat": datetime.utcnow()
            }
            return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        else:
            # Simple token fallback
            token_data = f"{username}:{role.value}:{secrets.token_hex(16)}"
            return hashlib.sha256(f"{SECRET_KEY}:{token_data}".encode()).hexdigest()

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """Verify token and return payload."""
        if HAS_JWT:
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                return {"username": payload["sub"], "role": Role(payload["role"])}
            except jwt.ExpiredSignatureError:
                return None
            except jwt.InvalidTokenError:
                return None
        else:
            # For simple tokens, we'd need to store them - not implemented
            return None


# Global instances
user_db = UserDatabase()
token_manager = TokenManager()


def setup_superuser(username: str = "admin", password: str = None) -> tuple[str, str]:
    """
    Create superuser account. Called during initial setup.
    Returns (username, password) - password is generated if not provided.
    """
    if not password:
        password = secrets.token_urlsafe(16)

    # Check if superuser exists
    existing = user_db.get(username)
    if existing:
        if existing.role == Role.ADMIN:
            print(f"Superuser '{username}' already exists")
            return username, "[existing - password not changed]"
        else:
            # Upgrade to admin
            existing.role = Role.ADMIN
            user_db.save()
            print(f"Upgraded '{username}' to admin")
            return username, "[existing - upgraded to admin]"

    # Create new superuser
    user_db.create(username=username, password=password, role=Role.ADMIN, email="admin@local")
    print(f"Created superuser: {username}")
    return username, password


def check_permission(user: User, required_role: Role) -> bool:
    """Check if user has required permission level."""
    role_hierarchy = {Role.ADMIN: 3, Role.USER: 2, Role.API: 1}
    return role_hierarchy.get(user.role, 0) >= role_hierarchy.get(required_role, 0)


# FastAPI Dependencies
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

security_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    bearer: HTTPAuthorizationCredentials = Security(security_bearer),
    api_key: str = Security(api_key_header)
) -> User:
    """
    Authenticate user via:
    1. Bearer token (JWT)
    2. API Key header
    """
    # Try Bearer token first
    if bearer:
        payload = token_manager.verify_token(bearer.credentials)
        if payload:
            user = user_db.get(payload["username"])
            if user and user.active:
                return user

    # Try API Key
    if api_key:
        # Legacy API key support (for backward compatibility)
        legacy_key = os.getenv("ATLAN_API_KEY", "atlan-secret-key-123")
        if api_key == legacy_key:
            # Return a pseudo-admin user for legacy key
            return User(
                username="legacy_api",
                password_hash="",
                role=Role.ADMIN,
                email="legacy@api"
            )

        # Check if API key matches any user's token
        for user in user_db.users.values():
            if user.active and api_key == token_manager.create_token(user.username, user.role):
                return user

    raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def require_role(required: Role):
    """Dependency factory for role-based access control."""
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not check_permission(user, required):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required.value}"
            )
        return user
    return role_checker


# Convenience dependencies
require_admin = require_role(Role.ADMIN)
require_user = require_role(Role.USER)
require_api = require_role(Role.API)
