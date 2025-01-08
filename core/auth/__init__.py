# core/auth/__init__.py
"""
Authentication and Authorization Module
-----------------------------------
Manages system security including:
- User authentication
- Session management
- Access control and permissions
- Security policies enforcement
- Token management for API access

This module works closely with the user module to ensure secure access
to system resources.
"""

from enum import Enum
from typing import Optional

class AuthProvider(Enum):
    """Supported authentication providers"""
    INTERNAL = "internal"
    JWT = "jwt"
    OAUTH2 = "oauth2"

class PermissionLevel(Enum):
    """System permission levels"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"