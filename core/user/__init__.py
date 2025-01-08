# core/user/__init__.py
"""
User Management Module
-------------------
Handles all user-related operations including:
- User registration and profile management
- User preferences and settings
- User activity tracking
- Role management and permissions

This module serves as the central point for user data management and enforces
user-related business rules and policies.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class UserProfile:
    """Base user profile structure"""
    id: str
    username: str
    email: str
    created_at: datetime
    active: bool = True
    preferences: dict = None
    roles: List[str] = None