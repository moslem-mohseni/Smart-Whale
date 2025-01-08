# web/client/__init__.py
"""
Client Web Interface Module
------------------------
This module implements the main user interface for the system.
It provides a modern, responsive web interface that allows users to:

1. Access AI-powered analysis tools
2. View real-time market data
3. Manage their profiles and preferences
4. Interact with AI models in multiple languages
5. View analytics and reports

The client interface is built with performance and user experience
as top priorities, ensuring smooth operation across different devices
and browsers.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ClientSection(Enum):
    """Main sections of the client interface"""
    DASHBOARD = "dashboard"
    AI_INTERFACE = "ai_interface"
    ANALYTICS = "analytics"
    PROFILE = "profile"


@dataclass
class UserPreferences:
    """User interface preferences"""
    language: str = "en"
    theme: str = "light"
    notifications_enabled: bool = True
    default_view: ClientSection = ClientSection.DASHBOARD


class ClientManager:
    """
    مدیریت رابط کاربری و تجربه کاربری
    این کلاس مسئول مدیریت تعامل کاربر با سیستم است.
    """

    def __init__(self):
        self._active_users = {}
        self._user_preferences = {}

    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """دریافت تنظیمات شخصی‌سازی کاربر"""
        return self._user_preferences.get(user_id, UserPreferences())

    def update_user_preferences(self, user_id: str, preferences: UserPreferences) -> None:
        """به‌روزرسانی تنظیمات کاربر"""
        self._user_preferences[user_id] = preferences