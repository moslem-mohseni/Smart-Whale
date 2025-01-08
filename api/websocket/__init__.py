# api/websocket/__init__.py
"""
WebSocket API Module
-----------------
Implements real-time communication channels using WebSocket protocol.

This module handles:
- Live market data streaming
- Real-time trading signals
- System notifications
- User session management
- Connection pooling and scaling

The WebSocket implementation ensures:
- Efficient bi-directional communication
- Automatic reconnection handling
- Message queuing and delivery guarantees
- Connection state management
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

class WSMessageType(Enum):
    """Types of WebSocket messages"""
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    NOTIFICATION = "notification"
    SYSTEM_EVENT = "system_event"

@dataclass
class WSConnection:
    """
    Represents a WebSocket connection with its metadata.
    این کلاس برای مدیریت اطلاعات هر اتصال استفاده میشه.
    """
    id: str
    user_id: str
    connected_at: datetime
    subscriptions: List[str]
    is_active: bool = True
    last_ping: Optional[datetime] = None