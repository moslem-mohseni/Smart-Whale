from typing import Dict, Any, List
from collections import defaultdict
import time
from ..knowledge_sharing.privacy_guard import PrivacyGuard

class AccessControl:
    """
    ماژول مدیریت دسترسی و امنیت در اشتراک‌گذاری دانش بین مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه سطح دسترسی مدل‌ها و سیستم لاگ‌گیری.
        """
        self.access_policies: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"read": [], "write": []})
        self.access_logs: List[Dict[str, Any]] = []
        self.privacy_guard = PrivacyGuard()

    def grant_access(self, model_id: str, permission_type: str, resource: str):
        """
        اعطای دسترسی به یک مدل برای یک منبع خاص.
        :param model_id: شناسه مدل.
        :param permission_type: نوع مجوز ('read' یا 'write').
        :param resource: منبع موردنظر برای دسترسی.
        """
        if permission_type not in ["read", "write"]:
            raise ValueError("نوع مجوز باید 'read' یا 'write' باشد.")

        if resource not in self.access_policies[model_id][permission_type]:
            self.access_policies[model_id][permission_type].append(resource)

        self._log_access_event(model_id, f"Granted {permission_type} access to {resource}")

    def revoke_access(self, model_id: str, permission_type: str, resource: str):
        """
        لغو دسترسی یک مدل از یک منبع خاص.
        :param model_id: شناسه مدل.
        :param permission_type: نوع مجوز ('read' یا 'write').
        :param resource: منبع موردنظر برای لغو دسترسی.
        """
        if permission_type not in ["read", "write"]:
            raise ValueError("نوع مجوز باید 'read' یا 'write' باشد.")

        if resource in self.access_policies[model_id][permission_type]:
            self.access_policies[model_id][permission_type].remove(resource)

        self._log_access_event(model_id, f"Revoked {permission_type} access to {resource}")

    def check_access(self, model_id: str, permission_type: str, resource: str) -> bool:
        """
        بررسی اینکه آیا یک مدل مجاز به دسترسی به یک منبع خاص است یا نه.
        :param model_id: شناسه مدل.
        :param permission_type: نوع مجوز ('read' یا 'write').
        :param resource: منبع موردنظر برای بررسی.
        :return: `True` اگر مدل دسترسی داشته باشد، `False` در غیر اینصورت.
        """
        return resource in self.access_policies[model_id][permission_type]

    def _log_access_event(self, model_id: str, event: str):
        """
        ثبت یک رویداد دسترسی برای ردیابی فعالیت مدل‌ها.
        :param model_id: شناسه مدل.
        :param event: متن رویداد.
        """
        timestamp = int(time.time())
        self.access_logs.append({
            "timestamp": timestamp,
            "model_id": model_id,
            "event": event
        })

    def get_access_logs(self, model_id: str) -> List[Dict[str, Any]]:
        """
        دریافت تمام لاگ‌های دسترسی مربوط به یک مدل خاص.
        :param model_id: شناسه مدل.
        :return: لیست رویدادهای دسترسی مدل.
        """
        return [log for log in self.access_logs if log["model_id"] == model_id]
