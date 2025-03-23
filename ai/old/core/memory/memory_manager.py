from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import asyncio
import json
import logging
from pathlib import Path
import aiofiles
import networkx as nx

logger = logging.getLogger(__name__)


class MemoryNode:
    def __init__(self, content: Dict[str, Any], memory_type: str):
        self.content = content
        self.type = memory_type
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.importance = self._calculate_importance()
        self.connections: Set[str] = set()

    def _calculate_importance(self) -> float:
        # محاسبه اهمیت بر اساس فاکتورهای مختلف
        base_importance = 0.5
        time_factor = 0.3
        access_factor = 0.2

        # اهمیت بر اساس زمان (کاهش با گذر زمان)
        age = (datetime.now() - self.created_at).total_seconds()
        time_score = max(0, 1 - (age / (7 * 24 * 3600)))  # کاهش در طول یک هفته

        # اهمیت بر اساس تعداد دسترسی
        access_score = min(1, self.access_count / 100)

        return (base_importance +
                (time_factor * time_score) +
                (access_factor * access_score))


class MemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'memory'))
        self.short_term_capacity = config.get('short_term_capacity', 1000)
        self.long_term_threshold = config.get('long_term_threshold', 0.7)

        self.short_term: Dict[str, MemoryNode] = {}
        self.long_term: Dict[str, MemoryNode] = {}
        self.memory_graph = nx.DiGraph()

    async def store(self, content: Dict[str, Any], memory_type: str) -> str:
        """ذخیره محتوا در حافظه"""
        node = MemoryNode(content, memory_type)
        memory_id = self._generate_id(content)

        if len(self.short_term) >= self.short_term_capacity:
            await self._consolidate_memory()

        self.short_term[memory_id] = node
        self.memory_graph.add_node(memory_id, **node.__dict__)

        await self._connect_related_memories(memory_id, node)
        await self._persist_memory(memory_id, node)

        return memory_id

    async def retrieve(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """بازیابی محتوا از حافظه"""
        node = self.short_term.get(memory_id) or self.long_term.get(memory_id)

        if node:
            node.last_accessed = datetime.now()
            node.access_count += 1
            node.importance = node._calculate_importance()
            return node.content

        return await self._retrieve_from_storage(memory_id)

    async def find_related(self, content: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """یافتن محتوای مرتبط"""
        temp_id = self._generate_id(content)
        temp_node = MemoryNode(content, "temporary")

        related_items = []
        for memory_id, node in {**self.short_term, **self.long_term}.items():
            similarity = self._calculate_similarity(temp_node, node)
            if similarity > 0.5:
                related_items.append((memory_id, node, similarity))

        related_items.sort(key=lambda x: x[2], reverse=True)
        return [self._prepare_memory_data(item[1]) for item in related_items[:limit]]

    async def update_connections(self, source_id: str, target_id: str,
                                 connection_type: str) -> None:
        """به‌روزرسانی ارتباطات بین محتواها"""
        if source_id in self.memory_graph and target_id in self.memory_graph:
            self.memory_graph.add_edge(
                source_id,
                target_id,
                type=connection_type,
                timestamp=datetime.now()
            )

            source_node = self.short_term.get(source_id) or self.long_term.get(source_id)
            target_node = self.short_term.get(target_id) or self.long_term.get(target_id)

            if source_node and target_node:
                source_node.connections.add(target_id)
                target_node.connections.add(source_id)

    async def _consolidate_memory(self) -> None:
        """انتقال حافظه کوتاه‌مدت به بلندمدت"""
        for memory_id, node in list(self.short_term.items()):
            if node.importance >= self.long_term_threshold:
                self.long_term[memory_id] = node
                await self._persist_memory(memory_id, node, long_term=True)
            del self.short_term[memory_id]

    def _generate_id(self, content: Dict[str, Any]) -> str:
        """تولید شناسه یکتا برای محتوا"""
        import hashlib
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:12]

    async def _persist_memory(self, memory_id: str, node: MemoryNode,
                              long_term: bool = False) -> None:
        """ذخیره دائمی حافظه"""
        directory = self.storage_path / ('long_term' if long_term else 'short_term')
        directory.mkdir(parents=True, exist_ok=True)

        file_path = directory / f"{memory_id}.json"
        memory_data = self._prepare_memory_data(node)

        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(memory_data, ensure_ascii=False, indent=2))

    async def _retrieve_from_storage(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """بازیابی حافظه از storage"""
        for directory in ['short_term', 'long_term']:
            file_path = self.storage_path / directory / f"{memory_id}.json"
            if file_path.exists():
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return json.loads(content)
        return None

    async def _connect_related_memories(self, memory_id: str, node: MemoryNode) -> None:
        """ایجاد ارتباط بین حافظه‌های مرتبط"""
        for other_id, other_node in {**self.short_term, **self.long_term}.items():
            if other_id != memory_id:
                similarity = self._calculate_similarity(node, other_node)
                if similarity > 0.7:
                    await self.update_connections(
                        memory_id,
                        other_id,
                        "semantic_similarity"
                    )

    def _calculate_similarity(self, node1: MemoryNode, node2: MemoryNode) -> float:
        """محاسبه شباهت بین دو محتوا"""
        # اینجا می‌توان از روش‌های پیشرفته‌تر استفاده کرد
        common_keys = set(node1.content.keys()) & set(node2.content.keys())
        if not common_keys:
            return 0.0

        similarity = 0.0
        for key in common_keys:
            if node1.content[key] == node2.content[key]:
                similarity += 1.0

        return similarity / len(common_keys)

    def _prepare_memory_data(self, node: MemoryNode) -> Dict[str, Any]:
        """آماده‌سازی داده‌های حافظه برای ذخیره"""
        return {
            'content': node.content,
            'type': node.type,
            'created_at': node.created_at.isoformat(),
            'last_accessed': node.last_accessed.isoformat(),
            'access_count': node.access_count,
            'importance': node.importance,
            'connections': list(node.connections)
        }

    async def cleanup(self) -> None:
        """پاکسازی حافظه‌های قدیمی"""
        cutoff_date = datetime.now() - timedelta(days=30)

        # پاکسازی حافظه کوتاه‌مدت
        for memory_id, node in list(self.short_term.items()):
            if node.last_accessed < cutoff_date and node.importance < 0.3:
                del self.short_term[memory_id]
                self.memory_graph.remove_node(memory_id)

        # پاکسازی حافظه بلندمدت
        for memory_id, node in list(self.long_term.items()):
            if node.last_accessed < cutoff_date and node.importance < 0.1:
                del self.long_term[memory_id]
                self.memory_graph.remove_node(memory_id)

        # پاکسازی فایل‌ها
        await self._cleanup_storage()