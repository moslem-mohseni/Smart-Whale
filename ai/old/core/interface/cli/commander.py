import click
import asyncio
from typing import Dict, Any
import json
from pathlib import Path
import sys
import logging

from ...learning.core_learner import CoreLearner
from ...knowledge.knowledge_manager import KnowledgeManager
from ...memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class CLI:
    def __init__(self):
        self.learner = None
        self.knowledge_manager = None
        self.memory_manager = None
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path("config.json")
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    async def initialize(self):
        self.learner = CoreLearner(self.config)
        self.knowledge_manager = KnowledgeManager()
        self.memory_manager = MemoryManager(self.config.get('memory', {}))

        await self.learner.initialize()
        await self.knowledge_manager.initialize()


@click.group()
def cli():
    """سیستم تست هوش مصنوعی"""
    pass


@cli.command()
@click.argument('query')
@click.option('--lang', default='fa', help='زبان ورودی')
def learn(query: str, lang: str):
    """یادگیری از ورودی جدید"""
    cli_instance = CLI()
    asyncio.run(cli_instance.handle_learn(query, lang))


@cli.command()
@click.argument('query')
def ask(query: str):
    """پرسش از سیستم"""
    cli_instance = CLI()
    asyncio.run(cli_instance.handle_ask(query))


@cli.command()
def status():
    """نمایش وضعیت سیستم"""
    cli_instance = CLI()
    asyncio.run(cli_instance.handle_status())


class CLI:
    async def handle_learn(self, query: str, lang: str):
        await self.initialize()
        try:
            context = {'language': lang}
            result = await self.learner.learn(query, context)
            click.echo(f"نتیجه یادگیری: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            click.echo(f"خطا در یادگیری: {e}")

    async def handle_ask(self, query: str):
        await self.initialize()
        try:
            result = await self.knowledge_manager.get_knowledge(query)
            click.echo(f"پاسخ: {json.dumps(result, indent=2, ensure_ascii=False)}")
        except Exception as e:
            click.echo(f"خطا در پاسخ‌دهی: {e}")

    async def handle_status(self):
        await self.initialize()
        try:
            memory_stats = await self.memory_manager.get_stats()
            knowledge_stats = await self.knowledge_manager.get_stats()

            click.echo("وضعیت سیستم:")
            click.echo(f"حافظه: {json.dumps(memory_stats, indent=2, ensure_ascii=False)}")
            click.echo(f"دانش: {json.dumps(knowledge_stats, indent=2, ensure_ascii=False)}")
        except Exception as e:
            click.echo(f"خطا در دریافت وضعیت: {e}")


if __name__ == '__main__':
    cli()