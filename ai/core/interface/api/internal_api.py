from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import logging

from ...learning.core_learner import CoreLearner
from ...knowledge.knowledge_manager import KnowledgeManager
from ...memory.memory_manager import MemoryManager
from ...errors.exceptions import AIError

logger = logging.getLogger(__name__)


class QueryInput(BaseModel):
    text: str
    language: str = "fa"
    context: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    memory_usage: float
    knowledge_count: int
    active_learners: int
    last_update: datetime


class InternalAPI:
    def __init__(self, config: Dict[str, Any]):
        self.app = FastAPI(title="AI Internal API")
        self.config = config
        self.learner = CoreLearner(config)
        self.knowledge_manager = KnowledgeManager()
        self.memory_manager = MemoryManager(config.get('memory', {}))

        self._setup_middleware()
        self._setup_routes()
        self._background_tasks = set()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.on_event("startup")
        async def startup():
            await self.learner.initialize()
            await self.knowledge_manager.initialize()

        @self.app.on_event("shutdown")
        async def shutdown():
            for task in self._background_tasks:
                task.cancel()

        @self.app.post("/learn")
        async def learn(query: QueryInput, background_tasks: BackgroundTasks):
            try:
                context = query.context or {"language": query.language}
                result = await self.learner.learn(query.text, context)

                # ثبت نتایج در پس‌زمینه
                task = asyncio.create_task(self._record_learning(result))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

                return {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now()
                }
            except AIError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Learning error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/query")
        async def query(query: QueryInput):
            try:
                result = await self.knowledge_manager.get_knowledge(
                    query.text,
                    context=query.context
                )
                return {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now()
                }
            except AIError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Query error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.get("/status")
        async def get_status() -> SystemStatus:
            try:
                memory_stats = await self.memory_manager.get_stats()
                knowledge_stats = await self.knowledge_manager.get_stats()

                return SystemStatus(
                    memory_usage=memory_stats.get("usage", 0),
                    knowledge_count=knowledge_stats.get("count", 0),
                    active_learners=len(self._background_tasks),
                    last_update=datetime.now()
                )
            except Exception as e:
                logger.error(f"Status error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/reset")
        async def reset_system():
            try:
                await self.learner.reset()
                await self.knowledge_manager.reset()
                await self.memory_manager.cleanup()
                return {"status": "success", "message": "System reset completed"}
            except Exception as e:
                logger.error(f"Reset error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

    async def _record_learning(self, result: Dict[str, Any]):
        """ثبت نتایج یادگیری در پس‌زمینه"""
        try:
            await self.memory_manager.store({
                "type": "learning_result",
                "content": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error recording learning result: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = InternalAPI({})
    api.run()