from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import asyncio
from dataclasses import dataclass
import logging
from enum import Enum

from ..memory.memory_manager import MemoryManager
from ..metrics.metrics_collector import MetricsCollector
from ..knowledge.knowledge_manager import KnowledgeManager

logger = logging.getLogger(__name__)


class DecisionConfidence(Enum):
    HIGH = "high"  # اطمینان بالا (> 0.8)
    MEDIUM = "medium"  # اطمینان متوسط (0.5-0.8)
    LOW = "low"  # اطمینان پایین (< 0.5)
    UNCERTAIN = "uncertain"  # عدم اطمینان


@dataclass
class DecisionContext:
    query: str
    constraints: Dict[str, Any]
    priority: int
    deadline: Optional[datetime]
    required_confidence: float
    max_attempts: int


@dataclass
class Decision:
    id: str
    content: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    sources: List[str]
    timestamp: datetime
    context: DecisionContext
    metadata: Dict[str, Any]


class DecisionMaker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_manager = MemoryManager(config.get('memory', {}))
        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        self.knowledge_manager = KnowledgeManager()

        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_retries = config.get('max_retries', 3)
        self.decision_weights = self._initialize_weights()
        self.learning_rate = 0.01

    def _initialize_weights(self) -> Dict[str, float]:
        return {
            'knowledge_confidence': 0.3,
            'memory_relevance': 0.2,
            'context_alignment': 0.2,
            'time_pressure': 0.1,
            'resource_availability': 0.1,
            'past_performance': 0.1
        }

    async def make_decision(self, context: DecisionContext) -> Decision:
        """اتخاذ تصمیم با در نظر گرفتن تمام فاکتورها"""
        start_time = datetime.now()
        decision = None
        attempts = 0

        while not decision and attempts < context.max_attempts:
            try:
                # جمع‌آوری اطلاعات از منابع مختلف
                knowledge_results = await self._gather_knowledge(context)
                memory_results = await self._check_memory(context)
                metrics_data = await self._get_metrics()

                # تولید گزینه‌های تصمیم
                options = await self._generate_options(
                    knowledge_results,
                    memory_results,
                    context
                )

                # ارزیابی و انتخاب بهترین گزینه
                best_option = await self._evaluate_options(options, context, metrics_data)

                if best_option:
                    decision = self._create_decision(best_option, context)
                    if decision.confidence >= context.required_confidence:
                        break

                attempts += 1
                if not decision and attempts < context.max_attempts:
                    await self._adjust_strategy(attempts, context)

            except Exception as e:
                logger.error(f"Error in decision making: {e}")
                attempts += 1

        if not decision:
            decision = await self._make_fallback_decision(context)

        # ثبت تصمیم و به‌روزرسانی سیستم
        await self._record_decision(decision)
        await self._update_learning(decision, start_time)

        return decision

    async def _gather_knowledge(self, context: DecisionContext) -> List[Dict[str, Any]]:
        """جمع‌آوری دانش مرتبط"""
        try:
            results = await self.knowledge_manager.learn(context.query)
            filtered_results = []

            for result in results:
                confidence = self._calculate_knowledge_confidence(result)
                if confidence >= self.min_confidence:
                    result['confidence'] = confidence
                    filtered_results.append(result)

            return filtered_results

        except Exception as e:
            logger.error(f"Error gathering knowledge: {e}")
            return []

    async def _check_memory(self, context: DecisionContext) -> List[Dict[str, Any]]:
        """بررسی تجربیات گذشته"""
        try:
            related_memories = await self.memory_manager.find_related({
                'query': context.query,
                'constraints': context.constraints
            })

            scored_memories = []
            for memory in related_memories:
                relevance = self._calculate_memory_relevance(memory, context)
                if relevance >= self.min_confidence:
                    memory['relevance'] = relevance
                    scored_memories.append(memory)

            return scored_memories

        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return []

    async def _generate_options(self, knowledge_results: List[Dict[str, Any]],
                                memory_results: List[Dict[str, Any]],
                                context: DecisionContext) -> List[Dict[str, Any]]:
        """تولید گزینه‌های ممکن برای تصمیم‌گیری"""
        options = []

        # ترکیب دانش جدید با تجربیات گذشته
        for kr in knowledge_results:
            base_option = {
                'content': kr['content'],
                'source': 'knowledge',
                'confidence': kr['confidence'],
                'reasoning': []
            }
            options.append(base_option)

            # بررسی تطابق با تجربیات گذشته
            for mr in memory_results:
                if self._are_compatible(kr, mr):
                    enhanced_option = self._merge_knowledge_and_memory(kr, mr)
                    options.append(enhanced_option)

        # اعمال محدودیت‌ها و فیلتر گزینه‌ها
        filtered_options = []
        for option in options:
            if self._meets_constraints(option, context.constraints):
                confidence = self._calculate_option_confidence(option, context)
                option['confidence'] = confidence
                filtered_options.append(option)

        return filtered_options

    async def _evaluate_options(self, options: List[Dict[str, Any]],
                                context: DecisionContext,
                                metrics_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ارزیابی و رتبه‌بندی گزینه‌ها"""
        if not options:
            return None

        scored_options = []
        for option in options:
            scores = {
                'knowledge_confidence': option['confidence'],
                'memory_relevance': self._calculate_memory_relevance(option, context),
                'context_alignment': self._calculate_context_alignment(option, context),
                'time_pressure': self._calculate_time_pressure(context),
                'resource_availability': self._calculate_resource_score(metrics_data),
                'past_performance': self._calculate_past_performance(option)
            }

            total_score = sum(score * self.decision_weights[factor]
                              for factor, score in scores.items())

            scored_options.append((option, total_score))

        if not scored_options:
            return None

        return max(scored_options, key=lambda x: x[1])[0]

    def _calculate_option_confidence(self, option: Dict[str, Any],
                                     context: DecisionContext) -> float:
        """محاسبه میزان اطمینان به یک گزینه"""
        base_confidence = option.get('confidence', 0.5)

        factors = {
            'completeness': self._calculate_completeness(option),
            'consistency': self._calculate_consistency(option),
            'context_match': self._calculate_context_match(option, context),
            'source_reliability': self._calculate_source_reliability(option)
        }

        weights = {'completeness': 0.3, 'consistency': 0.3,
                   'context_match': 0.2, 'source_reliability': 0.2}

        confidence = base_confidence * sum(score * weights[factor]
                                           for factor, score in factors.items())

        return min(1.0, max(0.0, confidence))

    def _calculate_completeness(self, option: Dict[str, Any]) -> float:
        """محاسبه کامل بودن گزینه"""
        required_fields = ['content', 'reasoning', 'sources']
        present_fields = sum(1 for field in required_fields if field in option)
        return present_fields / len(required_fields)

    def _calculate_consistency(self, option: Dict[str, Any]) -> float:
        """محاسبه سازگاری درونی گزینه"""
        try:
            content = option.get('content', {})
            reasoning = option.get('reasoning', [])

            if not content or not reasoning:
                return 0.0

            # بررسی سازگاری منطقی بین محتوا و استدلال
            consistency_score = 0.0
            for reason in reasoning:
                if any(key in str(reason).lower()
                       for key in str(content).lower().split()):
                    consistency_score += 1.0

            return min(1.0, consistency_score / len(reasoning))

        except Exception:
            return 0.0

    def _calculate_context_match(self, option: Dict[str, Any],
                                 context: DecisionContext) -> float:
        """محاسبه میزان تطابق با context"""
        matches = 0
        total_constraints = len(context.constraints)

        if total_constraints == 0:
            return 1.0

        for key, value in context.constraints.items():
            if key in option.get('content', {}) and option['content'][key] == value:
                matches += 1

        return matches / total_constraints

    def _calculate_source_reliability(self, option: Dict[str, Any]) -> float:
        """محاسبه قابلیت اعتماد منابع"""
        sources = option.get('sources', [])
        if not sources:
            return 0.0

        reliability_scores = {
            'knowledge_base': 0.9,
            'memory': 0.8,
            'inference': 0.7,
            'external': 0.6
        }

        total_score = 0.0
        for source in sources:
            for source_type, score in reliability_scores.items():
                if source_type in str(source).lower():
                    total_score += score
                    break

        return min(1.0, total_score / len(sources))

    async def _make_fallback_decision(self, context: DecisionContext) -> Decision:
        """تصمیم‌گیری پشتیبان در صورت شکست روش‌های اصلی"""
        logger.warning("Using fallback decision making")

        # استفاده از ساده‌ترین گزینه ممکن که محدودیت‌ها را رعایت می‌کند
        basic_content = {
            'action': 'fallback',
            'reason': 'No suitable options found with required confidence'
        }

        return Decision(
            id=self._generate_decision_id(),
            content=basic_content,
            confidence=self.min_confidence,
            reasoning=['Fallback decision due to insufficient confidence in other options'],
            sources=['fallback_system'],
            timestamp=datetime.now(),
            context=context,
            metadata={'fallback': True}
        )

    async def _record_decision(self, decision: Decision) -> None:
        """ثبت تصمیم برای یادگیری آینده"""
        try:
            await self.memory_manager.store({
                'decision': decision.content,
                'context': {
                    'query': decision.context.query,
                    'constraints': decision.context.constraints,
                    'timestamp': decision.timestamp.isoformat()
                },
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'sources': decision.sources
            }, 'decision')

        except Exception as e:
            logger.error(f"Error recording decision: {e}")

    async def _update_learning(self, decision: Decision, start_time: datetime) -> None:
        """به‌روزرسانی پارامترهای یادگیری"""
        try:
            execution_time = (datetime.now() - start_time).total_seconds()

            # به‌روزرسانی وزن‌ها بر اساس موفقیت تصمیم
            if decision.confidence >= decision.context.required_confidence:
                self._adjust_weights_positively(decision)
            else:
                self._adjust_weights_negatively(decision)

            # ثبت متریک‌ها
            await self.metrics_collector.record_decision_metrics({
                'decision_id': decision.id,
                'confidence': decision.confidence,
                'execution_time': execution_time,
                'success': decision.confidence >= decision.context.required_confidence
            })

        except Exception as e:
            logger.error(f"Error updating learning parameters: {e}")

    def _adjust_weights_positively(self, decision: Decision) -> None:
        """تنظیم مثبت وزن‌ها بر اساس تصمیم موفق"""
        for factor in self.decision_weights:
            if factor in decision.metadata:
                self.decision_weights[factor] += self.learning_rate

        # نرمال‌سازی وزن‌ها
        total = sum(self.decision_weights.values())
        self.decision_weights = {k: v / total for k, v in self.decision_weights.items()}

    def _adjust_weights_negatively(self, decision: Decision) -> None:
        """تنظیم منفی وزن‌ها بر اساس تصمیم ناموفق"""
        for factor in self.decision_weights:
            if factor in decision.metadata:
                self.decision_weights[factor] -= self.learning_rate
                self.decision_weights[factor] = max(0.1, self.decision_weights[factor])

        # نرمال‌سازی وزن‌ها
        total = sum(self.decision_weights.values())
        self.decision_weights = {k: v / total for k, v in self.decision_weights.items()}