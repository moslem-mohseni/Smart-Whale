# persian/language_processors/domain/domain_analysis.py

"""
ماژول domain_analysis.py

این ماژول توابع اصلی تحلیل دانش حوزه‌ای را پیاده‌سازی می‌کند. امکانات اصلی:
  - یافتن حوزه‌های مرتبط با متن (find_domain_for_text)
  - کشف خودکار حوزه‌ها و مفاهیم جدید (discover_new_domains, discover_new_concepts, discover_relations)
  - تولید سلسله‌مراتب حوزه‌ها (get_domain_hierarchy)
  - پاسخ به پرسش‌های حوزه‌ای (answer_domain_question)
  - دریافت بردار معنایی مفاهیم (get_concept_vector) و یافتن مفاهیم مشابه (find_similar_concepts)
  - استفاده از مدل‌های هوشمند (smart_model) و مدل معلم (teacher_model) در صورت وجود
  - fallback به روش‌های rule-based در صورت عدم موفقیت مدل‌های هوشمند
"""

import asyncio
import time
import logging
import hashlib
import random
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

# فرض بر این است که این فایل‌ها از قبل وجود دارند
from domain_data import DomainDataAccess
from domain_services import DomainServices
from domain_models import Domain, Concept, Relation, Attribute

# اگر فایل domain_config.py دارید، می‌توانید از آنجا مقادیر را دریافت کنید
# در اینجا برای نمونه به‌صورت ثابت استفاده می‌کنیم
SIMILARITY_THRESHOLD = 0.7
TOP_K = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DomainAnalysis:
    """
    کلاس DomainAnalysis مسئول پیاده‌سازی توابع تحلیل دانش حوزه‌ای است.
    از DomainDataAccess برای خواندن داده‌ها و از DomainServices برای افزودن داده‌های جدید بهره می‌گیرد.
    در صورت وجود مدل هوشمند (smart_model) و مدل معلم (teacher_model)، از آن‌ها نیز استفاده می‌کند.
    """

    def __init__(self,
                 data_access: Optional[DomainDataAccess] = None,
                 domain_services: Optional[DomainServices] = None,
                 smart_model: Optional[Any] = None,
                 teacher_model: Optional[Any] = None,
                 vector_store: Optional[Any] = None):
        """
        سازنده DomainAnalysis

        Args:
            data_access (Optional[DomainDataAccess]): شیء دسترسی به داده‌های حوزه
            domain_services (Optional[DomainServices]): لایه خدمات حوزه
            smart_model (اختیاری): مدل هوشمند (در صورت وجود)
            teacher_model (اختیاری): مدل معلم (در صورت وجود)
            vector_store (اختیاری): شیء جستجوی برداری برای شباهت مفاهیم (در صورت وجود)
        """
        self.data_access = data_access if data_access else DomainDataAccess()
        self.domain_services = domain_services if domain_services else DomainServices(self.data_access)
        self.smart_model = smart_model
        self.teacher_model = teacher_model
        self.vector_store = vector_store  # برای جستجوی nearest neighbor یا شبیه آن

    # -------------------------------------------------------------------------
    # 1) یافتن حوزه‌های مرتبط با متن
    # -------------------------------------------------------------------------
    async def find_domain_for_text(self, text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        یافتن حوزه‌های احتمالی برای متن ورودی.

        1) تلاش با smart_model (اگر موجود و اعتماد کافی)
        2) در صورت عدم موفقیت، تلاش با teacher_model
        3) در نهایت، fallback به rule-based

        Returns:
            لیستی از دیکشنری حوزه‌ها با ساختار:
            [
              {
                "domain_code": ...,
                "score": ...,
                "source": "smart_model" | "teacher" | "rule_based"
              },
              ...
            ]
        """
        text_lower = text.lower()
        source = "rule_based"
        results: List[Dict[str, Any]] = []

        # 1) تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "find_domains_for_text"):
            confidence = 0.7
            if hasattr(self.smart_model, "confidence_level"):
                confidence = self.smart_model.confidence_level(text)

            if confidence >= SIMILARITY_THRESHOLD:
                try:
                    domains = self.smart_model.find_domains_for_text(text, top_k)
                    if domains and isinstance(domains, list) and len(domains) > 0:
                        source = "smart_model"
                        results = domains
                        return self._format_domain_results(results, source)
                except Exception as e:
                    logger.error(f"خطا در find_domain_for_text با smart_model: {e}")

        # 2) تلاش با teacher_model
        if self.teacher_model and hasattr(self.teacher_model, "find_domains_for_text"):
            try:
                domains = self.teacher_model.find_domains_for_text(text, top_k)
                if domains and isinstance(domains, list) and len(domains) > 0:
                    source = "teacher"
                    # یادگیری از معلم
                    await self._learn_from_teacher(text, {"domains": domains})
                    results = domains
                    return self._format_domain_results(results, source)
            except Exception as e:
                logger.error(f"خطا در find_domain_for_text با teacher_model: {e}")

        # 3) fallback: rule-based
        results = await self._rule_based_domain_detection(text_lower, top_k)
        return self._format_domain_results(results, source)

    async def _rule_based_domain_detection(self, text_lower: str, top_k: int) -> List[Dict[str, Any]]:
        """
        تشخیص حوزه به روش ساده (rule-based):
        - بررسی تطابق کلمات کلیدی در نام مفاهیم
        - بررسی کلیدواژه‌های عمومی هر حوزه
        """
        # ابتدا همه حوزه‌ها و مفاهیم را می‌خوانیم
        all_domains = await self.data_access.load_domains()
        all_concepts = await self.data_access.load_concepts()

        # امتیازدهی بر اساس مفاهیم
        domain_scores = {}
        for concept_id, concept in all_concepts.items():
            domain_id = concept["domain_id"]
            concept_name = concept["concept_name"].lower()
            if concept_name in text_lower:
                score = text_lower.count(concept_name) * (concept.get("confidence", 0.8))
                domain_scores[domain_id] = domain_scores.get(domain_id, 0) + score

        # اگر چیزی پیدا نشد، یک fallback دیگر
        if not domain_scores:
            # نمونه ساده: بررسی کلمات کلیدی پیش‌فرض هر حوزه
            # (در عمل می‌توانید لیست کلمات کلیدی را در domain_config قرار دهید)
            domain_keywords = {
                "MEDICAL": ["بیمار", "درمان", "پزشک", "دارو", "بیماری", "جراحی"],
                "ENGINEERING": ["مهندس", "طراحی", "ساخت", "مکانیک", "عمران", "برق"],
                "IT": ["نرم‌افزار", "کدنویسی", "الگوریتم", "داده", "شبکه", "سرور"],
            }
            for code, keywords in domain_keywords.items():
                # پیدا کردن domain_id
                d_id = None
                for dd_id, dd_val in all_domains.items():
                    if dd_val["domain_code"] == code:
                        d_id = dd_id
                        break
                if not d_id:
                    continue

                score = 0
                for kw in keywords:
                    c = text_lower.count(kw)
                    score += c * 0.5
                if score > 0:
                    domain_scores[d_id] = domain_scores.get(d_id, 0) + score

        # ساخت لیست نتایج
        results: List[Dict[str, Any]] = []
        for d_id, sc in domain_scores.items():
            dom_info = all_domains[d_id]
            results.append({
                "domain_code": dom_info["domain_code"],
                "domain_name": dom_info["domain_name"],
                "score": sc
            })

        # مرتب‌سازی
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _format_domain_results(self, domains: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        فرمت کردن خروجی حوزه‌ها
        """
        for d in domains:
            d["source"] = source
        return domains

    # -------------------------------------------------------------------------
    # 2) کشف خودکار حوزه‌ها
    # -------------------------------------------------------------------------
    async def discover_new_domains(self, text: str) -> List[Dict[str, Any]]:
        """
        کشف خودکار حوزه‌های جدید از متن ورودی:
        1) تلاش با smart_model (اگر موجود)
        2) تلاش با teacher_model
        3) (اختیاری) rule-based
        4) افزودن حوزه‌های جدید از طریق domain_services
        """
        discovered = []

        # 1) تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "discover_domains"):
            try:
                domains = self.smart_model.discover_domains(text)
                if domains and isinstance(domains, list):
                    discovered = await self._add_new_domains(domains)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_new_domains با smart_model: {e}")

        # 2) تلاش با teacher_model
        if self.teacher_model and hasattr(self.teacher_model, "discover_domains"):
            try:
                domains = self.teacher_model.discover_domains(text)
                if domains and isinstance(domains, list):
                    # یادگیری از معلم
                    await self._learn_from_teacher(text, {"discovered_domains": domains})
                    discovered = await self._add_new_domains(domains)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_new_domains با teacher_model: {e}")

        # 3) fallback rule-based
        # اینجا صرفاً یک نمونه ساده‌ی ساختگی است
        # در عمل می‌توانید بر اساس کلیدواژه یا الگوهای خاص حوزه‌های جدید را تشخیص دهید
        new_domain_code = "FAKE_NEW_DOMAIN"
        if new_domain_code not in [d["domain_code"] for d in discovered]:
            added = await self.domain_services.add_domain("حوزه ساختگی", new_domain_code, "حوزه جدید کشف‌شده به صورت تستی")
            if added:
                discovered.append({"domain_code": new_domain_code, "domain_name": "حوزه ساختگی"})
        return discovered

    async def _add_new_domains(self, domains_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        افزودن حوزه‌های جدید کشف‌شده به سیستم
        """
        added_list = []
        for d_info in domains_info:
            domain_code = d_info.get("domain_code")
            domain_name = d_info.get("domain_name", "")
            parent_code = d_info.get("parent_domain_code", "")
            if domain_code:
                new_dom = await self.domain_services.add_domain(domain_name, domain_code, "", parent_code)
                if new_dom:
                    added_list.append({
                        "domain_code": domain_code,
                        "domain_name": domain_name
                    })
        return added_list

    # -------------------------------------------------------------------------
    # 3) کشف خودکار مفاهیم
    # -------------------------------------------------------------------------
    async def discover_new_concepts(self, text: str, domain_code: str) -> List[Dict[str, Any]]:
        """
        کشف خودکار مفاهیم جدید در یک حوزه مشخص
        1) تلاش با smart_model
        2) تلاش با teacher_model
        3) rule-based
        """
        discovered = []

        # 1) تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "discover_concepts"):
            try:
                concepts = self.smart_model.discover_concepts(text, domain_code)
                if concepts and isinstance(concepts, list):
                    discovered = await self._add_new_concepts(concepts, domain_code)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_new_concepts با smart_model: {e}")

        # 2) تلاش با teacher_model
        if self.teacher_model and hasattr(self.teacher_model, "discover_concepts"):
            try:
                concepts = self.teacher_model.discover_concepts(text, domain_code)
                if concepts and isinstance(concepts, list):
                    await self._learn_from_teacher(text, {"discovered_concepts": concepts})
                    discovered = await self._add_new_concepts(concepts, domain_code)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_new_concepts با teacher_model: {e}")

        # 3) fallback rule-based (نمونه ساده)
        # مثلاً هر کلمه‌ای که طولش > 5 باشد را یک concept جدید در نظر بگیریم (خیلی ساده و بی‌منطق!)
        words = text.split()
        for w in words:
            if len(w) > 5:
                concept_name = w
                c_info = {"concept_name": concept_name, "definition": f"تعریف ساده برای {concept_name}"}
                added = await self._add_new_concepts([c_info], domain_code)
                discovered.extend(added)
        return discovered

    async def _add_new_concepts(self, concepts_info: List[Dict[str, Any]], domain_code: str) -> List[Dict[str, Any]]:
        """
        افزودن مفاهیم جدید به یک حوزه
        """
        added_list = []
        for c_info in concepts_info:
            concept_name = c_info.get("concept_name", "")
            definition = c_info.get("definition", "")
            examples = c_info.get("examples", [])
            if concept_name:
                new_c = await self.domain_services.add_concept(domain_code, concept_name, definition, examples)
                if new_c:
                    added_list.append({
                        "concept_name": concept_name,
                        "definition": definition
                    })
        return added_list

    # -------------------------------------------------------------------------
    # 4) کشف خودکار روابط
    # -------------------------------------------------------------------------
    async def discover_relations(self, domain_code: str) -> List[Dict[str, Any]]:
        """
        کشف خودکار روابط بین مفاهیم یک حوزه
        1) تلاش با smart_model
        2) تلاش با teacher_model
        3) rule-based (مثلاً شباهت برداری یا فاصله Levenshtein)
        """
        discovered = []

        # یافتن حوزه
        domain = await self.data_access.find_domain_by_code(domain_code)
        if not domain:
            logger.error(f"حوزه با کد {domain_code} یافت نشد.")
            return discovered

        domain_id = domain["domain_id"]
        domain_concepts = await self.data_access.find_domain_concepts(domain_id)

        # 1) تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "discover_relations"):
            try:
                relations = self.smart_model.discover_relations(domain_concepts)
                if relations and isinstance(relations, list):
                    discovered = await self._add_new_relations(relations)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_relations با smart_model: {e}")

        # 2) تلاش با teacher_model
        if self.teacher_model and hasattr(self.teacher_model, "discover_relations"):
            try:
                relations = self.teacher_model.discover_relations(domain_concepts)
                if relations and isinstance(relations, list):
                    await self._learn_from_teacher(domain_code, {"discovered_relations": relations})
                    discovered = await self._add_new_relations(relations)
                    if discovered:
                        return discovered
            except Exception as e:
                logger.error(f"خطا در discover_relations با teacher_model: {e}")

        # 3) fallback rule-based
        # مثلاً شباهت معنایی (cosine similarity) بین بردارهای مفاهیم
        discovered = await self._discover_relations_rule_based(domain_concepts)
        return discovered

    async def _discover_relations_rule_based(self, domain_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        نمونه ساده: اگر شباهت بین دو مفهوم > 0.7 بود، رابطه SIMILAR_TO ایجاد کنیم.
        """
        discovered = []
        for i in range(len(domain_concepts)):
            for j in range(i + 1, len(domain_concepts)):
                c1 = domain_concepts[i]
                c2 = domain_concepts[j]
                sim = await self._calculate_concept_similarity(c1["concept_id"], c2["concept_id"])
                if sim > 0.7:
                    # ایجاد رابطه
                    rel = await self.domain_services.add_relation(
                        source_concept_id=c1["concept_id"],
                        target_concept_id=c2["concept_id"],
                        relation_type="SIMILAR_TO",
                        description=f"شباهت معنایی (rule-based) با امتیاز {sim:.2f}",
                        confidence=sim
                    )
                    if rel:
                        discovered.append({
                            "relation_id": rel.relation_id,
                            "source_concept_id": rel.source_concept_id,
                            "target_concept_id": rel.target_concept_id,
                            "relation_type": rel.relation_type,
                            "confidence": rel.confidence
                        })
        return discovered

    async def _add_new_relations(self, relations_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        افزودن روابط جدید کشف‌شده به سیستم
        """
        added_list = []
        for r_info in relations_info:
            source_id = r_info.get("source_concept_id")
            target_id = r_info.get("target_concept_id")
            rel_type = r_info.get("relation_type", "RELATED_TO")
            desc = r_info.get("description", "")
            conf = r_info.get("confidence", 0.8)
            new_r = await self.domain_services.add_relation(source_id, target_id, rel_type, desc, conf)
            if new_r:
                added_list.append({
                    "relation_id": new_r.relation_id,
                    "relation_type": new_r.relation_type,
                    "confidence": new_r.confidence
                })
        return added_list

    # -------------------------------------------------------------------------
    # 5) تولید سلسله‌مراتب حوزه
    # -------------------------------------------------------------------------
    async def get_domain_hierarchy(self, domain_code: str) -> Dict[str, Any]:
        """
        تولید ساختار سلسله‌مراتب مفاهیم یک حوزه بر اساس روابط IS_A یا PART_OF
        """
        domain = await self.data_access.find_domain_by_code(domain_code)
        if not domain:
            logger.error(f"حوزه با کد {domain_code} یافت نشد.")
            return {}

        domain_id = domain["domain_id"]
        all_relations = await self.data_access.load_relations()
        domain_concepts = await self.data_access.find_domain_concepts(domain_id)

        # تبدیل لیست به دیکشنری برای دسترسی سریع
        concept_dict = {c["concept_id"]: c for c in domain_concepts}

        # ساخت گراف
        graph = {}
        for r_id, rel in all_relations.items():
            if rel["relation_type"] in ["IS_A", "PART_OF"]:
                src = rel["source_concept_id"]
                tgt = rel["target_concept_id"]
                # هر دو مفهوم باید در این حوزه باشند
                if src in concept_dict and tgt in concept_dict:
                    if tgt not in graph:
                        graph[tgt] = []
                    graph[tgt].append(src)

        # یافتن ریشه‌ها (مفاهیمی که به‌عنوان فرزند در گراف نیامده‌اند)
        child_set = set()
        for children in graph.values():
            child_set.update(children)

        root_concepts = []
        for c_id in concept_dict:
            if c_id not in child_set:
                root_concepts.append(c_id)

        # ساختار بازگشتی
        def build_hierarchy(concept_id):
            if concept_id not in concept_dict:
                return None
            node = {
                "concept_id": concept_id,
                "concept_name": concept_dict[concept_id]["concept_name"],
                "children": []
            }
            if concept_id in graph:
                for child_id in graph[concept_id]:
                    child_node = build_hierarchy(child_id)
                    if child_node:
                        node["children"].append(child_node)
            return node

        hierarchy = []
        for root_id in root_concepts:
            h = build_hierarchy(root_id)
            if h:
                hierarchy.append(h)

        return {
            "domain_code": domain_code,
            "domain_name": domain["domain_name"],
            "hierarchy": hierarchy
        }

    # -------------------------------------------------------------------------
    # 6) پاسخ به پرسش‌های حوزه‌ای
    # -------------------------------------------------------------------------
    async def answer_domain_question(self, question: str, domain_code: Optional[str] = None) -> Dict[str, Any]:
        """
        پاسخ به پرسش‌های حوزه‌ای:
        1) اگر domain_code نامشخص باشد، ابتدا find_domain_for_text
        2) تلاش با smart_model
        3) تلاش با teacher_model
        4) fallback به پاسخ پیش‌فرض
        """
        if not domain_code:
            # تشخیص حوزه
            domains = await self.find_domain_for_text(question, top_k=1)
            if domains:
                domain_code = domains[0]["domain_code"]

        source = "fallback"
        answer_text = "متأسفانه پاسخ مشخصی برای این پرسش ندارم."
        confidence = 0.3

        # 1) تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "answer_domain_question"):
            try:
                ans = self.smart_model.answer_domain_question(question, domain_code)
                if ans and "answer" in ans:
                    source = "smart_model"
                    answer_text = ans["answer"]
                    confidence = ans.get("confidence", 0.8)
                    # (در صورت نیاز: اینجا ثبت usage یا منطق دیگر)
                    return {
                        "answer": answer_text,
                        "confidence": confidence,
                        "domain_code": domain_code,
                        "source": source
                    }
            except Exception as e:
                logger.error(f"خطا در answer_domain_question با smart_model: {e}")
                source = "teacher"

        # 2) تلاش با teacher_model
        if source == "teacher" or (self.teacher_model and hasattr(self.teacher_model, "answer_domain_question")):
            try:
                ans = self.teacher_model.answer_domain_question(question, domain_code)
                if ans and "answer" in ans:
                    source = "teacher"
                    answer_text = ans["answer"]
                    confidence = ans.get("confidence", 0.7)
                    # یادگیری از معلم
                    await self._learn_from_teacher(question, {"question_answer": ans, "domain": domain_code})
                    return {
                        "answer": answer_text,
                        "confidence": confidence,
                        "domain_code": domain_code,
                        "source": source
                    }
            except Exception as e:
                logger.error(f"خطا در answer_domain_question با teacher_model: {e}")
                source = "fallback"

        # 3) fallback
        return {
            "answer": answer_text,
            "confidence": confidence,
            "domain_code": domain_code,
            "source": source
        }

    # -------------------------------------------------------------------------
    # 7) دریافت بردار معنایی مفاهیم
    # -------------------------------------------------------------------------
    async def get_concept_vector(self, concept_id: str) -> List[float]:
        """
        دریافت بردار معنایی برای یک مفهوم:
          1) تلاش با smart_model
          2) تلاش با teacher_model
          3) fallback بردار تصادفی
        """
        # ابتدا بررسی کنیم مفهوم وجود دارد؟
        concept = await self._get_concept_by_id(concept_id)
        if not concept:
            logger.error(f"مفهوم با شناسه {concept_id} یافت نشد.")
            return []

        # اگر vector_store داریم، ابتدا بررسی کنیم شاید بردار قبلاً ذخیره شده باشد
        if self.vector_store and hasattr(self.vector_store, "get_vector"):
            existing_vec = await self.vector_store.get_vector("domain_vectors", concept_id)
            if existing_vec:
                return existing_vec

        # تلاش با smart_model
        if self.smart_model and hasattr(self.smart_model, "get_concept_vector"):
            try:
                text = f"{concept['concept_name']} - {concept.get('definition', '')}"
                vec = self.smart_model.get_concept_vector(text)
                if vec:
                    if isinstance(vec, list):
                        return vec
                    # اگر تنسور است، تبدیل به لیست
                    if hasattr(vec, "tolist"):
                        return vec.tolist()
            except Exception as e:
                logger.error(f"خطا در get_concept_vector با smart_model: {e}")

        # تلاش با teacher_model
        if self.teacher_model and hasattr(self.teacher_model, "get_concept_vector"):
            try:
                text = f"{concept['concept_name']} - {concept.get('definition', '')}"
                vec = self.teacher_model.get_concept_vector(text)
                if vec:
                    # یادگیری از معلم
                    await self._learn_from_teacher(text, {"vector": vec, "concept_id": concept_id})
                    if isinstance(vec, list):
                        return vec
                    if hasattr(vec, "tolist"):
                        return vec.tolist()
            except Exception as e:
                logger.error(f"خطا در get_concept_vector با teacher_model: {e}")

        # fallback: بردار تصادفی
        random_vec = [random.random() for _ in range(128)]
        return random_vec

    async def _get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        بازیابی یک مفهوم از طریق لایه داده
        """
        all_concepts = await self.data_access.load_concepts()
        return all_concepts.get(concept_id)

    # -------------------------------------------------------------------------
    # 8) یافتن مفاهیم مشابه
    # -------------------------------------------------------------------------
    async def find_similar_concepts(self, concept_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        یافتن مفاهیم مشابه از طریق جستجوی برداری (در صورت وجود vector_store)،
        یا محاسبه شباهت کسینوسی با همه مفاهیم.
        """
        concept = await self._get_concept_by_id(concept_id)
        if not concept:
            logger.error(f"مفهوم با شناسه {concept_id} یافت نشد.")
            return []

        query_vector = await self.get_concept_vector(concept_id)

        if self.vector_store and hasattr(self.vector_store, "search"):
            try:
                # جستجوی برداری در مجموعه domain_vectors
                results = await self.vector_store.search("domain_vectors", query_vector, top_k=top_k + 1)
                # فیلتر کردن خود مفهوم
                filtered = []
                for r in results:
                    if r["id"] == concept_id:
                        continue
                    sim = r["score"]
                    c_info = await self._get_concept_by_id(r["id"])
                    if c_info:
                        filtered.append({
                            "concept_id": r["id"],
                            "concept_name": c_info["concept_name"],
                            "similarity": sim
                        })
                return filtered[:top_k]
            except Exception as e:
                logger.error(f"خطا در find_similar_concepts با vector_store: {e}")

        # در غیر این صورت: شباهت کسینوسی با همه مفاهیم
        all_concepts = await self.data_access.load_concepts()
        sims = []
        for cid, cinfo in all_concepts.items():
            if cid == concept_id:
                continue
            vec2 = await self.get_concept_vector(cid)
            sim = self._cosine_similarity(query_vector, vec2)
            sims.append((cid, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_similar = sims[:top_k]
        results = []
        for cid, sim in top_similar:
            c_info = all_concepts[cid]
            results.append({
                "concept_id": cid,
                "concept_name": c_info["concept_name"],
                "similarity": sim
            })
        return results

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        محاسبه شباهت کسینوسی ساده
        """
        if len(v1) != len(v2) or not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def _calculate_concept_similarity(self, concept_id_1: str, concept_id_2: str) -> float:
        """
        دریافت بردار دو مفهوم و محاسبه شباهت کسینوسی
        """
        v1 = await self.get_concept_vector(concept_id_1)
        v2 = await self.get_concept_vector(concept_id_2)
        return self._cosine_similarity(v1, v2)

    # -------------------------------------------------------------------------
    # متدهای کمکی
    # -------------------------------------------------------------------------
    async def _learn_from_teacher(self, text: str, data: Any):
        """
        نمونه ساده یادگیری از معلم
        """
        if self.smart_model and hasattr(self.smart_model, "learn_from_teacher"):
            try:
                # فرضاً smart_model.learn_from_teacher(text, data)
                self.smart_model.learn_from_teacher(text, data)
            except Exception as e:
                logger.error(f"خطا در learn_from_teacher: {e}")


# ----------------------------------------------------------------------------
# نمونه تست مستقل
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def main():
        analyzer = DomainAnalysis()
        await analyzer.data_access.connect_resources()

        # 1) تست find_domain_for_text
        text = "این متن درباره مهندسی مکانیک و طراحی قطعات است."
        domains = await analyzer.find_domain_for_text(text)
        print("find_domain_for_text:", domains)

        # 2) تست discover_new_domains
        new_doms = await analyzer.discover_new_domains("یک متن که حوزه جدیدی را توصیف می‌کند...")
        print("discover_new_domains:", new_doms)

        # 3) تست discover_new_concepts
        new_concepts = await analyzer.discover_new_concepts("اینجا درباره الگوریتم‌های تستی صحبت می‌کنیم", "IT")
        print("discover_new_concepts:", new_concepts)

        # 4) تست discover_relations
        new_rels = await analyzer.discover_relations("IT")
        print("discover_relations:", new_rels)

        # 5) تست get_domain_hierarchy
        hierarchy = await analyzer.get_domain_hierarchy("IT")
        print("get_domain_hierarchy:", hierarchy)

        # 6) تست answer_domain_question
        ans = await analyzer.answer_domain_question("الگوریتم مرتب‌سازی چیست؟", "IT")
        print("answer_domain_question:", ans)

        # 7) تست get_concept_vector
        vec = await analyzer.get_concept_vector("c_1002_4_1")  # مثال: الگوریتم
        print("get_concept_vector:", vec[:5], "...")

        # 8) تست find_similar_concepts
        sim_concepts = await analyzer.find_similar_concepts("c_1002_4_1", top_k=3)
        print("find_similar_concepts:", sim_concepts)

        await analyzer.data_access.disconnect_resources()

    asyncio.run(main())
