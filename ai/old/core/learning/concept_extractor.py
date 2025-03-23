from typing import Dict, List, Set, Tuple, Optional
import spacy
import networkx as nx
from collections import defaultdict
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    name: str
    type: str
    importance: float
    references: List[str]
    language: str
    metadata: Dict


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    strength: float
    evidence: List[str]


class ConceptExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.nlp_models = {
            'en': spacy.load('en_core_web_lg'),
            'fa': spacy.load('xx_ent_wiki_sm')  # مدل چندزبانه
        }
        self.concept_graph = nx.DiGraph()
        self.importance_threshold = config.get('importance_threshold', 0.3)

    async def extract_concepts(self, text: str, language: str = 'en') -> List[Concept]:
        try:
            nlp = self.nlp_models.get(language)
            if not nlp:
                raise ValueError(f"Language {language} not supported")

            doc = nlp(text)
            concepts = []

            # استخراج موجودیت‌های نامی
            named_entities = self._extract_named_entities(doc)

            # استخراج عبارات کلیدی
            key_phrases = self._extract_key_phrases(doc)

            # استخراج اصطلاحات تخصصی
            technical_terms = self._extract_technical_terms(text, language)

            # ترکیب نتایج
            all_concepts = set()
            all_concepts.update(named_entities)
            all_concepts.update(key_phrases)
            all_concepts.update(technical_terms)

            for concept_text in all_concepts:
                concept = await self._create_concept(concept_text, doc, language)
                if concept.importance >= self.importance_threshold:
                    concepts.append(concept)
                    self._add_to_graph(concept)

            return concepts

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []

    async def extract_relationships(self, concepts: List[Concept], text: str) -> List[Relationship]:
        try:
            relationships = []

            # یافتن روابط مستقیم
            direct_rels = self._find_direct_relationships(concepts, text)
            relationships.extend(direct_rels)

            # یافتن روابط معنایی
            semantic_rels = await self._find_semantic_relationships(concepts)
            relationships.extend(semantic_rels)

            # یافتن روابط مبتنی بر گراف
            graph_rels = self._find_graph_relationships(concepts)
            relationships.extend(graph_rels)

            return self._filter_relationships(relationships)

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def _extract_named_entities(self, doc) -> Set[str]:
        entities = set()
        for ent in doc.ents:
            if len(ent.text) > 1:  # حذف موجودیت‌های تک‌حرفی
                entities.add(ent.text)
        return entities

    def _extract_key_phrases(self, doc) -> Set[str]:
        phrases = set()
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # محدودیت طول عبارت
                phrases.add(chunk.text)
        return phrases

    def _extract_technical_terms(self, text: str, language: str) -> Set[str]:
        # الگوهای مختلف برای تشخیص اصطلاحات تخصصی
        patterns = {
            'en': [
                r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # CamelCase
                r'\b[A-Z]{2,}\b',  # UPPERCASE
                r'\b\w+[_-]\w+\b'  # با خط تیره یا زیرخط
            ],
            'fa': [
                r'[\u0600-\u06FF]+[‌][\u0600-\u06FF]+',  # کلمات مرکب فارسی
                r'[\u0600-\u06FF]{4,}'  # کلمات طولانی فارسی
            ]
        }

        terms = set()
        for pattern in patterns.get(language, []):
            matches = re.finditer(pattern, text)
            terms.update(match.group() for match in matches)
        return terms

    async def _create_concept(self, text: str, doc, language: str) -> Concept:
        # محاسبه اهمیت مفهوم
        importance = self._calculate_importance(text, doc)

        # یافتن ارجاعات
        references = self._find_references(text, doc)

        # تشخیص نوع مفهوم
        concept_type = self._determine_concept_type(text, doc)

        return Concept(
            name=text,
            type=concept_type,
            importance=importance,
            references=references,
            language=language,
            metadata=self._extract_concept_metadata(text, doc)
        )

    def _calculate_importance(self, text: str, doc) -> float:
        # ترکیب معیارهای مختلف برای محاسبه اهمیت
        scores = {
            'frequency': self._calculate_frequency_score(text, doc),
            'centrality': self._calculate_centrality_score(text),
            'specificity': self._calculate_specificity_score(text, doc)
        }
        weights = {'frequency': 0.4, 'centrality': 0.3, 'specificity': 0.3}
        return sum(score * weights[metric] for metric, score in scores.items())

    def _calculate_frequency_score(self, text: str, doc) -> float:
        # محاسبه نمره بر اساس فراوانی نسبی
        word_count = len(doc)
        text_count = len(re.findall(re.escape(text), doc.text))
        return min(1.0, text_count / (word_count * 0.1))

    def _calculate_centrality_score(self, text: str) -> float:
        if text not in self.concept_graph:
            return 0.0
        # محاسبه مرکزیت در گراف مفاهیم
        centrality = nx.pagerank(self.concept_graph).get(text, 0)
        return centrality

    def _calculate_specificity_score(self, text: str, doc) -> float:
        # محاسبه نمره بر اساس خاص بودن مفهوم
        if len(text.split()) > 2:
            return 0.8  # عبارات طولانی‌تر معمولاً خاص‌تر هستند
        return 0.5

    def _find_references(self, text: str, doc) -> List[str]:
        # یافتن جملاتی که به مفهوم ارجاع می‌دهند
        references = []
        for sent in doc.sents:
            if text.lower() in sent.text.lower():
                references.append(sent.text)
        return references[:3]  # محدود کردن تعداد ارجاعات

    def _determine_concept_type(self, text: str, doc) -> str:
        # تشخیص نوع مفهوم بر اساس ویژگی‌های متنی
        types = {
            'PERSON': ['human', 'name'],
            'ORG': ['organization', 'company'],
            'TECH': ['technology', 'system'],
            'CONCEPT': ['abstract', 'theory']
        }

        for ent in doc.ents:
            if text in ent.text:
                return ent.label_
        return 'CONCEPT'

    def _add_to_graph(self, concept: Concept) -> None:
        if concept.name not in self.concept_graph:
            self.concept_graph.add_node(
                concept.name,
                type=concept.type,
                importance=concept.importance
            )

    def _find_direct_relationships(self, concepts: List[Concept], text: str) -> List[Relationship]:
        relationships = []
        concept_pairs = [(c1, c2) for i, c1 in enumerate(concepts)
                         for c2 in concepts[i + 1:]]

        for c1, c2 in concept_pairs:
            # یافتن جملاتی که هر دو مفهوم را دارند
            evidence = self._find_cooccurrence(c1.name, c2.name, text)
            if evidence:
                rel_type = self._determine_relationship_type(c1, c2, evidence)
                strength = self._calculate_relationship_strength(c1, c2, evidence)

                relationships.append(Relationship(
                    source=c1.name,
                    target=c2.name,
                    type=rel_type,
                    strength=strength,
                    evidence=evidence
                ))

        return relationships

    async def _find_semantic_relationships(self, concepts: List[Concept]) -> List[Relationship]:
        relationships = []
        for c1 in concepts:
            for c2 in concepts:
                if c1 != c2:
                    similarity = await self._calculate_semantic_similarity(c1, c2)
                    if similarity > 0.7:
                        relationships.append(Relationship(
                            source=c1.name,
                            target=c2.name,
                            type='SEMANTIC_SIMILARITY',
                            strength=similarity,
                            evidence=[]
                        ))
        return relationships

    def _find_graph_relationships(self, concepts: List[Concept]) -> List[Relationship]:
        relationships = []
        concept_names = {c.name for c in concepts}

        for c1 in concept_names:
            for c2 in concept_names:
                if c1 != c2 and self.concept_graph.has_edge(c1, c2):
                    edge_data = self.concept_graph.get_edge_data(c1, c2)
                    relationships.append(Relationship(
                        source=c1,
                        target=c2,
                        type=edge_data.get('type', 'RELATED'),
                        strength=edge_data.get('weight', 0.5),
                        evidence=edge_data.get('evidence', [])
                    ))

        return relationships

    def _filter_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        # حذف روابط ضعیف و تکراری
        filtered = []
        seen = set()

        for rel in sorted(relationships, key=lambda x: x.strength, reverse=True):
            key = f"{rel.source}_{rel.type}_{rel.target}"
            if key not in seen and rel.strength >= 0.3:
                filtered.append(rel)
                seen.add(key)

        return filtered

    def _find_cooccurrence(self, concept1: str, concept2: str, text: str) -> List[str]:
        # یافتن جملاتی که هر دو مفهوم را دارند
        sentences = re.split('[.!?]', text)
        evidence = []

        for sent in sentences:
            if concept1.lower() in sent.lower() and concept2.lower() in sent.lower():
                evidence.append(sent.strip())

        return evidence

    def _determine_relationship_type(self, concept1: Concept, concept2: Concept,
                                     evidence: List[str]) -> str:
        # تعیین نوع رابطه بر اساس شواهد متنی
        if not evidence:
            return 'RELATED'

        text = ' '.join(evidence)

        # الگوهای رابطه
        patterns = {
            'IS_A': [r'is a', r'type of', r'kind of', r'نوعی از', r'از نوع'],
            'PART_OF': [r'part of', r'contains', r'includes', r'بخشی از', r'شامل'],
            'CAUSES': [r'causes', r'leads to', r'results in', r'باعث', r'منجر به'],
            'DEPENDS_ON': [r'depends on', r'requires', r'needs', r'نیاز به', r'وابسته به']
        }

        for rel_type, rel_patterns in patterns.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in rel_patterns):
                return rel_type

        return 'RELATED'

    def _calculate_relationship_strength(self, concept1: Concept, concept2: Concept,
                                         evidence: List[str]) -> float:
        # محاسبه قدرت رابطه
        if not evidence:
            return 0.0

        factors = {
            'evidence_count': len(evidence) / 5.0,  # نرمال‌سازی تعداد شواهد
            'concept_importance': (concept1.importance + concept2.importance) / 2,
            'relationship_specificity': self._calculate_relationship_specificity(evidence)
        }

        weights = {
            'evidence_count': 0.4,
            'concept_importance': 0.3,
            'relationship_specificity': 0.3
        }

        return min(1.0, sum(score * weights[factor] for factor, score in factors.items()))

    def _calculate_relationship_specificity(self, evidence: List[str]) -> float:
        # محاسبه خاص بودن رابطه
        total_words = sum(len(e.split()) for e in evidence)
        return min(1.0, total_words / 100.0)  # نرمال‌سازی بر اساس طول متن

    async def _calculate_semantic_similarity(self, concept1: Concept,
                                             concept2: Concept) -> float:
        try:
            # استفاده از مدل زبانی برای محاسبه شباهت معنایی
            if concept1.language != concept2.language:
                return 0.0

            nlp = self.nlp_models.get(concept1.language)
            if not nlp:
                return 0.0

            doc1 = nlp(concept1.name)
            doc2 = nlp(concept2.name)

            return doc1.similarity(doc2)

        except Exception:
            return 0.0