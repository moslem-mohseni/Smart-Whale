# persian/language_processors/domain/domain_services.py

"""
ماژول domain_services.py

این ماژول منطق تجاری مربوط به مدیریت دانش حوزه‌ای را پیاده‌سازی می‌کند.
توابع این ماژول شامل افزودن، به‌روزرسانی، ادغام حوزه‌ها، مفاهیم، روابط و ویژگی‌ها می‌باشد.
این توابع از لایه داده (domain_data.py) استفاده می‌کنند و با ساختار async/await نوشته شده‌اند.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

# فرض بر این است که domain_models.py وجود دارد و کلاس‌های زیر در آن تعریف شده‌اند.
# اگر از ساختار dict استفاده می‌کنید، می‌توانید همینجا دیکشنری بسازید.
from domain_models import Domain, Concept, Relation, Attribute

# لایه داده
from domain_data import DomainDataAccess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DomainServices:
    def __init__(self, data_access: Optional[DomainDataAccess] = None):
        """
        سازنده DomainServices

        Args:
            data_access (Optional[DomainDataAccess]): شیء دسترسی به داده‌های حوزه؛
                در صورت عدم ارائه، یک شیء جدید ساخته می‌شود.
        """
        self.data_access = data_access if data_access is not None else DomainDataAccess()

    async def add_domain(self, domain_name: str, domain_code: str,
                         description: str = "",
                         parent_domain_code: Optional[str] = None) -> Optional[Domain]:
        """
        افزودن یک حوزه جدید به سیستم.

        Args:
            domain_name (str): نام حوزه
            domain_code (str): کد یکتای حوزه
            description (str): توضیحات مربوط به حوزه
            parent_domain_code (Optional[str]): کد حوزه والد (اختیاری)

        Returns:
            Optional[Domain]: شیء Domain ایجاد‌شده یا None در صورت خطا
        """
        # بررسی وجود حوزه با این کد
        existing = await self.data_access.find_domain_by_code(domain_code)
        if existing:
            logger.warning(f"حوزه با کد {domain_code} قبلاً موجود است.")
            # می‌توانید همین شیء را بازگردانید یا None
            return Domain(**existing)

        # در صورت وجود parent_domain_code، آن را جستجو می‌کنیم
        parent_domain_id = ""
        if parent_domain_code:
            parent = await self.data_access.find_domain_by_code(parent_domain_code)
            if not parent:
                logger.error(f"حوزه والد با کد {parent_domain_code} یافت نشد.")
                return None
            parent_domain_id = parent["domain_id"]

        # ساخت شیء Domain
        domain_id = f"d_{int(time.time())}"
        new_domain = Domain(
            domain_id=domain_id,
            domain_name=domain_name,
            domain_code=domain_code,
            parent_domain=parent_domain_id,
            description=description,
            popularity=0.5,  # مقدار پیش‌فرض
            source="api",
            discovery_time=datetime.now()
        )

        # ذخیره در پایگاه داده
        success = await self.data_access.store_domain(new_domain.to_dict())
        if success:
            logger.info(f"حوزه جدید با کد {domain_code} اضافه شد.")
            return new_domain
        else:
            logger.error("خطا در افزودن حوزه.")
            return None

    async def add_concept(self, domain_code: str, concept_name: str,
                          definition: str,
                          examples: Optional[List[str]] = None,
                          confidence: float = 0.8) -> Optional[Concept]:
        """
        افزودن یک مفهوم جدید به یک حوزه

        Args:
            domain_code (str): کد حوزه مربوطه
            concept_name (str): نام مفهوم
            definition (str): تعریف مفهوم
            examples (Optional[List[str]]): لیست مثال‌ها (اختیاری)
            confidence (float): سطح اطمینان پیش‌فرض

        Returns:
            Optional[Concept]: شیء Concept ایجاد‌شده یا None در صورت خطا
        """
        # یافتن حوزه مربوطه
        domain_dict = await self.data_access.find_domain_by_code(domain_code)
        if not domain_dict:
            logger.error(f"حوزه با کد {domain_code} یافت نشد.")
            return None
        domain_id = domain_dict["domain_id"]

        # بررسی تکراری نبودن مفهوم در این حوزه
        domain_concepts = await self.data_access.find_domain_concepts(domain_id)
        for c in domain_concepts:
            if c["concept_name"].lower() == concept_name.lower():
                logger.warning(f"مفهوم {concept_name} در حوزه {domain_code} قبلاً وجود دارد.")
                # افزایش شمارنده استفاده از مفهوم
                await self.data_access.increment_concept_usage(c["concept_id"])
                return Concept(**c)

        # ساخت شیء Concept
        concept_id = f"c_{int(time.time())}"
        new_concept = Concept(
            concept_id=concept_id,
            domain_id=domain_id,
            concept_name=concept_name,
            definition=definition,
            examples=examples if examples else [],
            confidence=confidence,
            source="api",
            usage_count=1,
            discovery_time=datetime.now()
        )

        success = await self.data_access.store_concept(new_concept.to_dict())
        if success:
            logger.info(f"مفهوم {concept_name} در حوزه {domain_code} اضافه شد.")
            return new_concept
        else:
            logger.error("خطا در افزودن مفهوم.")
            return None

    async def add_relation(self, source_concept_id: str,
                           target_concept_id: str,
                           relation_type: str,
                           description: str = "",
                           confidence: float = 0.8) -> Optional[Relation]:
        """
        افزودن یک رابطه بین دو مفهوم

        Args:
            source_concept_id (str): شناسه مفهوم منبع
            target_concept_id (str): شناسه مفهوم هدف
            relation_type (str): نوع رابطه (مثلاً IS_A، RELATED_TO)
            description (str): توضیحات رابطه
            confidence (float): سطح اطمینان

        Returns:
            Optional[Relation]: شیء Relation ایجادشده یا None در صورت خطا
        """
        # بررسی تکراری نبودن رابطه
        existing_rels = await self.data_access.find_relations_between(source_concept_id, target_concept_id, relation_type)
        if existing_rels:
            logger.warning("رابطه مورد نظر قبلاً وجود دارد. افزایش شمارنده استفاده.")
            await self.data_access.increment_relation_usage(existing_rels[0]["relation_id"])
            return Relation(**existing_rels[0])

        # ساخت شیء Relation
        relation_id = f"r_{int(time.time())}"
        new_relation = Relation(
            relation_id=relation_id,
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            relation_type=relation_type,
            description=description,
            confidence=confidence,
            source="api",
            usage_count=1,
            discovery_time=datetime.now()
        )

        success = await self.data_access.store_relation(new_relation.to_dict())
        if success:
            logger.info("رابطه جدید اضافه شد.")
            return new_relation
        else:
            logger.error("خطا در افزودن رابطه.")
            return None

    async def add_attribute(self, concept_id: str,
                            attribute_name: str,
                            attribute_value: str,
                            description: str = "",
                            confidence: float = 0.8) -> Optional[Attribute]:
        """
        افزودن یک ویژگی به یک مفهوم
        """
        # ابتدا بررسی کنیم این مفهوم اصلاً وجود دارد یا نه
        # می‌توانیم با استفاده از load_concepts یا روش‌های دیگر این کار را انجام دهیم
        # ولی در اینجا فرض می‌کنیم concept_id معتبر است

        # بررسی تکراری نبودن ویژگی
        existing_attrs = await self.data_access.find_attributes_by_concept(concept_id)
        for attr in existing_attrs:
            if attr["attribute_name"].lower() == attribute_name.lower():
                logger.warning("ویژگی مورد نظر قبلاً وجود دارد. بروزرسانی مقدار.")
                await self.data_access.update_attribute_value(attr["attribute_id"], attribute_value)
                return Attribute(**attr)

        # ساخت شیء Attribute
        attribute_id = f"a_{int(time.time())}"
        new_attribute = Attribute(
            attribute_id=attribute_id,
            concept_id=concept_id,
            attribute_name=attribute_name,
            attribute_value=attribute_value,
            description=description,
            confidence=confidence,
            source="api",
            usage_count=1,
            discovery_time=datetime.now()
        )

        success = await self.data_access.store_attribute(new_attribute.to_dict())
        if success:
            logger.info("ویژگی جدید اضافه شد.")
            return new_attribute
        else:
            logger.error("خطا در افزودن ویژگی.")
            return None

    async def merge_domains(self, source_domain_code: str, target_domain_code: str) -> Dict[str, Any]:
        """
        ادغام دو حوزه تخصصی:
          1) انتقال مفاهیم حوزه منبع به حوزه هدف
          2) تغییر والدیت زیرحوزه‌های منبع به هدف
          3) میانگین محبوبیت بین دو حوزه
        """
        source_domain = await self.data_access.find_domain_by_code(source_domain_code)
        target_domain = await self.data_access.find_domain_by_code(target_domain_code)

        if not source_domain:
            logger.error(f"حوزه منبع با کد {source_domain_code} یافت نشد.")
            return {"error": f"حوزه منبع با کد {source_domain_code} یافت نشد."}

        if not target_domain:
            logger.error(f"حوزه هدف با کد {target_domain_code} یافت نشد.")
            return {"error": f"حوزه هدف با کد {target_domain_code} یافت نشد."}

        source_domain_id = source_domain["domain_id"]
        target_domain_id = target_domain["domain_id"]

        # انتقال مفاهیم از حوزه منبع به هدف
        concepts = await self.data_access.find_domain_concepts(source_domain_id)
        moved_count = 0
        for c in concepts:
            concept_id = c["concept_id"]
            success = await self.data_access.update_concept_domain(concept_id, target_domain_id)
            if success:
                moved_count += 1

        # به‌روزرسانی زیرحوزه‌ها
        subdomains = await self.data_access.find_subdomains(source_domain_id)
        subdomains_updated = 0
        for sd in subdomains:
            d_id = sd["domain_id"]
            success = await self.data_access.update_domain_parent(d_id, target_domain_id)
            if success:
                subdomains_updated += 1

        # انتقال محبوبیت
        new_popularity = (source_domain["popularity"] + target_domain["popularity"]) / 2
        popularity_updated = await self.data_access.update_domain_popularity(target_domain_id, new_popularity)

        result = {
            "source_domain": source_domain_code,
            "target_domain": target_domain_code,
            "concepts_moved": moved_count,
            "subdomains_updated": subdomains_updated,
            "new_popularity": new_popularity,
            "popularity_updated": popularity_updated
        }
        logger.info("ادغام حوزه‌ها با موفقیت انجام شد.")
        return result


# نمونه تست در صورت اجرای مستقیم فایل
if __name__ == "__main__":
    import asyncio
    from domain_models import Domain, Concept, Relation, Attribute

    async def main():
        ds = DomainServices()

        # اتصال به منابع (در صورت نیاز)
        await ds.data_access.connect_resources()

        # 1) تست افزودن حوزه
        domain = await ds.add_domain("پزشکی تستی", "MEDICAL_TEST", "توضیحات حوزه تستی")
        if domain:
            print("Added Domain:", domain.to_dict())
        else:
            print("Error adding domain.")

        # 2) تست افزودن مفهوم
        concept = await ds.add_concept("MEDICAL_TEST", "آناتومی تستی", "تعریف تستی آناتومی", ["مثال1", "مثال2"])
        if concept:
            print("Added Concept:", concept.to_dict())
        else:
            print("Error adding concept.")

        # 3) تست افزودن رابطه
        # ابتدا نیاز است منبع و هدف وجود داشته باشند (اینجا فرضی)
        relation = await ds.add_relation("c_12345", "c_67890", "RELATED_TO", "رابطه تستی")
        if relation:
            print("Added Relation:", relation.to_dict())
        else:
            print("Error adding relation.")

        # 4) تست افزودن ویژگی
        attribute = await ds.add_attribute("c_12345", "سطح دشواری", "زیاد")
        if attribute:
            print("Added Attribute:", attribute.to_dict())
        else:
            print("Error adding attribute.")

        # 5) تست ادغام حوزه‌ها
        merge_result = await ds.merge_domains("MEDICAL_TEST", "ENGINEERING")
        print("Merge Result:", merge_result)

        # قطع اتصال
        await ds.data_access.disconnect_resources()

    asyncio.run(main())
