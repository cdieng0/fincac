# pipelines.py

from scrapy.exceptions import DropItem


class DeduplicationPipeline:
    """Drop articles already seen in this crawl session (URL-based)."""

    def __init__(self):
        self._seen_ids: set[str] = set()

    def process_item(self, item, spider):
        article_id = item.get("article_id")
        if article_id in self._seen_ids:
            raise DropItem(f"Duplicate article: {item.get('source_url')}")
        self._seen_ids.add(article_id)
        return item


class ValidationPipeline:
    """Drop items missing the minimum required fields."""

    REQUIRED = {"headline", "source_url", "companies_mentioned"}

    def process_item(self, item, spider):
        missing = [f for f in self.REQUIRED if not item.get(f)]
        if missing:
            raise DropItem(f"Missing fields {missing} for {item.get('source_url')}")
        return item
