# settings.py  ── Scrapy project settings for les_echos_news

BOT_NAME = "les_echos_scraper"

SPIDER_MODULES   = ["les_echos_scraper.spiders"]
NEWSPIDER_MODULE = "les_echos_scraper.spiders"

# ── Ethics & rate limiting ────────────────────────────────────────────────────
ROBOTSTXT_OBEY          = True
DOWNLOAD_DELAY          = 2          # minimum delay between requests (seconds)
AUTOTHROTTLE_ENABLED    = True
AUTOTHROTTLE_START_DELAY      = 2
AUTOTHROTTLE_MAX_DELAY        = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Identify your bot (important for open-source / academic use)
USER_AGENT = (
    "les-echos-cac40-dataset-bot/1.0 "
    "(academic NLP research; contact: your@email.com)"
)

# ── Deduplication ────────────────────────────────────────────────────────────
DUPEFILTER_CLASS = "scrapy.dupefilters.RFPDupeFilter"

# ── Output ───────────────────────────────────────────────────────────────────
# Run with: scrapy crawl les_echos_news -o data/les_echos_raw.jsonl
FEEDS = {
    "data/les_echos_%(time)s.jsonl": {
        "format":   "jsonlines",
        "encoding": "utf-8",
        "store_empty": False,
    }
}

# ── Pipelines ────────────────────────────────────────────────────────────────
ITEM_PIPELINES = {
    "les_echos_scraper.pipelines.DeduplicationPipeline": 100,
    "les_echos_scraper.pipelines.ValidationPipeline":    200,
}

# ── Misc ─────────────────────────────────────────────────────────────────────
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
LOG_LEVEL = "INFO"
