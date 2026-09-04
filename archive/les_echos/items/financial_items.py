import scrapy


class NewsItem(scrapy.Item):
    # ── Identifiers ──────────────────────────────────────────────────────────
    source_url      = scrapy.Field()   # canonical article URL
    article_id      = scrapy.Field()   # hash(source_url) for dedup

    # ── Content ──────────────────────────────────────────────────────────────
    headline        = scrapy.Field()   # <h1>
    summary         = scrapy.Field()   # meta description or lead paragraph
    content         = scrapy.Field()   # full article body text
    author          = scrapy.Field()

    # ── Temporal ─────────────────────────────────────────────────────────────
    publish_date    = scrapy.Field()   # ISO 8601 string
    extract_date    = scrapy.Field()   # crawl timestamp

    # ── Classification ───────────────────────────────────────────────────────
    categories      = scrapy.Field()   # tags/rubrics from the article page
    language        = scrapy.Field()   # always 'fr' for Les Echos

    # ── Company signals ──────────────────────────────────────────────────────
    companies_mentioned = scrapy.Field()  # list of {name, ticker, euronext_code}

    # ── NLP hooks (left empty at scrape time, filled post-processing) ─────────
    sentiment_label = scrapy.Field()   # positive / negative / neutral
    sentiment_score = scrapy.Field()   # float [-1, 1]
    ner_entities    = scrapy.Field()   # list of {text, label}
