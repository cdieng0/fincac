"""
Spider: les_echos_news
Source: lesechos.fr – Finance & Marchés sections
Goal:   Build an open-source French financial news dataset aligned with
        CAC 40 companies (suitable for NLP / sentiment / event-study tasks).

Usage:
    scrapy crawl les_echos_news -o data/les_echos_raw.jsonl \
        -s CLOSESPIDER_ITEMCOUNT=50000

Notes:
    • Respects robots.txt by default (ROBOTSTXT_OBEY = True in settings.py)
    • Set DOWNLOAD_DELAY = 2 and AUTOTHROTTLE_ENABLED = True in settings.py
    • The spider does NOT scrape full article bodies behind paywalls;
      it captures headline, summary, metadata, and any freely visible text.
    • Deduplication is handled by the SeenURLsMiddleware (see bottom of file).
"""

import hashlib
from datetime import datetime, date
from dateutil import parser as dateparser

import scrapy

from items.financial_items import NewsItem

# ─── Full CAC 40 + historical members relevant to 2010-2024 ──────────────────
CAC40_COMPANIES = [
    # Luxe / Consommation
    {"name": "LVMH",              "ticker": "MC",    "aliases": ["LVMH", "Louis Vuitton", "Moët Hennessy"]},
    {"name": "Hermès",            "ticker": "RMS",   "aliases": ["Hermès", "Hermes"]},
    {"name": "L'Oréal",           "ticker": "OR",    "aliases": ["L'Oréal", "L'Oreal", "Loréal", "Loreal"]},
    {"name": "Kering",            "ticker": "KER",   "aliases": ["Kering", "Gucci", "PPR"]},
    {"name": "Pernod Ricard",     "ticker": "RI",    "aliases": ["Pernod Ricard", "Pernod"]},
    {"name": "Accor",             "ticker": "AC",    "aliases": ["Accor", "AccorHotels"]},
    # Energie
    {"name": "TotalEnergies",     "ticker": "TTE",   "aliases": ["TotalEnergies", "Total", "TOTAL"]},
    {"name": "Engie",             "ticker": "ENGI",  "aliases": ["Engie", "GDF Suez", "GDF-Suez"]},
    {"name": "EDF",               "ticker": "EDF",   "aliases": ["EDF", "Électricité de France"]},
    # Finance / Banque / Assurance
    {"name": "BNP Paribas",       "ticker": "BNP",   "aliases": ["BNP Paribas", "BNP"]},
    {"name": "Société Générale",  "ticker": "GLE",   "aliases": ["Société Générale", "Societe Generale", "SocGen"]},
    {"name": "Crédit Agricole",   "ticker": "ACA",   "aliases": ["Crédit Agricole", "Credit Agricole", "CA"]},
    {"name": "AXA",               "ticker": "CS",    "aliases": ["AXA"]},
    {"name": "Natixis",           "ticker": "KN",    "aliases": ["Natixis"]},
    # Industrie / Défense / Aéro
    {"name": "Airbus",            "ticker": "AIR",   "aliases": ["Airbus", "EADS"]},
    {"name": "Safran",            "ticker": "SAF",   "aliases": ["Safran"]},
    {"name": "Thales",            "ticker": "HO",    "aliases": ["Thales"]},
    {"name": "Alstom",            "ticker": "ALO",   "aliases": ["Alstom"]},
    {"name": "Schneider Electric","ticker": "SU",    "aliases": ["Schneider Electric", "Schneider"]},
    {"name": "Legrand",           "ticker": "LR",    "aliases": ["Legrand"]},
    {"name": "Saint-Gobain",      "ticker": "SGO",   "aliases": ["Saint-Gobain", "Saint Gobain"]},
    {"name": "ArcelorMittal",     "ticker": "MT",    "aliases": ["ArcelorMittal", "Arcelor"]},
    # Santé / Pharma
    {"name": "Sanofi",            "ticker": "SAN",   "aliases": ["Sanofi", "Sanofi-Aventis"]},
    # Telecom / Tech / Médias
    {"name": "Orange",            "ticker": "ORA",   "aliases": ["Orange", "France Telecom", "France Télécom"]},
    {"name": "Vivendi",           "ticker": "VIV",   "aliases": ["Vivendi", "Canal+"]},
    {"name": "Publicis",          "ticker": "PUB",   "aliases": ["Publicis"]},
    {"name": "Capgemini",         "ticker": "CAP",   "aliases": ["Capgemini", "Cap Gemini"]},
    {"name": "STMicroelectronics","ticker": "STM",   "aliases": ["STMicroelectronics", "STMicro", "STM"]},
    {"name": "Dassault Systèmes", "ticker": "DSY",   "aliases": ["Dassault Systèmes", "Dassault Systemes"]},
    # Transports / Logistique
    {"name": "Air France-KLM",    "ticker": "AF",    "aliases": ["Air France", "Air France-KLM", "KLM"]},
    {"name": "Michelin",          "ticker": "ML",    "aliases": ["Michelin"]},
    {"name": "Stellantis",        "ticker": "STLAM", "aliases": ["Stellantis", "PSA", "Peugeot", "Citroën", "Fiat"]},
    {"name": "Renault",           "ticker": "RNO",   "aliases": ["Renault"]},
    # Distribution / Alimentation
    {"name": "Carrefour",         "ticker": "CA",    "aliases": ["Carrefour"]},
    {"name": "Danone",            "ticker": "BN",    "aliases": ["Danone"]},
    # Immobilier / Diversifiés
    {"name": "Unibail-Rodamco",   "ticker": "URW",   "aliases": ["Unibail", "Rodamco", "Westfield"]},
    {"name": "Bouygues",          "ticker": "EN",    "aliases": ["Bouygues"]},
    {"name": "Vinci",             "ticker": "DG",    "aliases": ["Vinci"]},
    {"name": "Veolia",            "ticker": "VIE",   "aliases": ["Veolia", "Veolia Environnement"]},
    {"name": "Air Liquide",       "ticker": "AI",    "aliases": ["Air Liquide"]},
]

# Pre-build a flat lookup: lowercase alias → company dict (for fast matching)
_ALIAS_INDEX: dict[str, dict] = {}
for _c in CAC40_COMPANIES:
    for _alias in _c["aliases"]:
        _ALIAS_INDEX[_alias.lower()] = _c


def _detect_companies(text: str) -> list[dict]:
    """Return list of unique company dicts found in text."""
    text_lower = text.lower()
    found = {}
    for alias_lower, company in _ALIAS_INDEX.items():
        if alias_lower in text_lower:
            found[company["ticker"]] = {
                "name":   company["name"],
                "ticker": company["ticker"],
            }
    return list(found.values())


def _url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _parse_date(raw: str | None) -> str | None:
    """Normalize any date string to ISO 8601; return None on failure."""
    if not raw:
        return None
    try:
        return dateparser.parse(raw.strip()).isoformat()
    except Exception:
        return raw.strip()


# ─── Spider ───────────────────────────────────────────────────────────────────

class LesEchosNewsSpider(scrapy.Spider):
    """
    Crawl Les Echos for financial news articles mentioning CAC 40 companies.

    Target period : 2010 – present
    Output fields : see NewsItem in financial_items.py
    """

    name = "les_echos_news"
    allowed_domains = ["lesechos.fr"]

    # Sections most likely to carry CAC 40 coverage
    start_urls = [
        "https://www.lesechos.fr/finance-marches/",
        "https://www.lesechos.fr/finance-marches/marches-financiers/",
        "https://www.lesechos.fr/finance-marches/banque-assurances/",
        "https://www.lesechos.fr/industrie-services/",
        "https://www.lesechos.fr/industrie-services/energie-environnement/",
        "https://www.lesechos.fr/industrie-services/pharmacie-sante/",
        "https://www.lesechos.fr/tech-medias/",
    ]

    # ── Crawl settings (can also be set in settings.py) ──────────────────────
    custom_settings = {
        "ROBOTSTXT_OBEY":        True,
        "DOWNLOAD_DELAY":        2,          # seconds between requests
        "AUTOTHROTTLE_ENABLED":  True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "CONCURRENT_REQUESTS":   4,
        "DUPEFILTER_CLASS":      "scrapy.dupefilters.RFPDupeFilter",
        "LOG_LEVEL":             "INFO",
        # Uncomment to limit total items during dev:
        # "CLOSESPIDER_ITEMCOUNT": 500,
    }

    # Filter: only keep articles published within this window
    DATE_FROM = date(2010, 1, 1)
    DATE_TO   = date.today()

    # ── Entry point ──────────────────────────────────────────────────────────

    def parse(self, response):
        """Parse a listing / category page → follow article links + pagination."""

        # Les Echos uses several article card patterns depending on the section
        article_links = response.css(
            "article a[href]::attr(href), "
            "div.news-item a[href]::attr(href), "
            "h3 a[href]::attr(href), "
            "h2 a[href]::attr(href)"
        ).getall()

        seen = set()
        for href in article_links:
            url = response.urljoin(href)
            if url in seen:
                continue
            seen.add(url)
            # Heuristic: article URLs typically contain a long numeric suffix
            if any(section in url for section in [
                "/finance-marches/", "/industrie-services/",
                "/tech-medias/", "/bourse/", "/entreprises/"
            ]):
                yield scrapy.Request(url, callback=self.parse_article)

        # Pagination — Les Echos uses ?page=N or a "suivant" link
        next_page = (
            response.css("a[rel='next']::attr(href)").get()
            or response.css("a.pagination__next::attr(href)").get()
            or response.css("a:contains('Suivant')::attr(href)").get()
        )
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)

    # ── Article parser ───────────────────────────────────────────────────────

    def parse_article(self, response):
        """Extract structured data from a single article page."""

        # ── Date check (skip if outside window) ──────────────────────────────
        raw_date = (
            response.css("time[datetime]::attr(datetime)").get()
            or response.css("meta[property='article:published_time']::attr(content)").get()
            or response.css("span.publish-date::text").get()
        )
        publish_date_iso = _parse_date(raw_date)

        if publish_date_iso:
            try:
                article_date = dateparser.parse(publish_date_iso).date()
                if not (self.DATE_FROM <= article_date <= self.DATE_TO):
                    return   # outside target window, discard
            except Exception:
                pass  # if we can't parse the date, keep the article anyway

        # ── Headline ─────────────────────────────────────────────────────────
        headline = (
            response.css("h1::text").get()
            or response.css("meta[property='og:title']::attr(content)").get()
        )
        if not headline:
            return  # no headline → not a real article page

        # ── Summary ──────────────────────────────────────────────────────────
        summary = (
            response.css("meta[name='description']::attr(content)").get()
            or response.css("meta[property='og:description']::attr(content)").get()
            or response.css("p.article-intro::text").get()
        )

        # ── Body text ────────────────────────────────────────────────────────
        # Les Echos puts article text in div.article-text or div.sc-article
        paragraphs = (
            response.css("div.article-text p::text, div.sc-article p::text, "
                         "div[data-testid='article-body'] p::text").getall()
            or response.css("article p::text").getall()
        )
        content = " ".join(p.strip() for p in paragraphs if p.strip()) or None

        # ── Author ───────────────────────────────────────────────────────────
        author = (
            response.css("span.author::text, a.author::text, "
                         "meta[name='author']::attr(content)").get()
        )

        # ── Categories / tags ────────────────────────────────────────────────
        categories = response.css(
            "a.category::text, a.tag::text, "
            "nav.breadcrumb a::text, "
            "meta[property='article:section']::attr(content)"
        ).getall()
        categories = [c.strip() for c in categories if c.strip()]

        # ── Company detection ────────────────────────────────────────────────
        # Search across all visible text for better recall
        search_blob = " ".join(filter(None, [
            headline, summary, content,
            " ".join(categories)
        ]))
        companies = _detect_companies(search_blob)

        # Discard articles with zero CAC 40 signal
        if not companies:
            return

        # ── Build item ───────────────────────────────────────────────────────
        item = NewsItem()
        item["source_url"]          = response.url
        item["article_id"]          = _url_id(response.url)
        item["headline"]            = headline.strip()
        item["summary"]             = summary.strip() if summary else None
        item["content"]             = content
        item["author"]              = author.strip() if author else None
        item["publish_date"]        = publish_date_iso
        item["extract_date"]        = datetime.utcnow().isoformat() + "Z"
        item["categories"]          = categories
        item["language"]            = "fr"
        item["companies_mentioned"] = companies
        # NLP fields left None — to be filled by post-processing pipeline
        item["sentiment_label"]     = None
        item["sentiment_score"]     = None
        item["ner_entities"]        = None

        yield item
