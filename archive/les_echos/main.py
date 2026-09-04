"""
main.py — Point d'entrée autonome du scraper Les Echos CAC 40
─────────────────────────────────────────────────────────────
Usage :
    python main.py

Sortie :
    data/les_echos_YYYYMMDD_HHMMSS.csv

Comportement sur blocage / rate-limit :
    Stop propre dès détection (HTTP 403/429/503 répété ou
    timeout consécutif) + sauvegarde du CSV partiel.

Dépendances :
    pip install scrapy python-dateutil
"""

import csv
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider
from scrapy.utils.project import get_project_settings

# ─── Chemins ──────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).parent
DATA_DIR  = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RUN_TS    = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH  = DATA_DIR / f"les_echos_{RUN_TS}.csv"
LOG_PATH  = DATA_DIR / f"les_echos_{RUN_TS}.log"

# ─── CSV columns (ordre fixe pour la reproductibilité) ───────────────────────

CSV_COLUMNS = [
    "article_id",
    "publish_date",
    "extract_date",
    "headline",
    "summary",
    "content",
    "author",
    "categories",           # sérialisé JSON (liste)
    "companies_mentioned",  # sérialisé JSON (liste de dicts)
    "language",
    "source_url",
    "sentiment_label",
    "sentiment_score",
    "ner_entities",
]

# ─── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Compteurs partagés (accès depuis le pipeline et le signal handler) ───────

_state = {
    "items_scraped":    0,
    "items_dropped":    0,
    "consecutive_errors": 0,
    "csv_writer":       None,
    "csv_file":         None,
    "crawler":          None,
}

MAX_CONSECUTIVE_ERRORS = 5   # Stop propre après N erreurs réseau de suite


# ─── Pipeline CSV inline (évite de dépendre d'un settings.py externe) ────────

class CSVWriterPipeline:
    """
    Écrit chaque item directement dans le CSV à la volée.
    → En cas d'arrêt brutal, les données déjà écrites sont préservées.
    """

    def open_spider(self, spider):
        _state["csv_file"] = open(CSV_PATH, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            _state["csv_file"],
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",   # ignore champs non listés dans CSV_COLUMNS
        )
        writer.writeheader()
        _state["csv_writer"] = writer
        logger.info(f"📂 CSV ouvert : {CSV_PATH}")

    def process_item(self, item, spider):
        row = dict(item)
        # Sérialiser les champs liste/dict en JSON string
        for field in ("categories", "companies_mentioned", "ner_entities"):
            if row.get(field) is not None:
                row[field] = json.dumps(row[field], ensure_ascii=False)
        _state["csv_writer"].writerow(row)
        _state["csv_file"].flush()   # flush à chaque article → sauvegarde partielle garantie
        _state["items_scraped"] += 1
        if _state["items_scraped"] % 50 == 0:
            logger.info(f"  ✓ {_state['items_scraped']} articles sauvegardés")
        return item

    def close_spider(self, spider):
        if _state["csv_file"]:
            _state["csv_file"].close()
            logger.info(f"📂 CSV fermé : {CSV_PATH}")


# ─── Middleware de détection de blocage ───────────────────────────────────────

class BlockDetectionMiddleware:
    """
    Surveille les réponses HTTP et les timeouts.
    Déclenche un stop propre si Les Echos bloque le bot.
    """

    BLOCK_CODES = {403, 429, 503}

    def process_response(self, request, response, spider):
        if response.status in self.BLOCK_CODES:
            _state["consecutive_errors"] += 1
            logger.warning(
                f"⚠️  HTTP {response.status} sur {request.url} "
                f"(erreur consécutive n°{_state['consecutive_errors']})"
            )
            if _state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"🛑 {MAX_CONSECUTIVE_ERRORS} erreurs réseau consécutives → "
                    f"stop propre. {_state['items_scraped']} articles sauvegardés."
                )
                raise CloseSpider(f"Blocage détecté (HTTP {response.status})")
        else:
            _state["consecutive_errors"] = 0   # reset sur succès
        return response

    def process_exception(self, request, exception, spider):
        _state["consecutive_errors"] += 1
        logger.warning(
            f"⚠️  Exception réseau sur {request.url} : {exception} "
            f"(erreur consécutive n°{_state['consecutive_errors']})"
        )
        if _state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
            logger.error(
                f"🛑 {MAX_CONSECUTIVE_ERRORS} exceptions consécutives → "
                f"stop propre. {_state['items_scraped']} articles sauvegardés."
            )
            raise CloseSpider("Trop d'exceptions réseau consécutives")
        return None   # laisse Scrapy gérer (retry si configuré)


# ─── Gestion CTRL+C (SIGINT) ─────────────────────────────────────────────────

def _handle_sigint(sig, frame):
    logger.warning(
        f"\n⚡ CTRL+C reçu — arrêt propre en cours. "
        f"{_state['items_scraped']} articles déjà sauvegardés dans {CSV_PATH}"
    )
    if _state["crawler"]:
        _state["crawler"].engine.close_spider(
            _state["crawler"].spider, "SIGINT reçu"
        )

signal.signal(signal.SIGINT, _handle_sigint)


# ─── Fonction principale ──────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  Les Echos CAC 40 Scraper — démarrage")
    logger.info(f"  Sortie CSV  : {CSV_PATH}")
    logger.info(f"  Log         : {LOG_PATH}")
    logger.info("=" * 60)

    # ── Paramètres Scrapy ──────────────────────────────────────────────────
    settings = get_project_settings()
    settings.setdict({
        # Bot & politesse
        "BOT_NAME":              "les-echos-cac40-dataset-bot",
        "USER_AGENT": (
            "les-echos-cac40-dataset-bot/1.0 "
            "(academic NLP research; open-source dataset)"
        ),
        "ROBOTSTXT_OBEY":        True,
        "DOWNLOAD_DELAY":        2,
        "AUTOTHROTTLE_ENABLED":  True,
        "AUTOTHROTTLE_START_DELAY":          2,
        "AUTOTHROTTLE_MAX_DELAY":            10,
        "AUTOTHROTTLE_TARGET_CONCURRENCY":   1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN":    2,

        # Retry limité (on préfère stop propre)
        "RETRY_ENABLED":         True,
        "RETRY_TIMES":           2,
        "RETRY_HTTP_CODES":      [500, 502, 503, 504],

        # Deduplication URL
        "DUPEFILTER_CLASS":      "scrapy.dupefilters.RFPDupeFilter",

        # Pipeline CSV + middleware blocage (définis dans ce fichier)
        "ITEM_PIPELINES": {
            f"{__name__}.CSVWriterPipeline": 100,
        },
        "DOWNLOADER_MIDDLEWARES": {
            f"{__name__}.BlockDetectionMiddleware": 543,
        },

        # Logs
        "LOG_ENABLED":   True,
        "LOG_LEVEL":     "WARNING",   # Scrapy interne silencieux, on log nous-mêmes
        "LOG_FILE":      str(LOG_PATH),

        # Désactiver la sortie feed de Scrapy (on gère le CSV nous-mêmes)
        "FEEDS": {},
    })

    # ── Import spider ici (évite circular import) ──────────────────────────
    # Ajuste le chemin si ton projet Scrapy a une structure différente
    # APRÈS
    sys.path.insert(0, str(ROOT_DIR))
    try:
        from spiders.les_echos_news import LesEchosNewsSpider
    except ModuleNotFoundError as e:
        logger.error(
            f"Impossible d'importer le spider : {e}\n"
            "Vérifie que main.py est dans le dossier racine du projet."
        )
        sys.exit(1)

    # ── Lancer le crawler ──────────────────────────────────────────────────
    process = CrawlerProcess(settings)
    crawler = process.create_crawler(LesEchosNewsSpider)
    _state["crawler"] = crawler

    # Signal de fin pour log propre
    def _on_spider_closed(spider, reason):
        logger.info("─" * 60)
        logger.info(f"  Spider fermé : {reason}")
        logger.info(f"  Articles sauvegardés : {_state['items_scraped']}")
        logger.info(f"  Fichier CSV          : {CSV_PATH}")
        logger.info("─" * 60)

    crawler.signals.connect(_on_spider_closed, signal=signals.spider_closed)

    process.crawl(crawler)
    process.start()   # bloquant jusqu'à la fin du crawl


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
