"""
process_amf_extraction_v4_1.py — Pipeline AMF FraFin-Reasoning (Optimisé Ressources)
═══════════════════════════════════════════════════════════════════════════════════════
Corrections des crashes et surchauffe identifiés sur le run v4 :

  [MEM-01] Soumission bornée : traitement par batches de BATCH_SIZE docs
           au lieu de 23 753 futures simultanés → RAM contrôlée
  [CPU-01] MAX_WORKERS réduit à 3 + délai inter-requêtes → CPU < 70%
  [MEM-02] Limite PDF : MAX_PDF_PAGES=12 + MAX_PDF_MB=4 → skip les mastodontes
  [REC-01] Checkpoint JSON → reprise exacte après crash sans tout recommencer
  [MEM-03] gc.collect() + session.close() entre chaque batch → libération mémoire
  [CPU-02] Pause BATCH_COOLDOWN entre batches → CPU peut refroidir

Usage :
    python process_amf_extraction_v4_1.py
    python process_amf_extraction_v4_1.py --workers 3 --batch 30 --min-words 15
    python process_amf_extraction_v4_1.py --resume   # reprend après un crash
"""

import argparse
import csv
import gc
import hashlib
import io
import json
import logging
import re
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
import requests
from dateutil import parser as dateparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("pip install pdfplumber")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_SUPPORT = True
except ImportError:
    PARQUET_SUPPORT = False

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Paramètres ressources (ajuster selon votre machine)
# ═════════════════════════════════════════════════════════════════════════════

EXPORT_FILE = "flux-amf-new-prod.csv"
DATE_FROM   = date(2010, 1, 1)
DATE_TO     = date(2026, 3, 31)

# ── Contrôle CPU / RAM ────────────────────────────────────────────────────────
MAX_WORKERS    = 3      # [CPU-01] 3 au lieu de 8 → CPU reste < 70% sur laptop
BATCH_SIZE     = 30     # [MEM-01] Docs traités avant GC et pause
BATCH_COOLDOWN = 8.0    # [CPU-02] Secondes de pause entre batches (refroidissement)
REQUEST_DELAY  = 0.4    # Délai entre requêtes dans chaque worker

# ── Contrôle taille PDFs ──────────────────────────────────────────────────────
MAX_PDF_PAGES  = 12     # [MEM-02] Pages max par PDF (rapports annuels = 200+ pages)
MAX_PDF_MB     = 4.0    # [MEM-02] Skip les PDFs > 4 Mo (mastodontes)

# ── Paragraphes ───────────────────────────────────────────────────────────────
MIN_WORDS_PER_PARAGRAPH = 15
MAX_WORDS_PER_PARAGRAPH = 350

# ── HTTP ─────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT     = 45
HTTP_RETRIES     = 2
HTTP_BACKOFF     = 1.0
HTTP_RETRY_CODES = {429, 500, 502, 503, 504}

# ═════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ═════════════════════════════════════════════════════════════════════════════

ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RUN_TS          = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT         = DATA_DIR / f"frafin_raw_{RUN_TS}.csv"
PARQUET_OUT     = DATA_DIR / f"frafin_raw_{RUN_TS}.parquet"
META_OUT        = DATA_DIR / f"run_meta_{RUN_TS}.json"
LOG_PATH        = DATA_DIR / f"frafin_processor_{RUN_TS}.log"
CHECKPOINT_FILE = DATA_DIR / "checkpoint_urls_done.json"   # [REC-01]

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# SCHÉMA CSV
# ═════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "paragraph_id", "article_id", "publish_date", "extract_date",
    "headline", "content", "author", "categories", "companies_mentioned",
    "language", "source_url", "doc_type", "emetteur_lei",
    # Réservés Phase 2
    "sentiment_label", "sentiment_score",
    "csrd_category", "materialite_score", "market_surprise", "chain_of_thought",
]

# ═════════════════════════════════════════════════════════════════════════════
# LOCKS & COMPTEURS
# ═════════════════════════════════════════════════════════════════════════════

error_counts: dict[str, int] = defaultdict(int)
error_lock   = Lock()
csv_lock     = Lock()

# ─── Stats globales (thread-safe via Lock) ────────────────────────────────────
_stats_lock        = Lock()
_total_ok          = 0
_total_fail        = 0
_total_paragraphs  = 0


def _update_stats(ok: bool, n_paragraphs: int = 0, err: str = "") -> None:
    global _total_ok, _total_fail, _total_paragraphs
    with _stats_lock:
        if ok:
            _total_ok         += 1
            _total_paragraphs += n_paragraphs
        else:
            _total_fail += 1
    if err:
        with error_lock:
            error_counts[err] += 1

# ═════════════════════════════════════════════════════════════════════════════
# [REC-01] CHECKPOINT — Reprise après crash
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> set[str]:
    """Charge les URLs déjà traitées depuis un run précédent."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = set(data.get("done_urls", []))
        logger.info(f"  ♻  Checkpoint chargé : {len(done):,} URLs déjà traitées")
        return done
    return set()


def save_checkpoint(done_urls: set[str]) -> None:
    """Sauvegarde les URLs traitées après chaque batch."""
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_urls": list(done_urls), "saved_at": datetime.now().isoformat()}, f)
    tmp.replace(CHECKPOINT_FILE)   # écriture atomique


def clear_checkpoint() -> None:
    """Supprime le checkpoint après un run complet réussi."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("  ✅ Checkpoint effacé (run complet)")

# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def _url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _paragraph_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _parse_date(raw) -> date | None:
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    if not raw:
        return None
    try:
        return dateparser.parse(str(raw).strip()).date()
    except Exception:
        return None


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_session() -> requests.Session:
    """Session légère, créée par worker, fermée explicitement après usage."""
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=HTTP_RETRY_CODES,
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=1,   # [MEM-03] 1 connexion par session (pas de pool)
        pool_maxsize=1,
    )
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    s.headers["User-Agent"] = "FraFin-Dataset-Bot/4.1 (Academic Research, ENSAE Paris)"
    return s

# ═════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION PARAGRAPHES PDF (correctif v4 conservé)
# ═════════════════════════════════════════════════════════════════════════════

def _rebuild_pdf_paragraphs(raw_text: str) -> str:
    """
    Reconstruit des blocs sémantiques depuis le texte pdfplumber
    (qui sort ligne par ligne, 3-8 mots/ligne).
    Sans ce traitement : 95% des lignes < 30 mots → tout filtré.
    """
    if not raw_text:
        return ""

    SENTENCE_END = re.compile(r"[.!?;:»]\s*$")
    HEADER_LINE  = re.compile(r"^[A-ZÉÈÊÀÙÎÔÂÛÏË\d\s\-–—/&]{4,}$")

    blocks:  list[str] = []
    current: list[str] = []

    for line in raw_text.split("\n"):
        stripped = line.strip()

        if not stripped:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        wc = len(stripped.split())

        if wc <= 8 and HEADER_LINE.match(stripped):
            if current:
                blocks.append(" ".join(current))
                current = []
            blocks.append(stripped)
            continue

        current.append(stripped)

        if wc <= 15 and SENTENCE_END.search(stripped):
            blocks.append(" ".join(current))
            current = []

    if current:
        blocks.append(" ".join(current))

    return "\n\n".join(b for b in blocks if b.strip())


def split_into_paragraphs(raw_text: str, is_pdf: bool = False) -> list[str]:
    """Découpage en paragraphes sans filtre sémantique (rigueur académique)."""
    if not raw_text:
        return []

    if is_pdf:
        raw_text = _rebuild_pdf_paragraphs(raw_text)

    raw_pars = []
    for block in re.split(r"\n{2,}", raw_text):
        block = block.strip()
        if block:
            raw_pars.append(block)

    if not raw_pars:
        raw_pars = [p.strip() for p in raw_text.split("\n") if p.strip()]

    valid: list[str] = []

    for p in raw_pars:
        p  = re.sub(r" {2,}", " ", p)
        wc = len(p.split())

        if MIN_WORDS_PER_PARAGRAPH <= wc <= MAX_WORDS_PER_PARAGRAPH:
            valid.append(p)

        elif wc > MAX_WORDS_PER_PARAGRAPH:
            sub  = re.split(r"(?<=[.!?])\s+", p)
            cur  = []
            cnt  = 0
            for part in sub:
                pc = len(part.split())
                if cnt + pc <= MAX_WORDS_PER_PARAGRAPH:
                    cur.append(part)
                    cnt += pc
                else:
                    chunk = " ".join(cur).strip()
                    if len(chunk.split()) >= MIN_WORDS_PER_PARAGRAPH:
                        valid.append(chunk)
                    cur, cnt = [part], pc
            if cur:
                chunk = " ".join(cur).strip()
                if len(chunk.split()) >= MIN_WORDS_PER_PARAGRAPH:
                    valid.append(chunk)

    return valid

# ═════════════════════════════════════════════════════════════════════════════
# MOTEURS D'EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _extract_pdf(pdf_bytes: bytes) -> tuple[str | None, str | None]:
    """
    [MEM-02] Limite à MAX_PDF_PAGES pages.
    Les rapports annuels AMF font souvent 150-200 pages → pdfplumber
    consomme des centaines de Mo si on les parse en entier.
    """
    if not PDF_SUPPORT:
        return None, "no_pdf_support"

    # [MEM-02] Skip les PDFs trop gros
    size_mb = len(pdf_bytes) / 1_048_576
    if size_mb > MAX_PDF_MB:
        return None, f"pdf_too_large:{size_mb:.1f}MB"

    try:
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n_pages = len(pdf.pages)
            pages_to_read = min(n_pages, MAX_PDF_PAGES)
            for page in pdf.pages[:pages_to_read]:
                t = page.extract_text()
                if t and t.strip():
                    parts.append(t.strip())

        text = "\n\n".join(parts)   # double newline pour le rebuilder
        if len(text) > 50:
            return text, None
        return None, "pdf_empty"

    except Exception as e:
        return None, f"pdf_parse_error:{type(e).__name__}"


def _fetch_text(
    url: str,
    session: requests.Session,
) -> tuple[str | None, str | None, bool]:
    """Retourne (texte, code_erreur, is_pdf)."""
    if not url:
        return None, "empty_url", False
    try:
        # Streaming HEAD pour vérifier la taille avant de tout télécharger
        head = session.head(url, timeout=10, allow_redirects=True)
        content_length = int(head.headers.get("content-length", 0))
        if content_length > MAX_PDF_MB * 1_048_576 * 1.2:
            # Taille annoncée trop grande → skip sans télécharger
            size_mb = content_length / 1_048_576
            return None, f"pdf_too_large:{size_mb:.1f}MB", False

        r = session.get(str(url), timeout=HTTP_TIMEOUT)

        if r.status_code == 404:
            return None, "http_404", False
        if 400 <= r.status_code < 500:
            return None, f"http_{r.status_code}", False
        if 500 <= r.status_code < 600:
            return None, f"http_{r.status_code}", False

        ct     = r.headers.get("content-type", "").lower()
        is_pdf = "pdf" in ct or url.lower().endswith(".pdf")

        if is_pdf:
            text, err = _extract_pdf(r.content)
            # [MEM-03] Libérer le contenu brut immédiatement
            del r
            return text, err, True

        # HTML
        if BS4_AVAILABLE:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "noscript"]):
                tag.decompose()
            for block in soup.find_all(["p","div","h1","h2","h3","h4","li"]):
                block.append("\n")
            text = soup.get_text(separator=" ").strip()
            del soup
        else:
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text).strip()

        del r
        return (text, None, False) if len(text) > 100 else (None, "html_empty", False)

    except requests.exceptions.Timeout:
        return None, "timeout", False
    except requests.exceptions.ConnectionError:
        return None, "connection_error", False
    except Exception as e:
        return None, f"fetch_error:{type(e).__name__}", False

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & DÉTECTION COLONNES
# ═════════════════════════════════════════════════════════════════════════════

def load_and_analyze(filepath: str) -> tuple[pd.DataFrame, Path]:
    path = Path(filepath)
    if not path.exists():
        candidates = sorted(
            list(Path(".").glob("*.csv")) + list(Path(".").glob("*.xlsx")),
            key=lambda p: p.stat().st_size, reverse=True,
        )
        path = candidates[0] if candidates else None
        if not path:
            logger.error("Aucun fichier CSV/XLSX trouvé.")
            sys.exit(1)

    logger.info(f"Chargement : {path} ({path.stat().st_size / 1_048_576:.1f} Mo)")
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        df = None
        for sep in [";", ",", "\t"]:
            for enc in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    df = pd.read_csv(path, sep=sep, dtype=str, encoding=enc,
                                     low_memory=False, on_bad_lines="warn")
                    if len(df.columns) > 5:
                        logger.info(f"  Encodage : {enc} | Séparateur : '{sep}'")
                        break
                except Exception:
                    continue
            if df is not None and len(df.columns) > 5:
                break

    logger.info(f"  Lignes : {len(df):,}  |  Colonnes : {len(df.columns)}")
    return df, path


def detect_columns(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    def find(*keywords) -> str | None:
        for kw in keywords:
            kw_n = kw.lower().replace(" ", "_").replace("'", "_").replace("'","_")
            for cl, corig in cols_lower.items():
                if kw_n in cl:
                    return corig
        return None

    mapping = {
        "cac40":    find("cac40", "cac_40"),
        "name_cac": find("name_cac40", "name_cac"),
        "societe":  find("société", "societe", "emetteur_lib", "emetteur"),
        "isin":     find("isin"),
        "date":     find("date_envoi_amf", "date_de_ti", "date_tit", "date_dep"),
        "titre":    find("titre_du_fichier", "titre", "title"),
        # [FIX-02] Sous-type en priorité (52 656 valeurs vs 1 957)
        "type_doc": find("sous-type_d", "sous_type_d",
                         "type_d_information_nouveau",
                         "type_d_information_ancien"),
        "langue":   find("langue", "language"),
        "url":      find("url_de_recuperation", "url_recup", "url_doc", "url"),
        "lei":      find("identificationsociete_iso_cd_lei", "lei"),
    }

    logger.info("\n  Mapping colonnes :")
    for k, v in mapping.items():
        status = "✅" if v else "❌"
        suffix = ""
        if k == "type_doc" and v:
            n = df[v].notna().sum()
            suffix = f"  ({n:,} valeurs)"
            if n < 40_000:
                suffix += "  ⚠ FAIBLE — vérifier"
        logger.info(f"    {status}  {k:<15} → {v or 'NON TROUVÉ'}{suffix}")

    missing = [k for k in ("url", "date", "titre") if not mapping[k]]
    if missing:
        logger.error(f"Colonnes critiques manquantes : {missing}")
        sys.exit(1)

    return mapping

# ═════════════════════════════════════════════════════════════════════════════
# FILTRAGE MÉTADONNÉES
# ═════════════════════════════════════════════════════════════════════════════

def filter_metadata(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    logger.info(f"\n{'═'*65}")
    logger.info(f"  FILTRAGE   ({len(df):,} lignes initiales)")
    logger.info(f"{'═'*65}")

    if mapping["cac40"]:
        col = mapping["cac40"]
        df  = df[df[col].str.strip().str.lower().isin(
            {"1", "true", "oui", "yes", "vrai", "x"}
        )].copy()
        logger.info(f"  ✔ CAC40   : {len(df):,}")

    if mapping["langue"] and len(df) > 0:
        col  = mapping["langue"]
        df_fr = df[df[col].str.strip().str.lower().isin(
            {"fr", "français", "francais", "french", "fre"}
        )]
        if len(df_fr) > 0:
            df = df_fr.copy()
            logger.info(f"  ✔ Langue  : {len(df):,}")

    if mapping["date"] and len(df) > 0:
        df         = df.copy()
        df["_date"] = df[mapping["date"]].apply(_parse_date)
        before      = len(df)
        df          = df[df["_date"].apply(
            lambda d: d is not None and DATE_FROM <= d <= DATE_TO
        )].copy()
        logger.info(f"  ✔ Dates   : {len(df):,}  (rejetés : {before-len(df):,})")
        df["_year"] = df["_date"].apply(lambda d: d.year if d else None)
        for yr, n in df["_year"].value_counts().sort_index().items():
            bar = "█" * max(1, n // max(1, len(df) // 40))
            logger.info(f"    {yr}  {bar:<42} {n:>5,}")

    if mapping["url"] and len(df) > 0:
        col = mapping["url"]
        df  = df[df[col].notna() & (df[col].str.strip() != "")].copy()
        logger.info(f"  ✔ URL     : {len(df):,}")

    if mapping["type_doc"] and len(df) > 0:
        col = mapping["type_doc"]
        logger.info("  Top types de documents :")
        for t, n in df[col].value_counts().head(15).items():
            logger.info(f"    {str(t):<60} {n:>5,}")

    logger.info(f"\n  ✅ Total : {len(df):,} documents CAC 40 FR\n")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# WORKER
# ═════════════════════════════════════════════════════════════════════════════

def pipeline_worker(row_data: dict, mapping: dict) -> tuple[list[dict] | None, str]:
    """
    Worker exécuté dans le pool. Crée et ferme sa propre session ([MEM-03]).
    Délai REQUEST_DELAY pour ne pas saturer le serveur ni le CPU.
    """
    session = make_session()
    try:
        time.sleep(REQUEST_DELAY)   # [CPU-02] étalement des requêtes

        url      = str(row_data.get(mapping["url"] or "",     "") or "").strip()
        titre    = str(row_data.get(mapping["titre"] or "",   "") or "").strip()
        societe  = str(row_data.get(mapping["societe"] or "", "") or "").strip()
        type_doc = str(row_data.get(mapping["type_doc"] or "", "") or "").strip() \
                   if mapping["type_doc"] else ""
        isin     = str(row_data.get(mapping["isin"] or "", "") or "").strip() \
                   if mapping["isin"] else ""
        lei      = str(row_data.get(mapping["lei"] or "",  "") or "").strip() \
                   if mapping["lei"] else ""
        name_cac = str(row_data.get(mapping["name_cac"] or "", "") or societe).strip() \
                   if mapping["name_cac"] else societe

        d          = row_data.get("_date")
        pub_iso    = d.isoformat() if d and not isinstance(d, float) else None
        article_id = _url_id(url)

        raw_text, err_code, is_pdf = _fetch_text(url, session)
        if err_code:
            return None, err_code

        chunks = split_into_paragraphs(raw_text, is_pdf=is_pdf)

        # [MEM-03] Libérer le texte brut
        del raw_text

        if not chunks:
            return None, "no_valid_paragraphs"

        now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rows: list[dict] = []

        for chunk in chunks:
            rows.append({
                "paragraph_id":        _paragraph_id(chunk),
                "article_id":          article_id,
                "publish_date":        pub_iso,
                "extract_date":        now_utc,
                "headline":            titre,
                "content":             chunk,
                "author":              societe,
                "categories":          json.dumps([type_doc], ensure_ascii=False),
                "companies_mentioned": json.dumps(
                    [{"name": name_cac, "isin": isin}], ensure_ascii=False
                ),
                "language":            "fr",
                "source_url":          url,
                "doc_type":            type_doc,
                "emetteur_lei":        lei,
                "sentiment_label":     None,
                "sentiment_score":     None,
                "csrd_category":       None,
                "materialite_score":   None,
                "market_surprise":     None,
                "chain_of_thought":    None,
            })

        return rows, "ok"

    finally:
        session.close()   # [MEM-03] Fermeture explicite de la session

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — Traitement par batches bornés
# ═════════════════════════════════════════════════════════════════════════════

def process_and_save(
    df: pd.DataFrame,
    mapping: dict,
    done_urls: set[str],   # [REC-01] URLs déjà traitées
    csv_file,              # fichier CSV ouvert (append si reprise)
    writer,
) -> dict:
    """
    [MEM-01] Traitement par batches de BATCH_SIZE documents.
    Entre chaque batch : GC, sauvegarde checkpoint, pause CPU.
    """
    global _total_ok, _total_fail, _total_paragraphs

    url_col = mapping["url"]

    # Déduplication + exclusion des URLs déjà traitées (checkpoint)
    seen_urls: set[str] = set(done_urls)
    unique_rows: list[dict] = []
    for _, row in df.iterrows():
        url = str(row.get(url_col, "") or "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_rows.append(row.to_dict())

    total_docs = len(unique_rows)
    logger.info(f"  Documents à traiter : {total_docs:,}  "
                f"(checkpoint : {len(done_urls):,} déjà traités, "
                f"doublons : {len(df) - total_docs - len(done_urls):,})")

    # ── Traitement par batches ────────────────────────────────────────────────
    for batch_start in range(0, total_docs, BATCH_SIZE):
        batch      = unique_rows[batch_start : batch_start + BATCH_SIZE]
        batch_num  = batch_start // BATCH_SIZE + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        processed  = batch_start + len(batch)
        pct        = processed / total_docs * 100

        logger.info(
            f"\n  ── Batch {batch_num}/{total_batches} "
            f"[{batch_start+1}–{processed}/{total_docs}]  {pct:.1f}%  "
            f"| ok={_total_ok}  fail={_total_fail}  §={_total_paragraphs:,}"
        )

        # ── Exécution du batch dans le pool ──────────────────────────────────
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(pipeline_worker, rd, mapping): rd.get(url_col, "")
                for rd in batch
            }

            for future in as_completed(futures):
                url_done = futures[future]
                try:
                    result, status = future.result()
                except Exception as e:
                    result, status = None, f"exception:{type(e).__name__}"

                if result:
                    with csv_lock:
                        for p_row in result:
                            writer.writerow(p_row)
                        csv_file.flush()
                    _update_stats(ok=True, n_paragraphs=len(result))
                    done_urls.add(url_done)
                else:
                    _update_stats(ok=False, err=status)
                    done_urls.add(url_done)   # marquer quand même pour éviter retry

        # ── [REC-01] Checkpoint après chaque batch ────────────────────────────
        save_checkpoint(done_urls)

        # ── [MEM-03] Libération mémoire entre batches ─────────────────────────
        gc.collect()

        # ── [CPU-02] Pause refroidissement (sauf dernier batch) ───────────────
        if batch_start + BATCH_SIZE < total_docs:
            logger.info(f"  ⏸  Pause {BATCH_COOLDOWN}s (refroidissement CPU)...")
            time.sleep(BATCH_COOLDOWN)

    return {
        "total_docs_submitted": total_docs + len(done_urls),
        "total_docs_ok":        _total_ok,
        "total_docs_fail":      _total_fail,
        "total_paragraphs":     _total_paragraphs,
        "coverage_rate_pct":    round(_total_ok / max(1, total_docs) * 100, 2),
        "error_breakdown":      dict(error_counts),
    }

# ═════════════════════════════════════════════════════════════════════════════
# EXPORT PARQUET
# ═════════════════════════════════════════════════════════════════════════════

def export_parquet(csv_path: Path) -> None:
    if not PARQUET_SUPPORT:
        logger.warning("  ⚠ Parquet ignoré (pip install pyarrow)")
        return
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        df["publish_date"]      = pd.to_datetime(df["publish_date"], errors="coerce").dt.date
        df["extract_date"]      = pd.to_datetime(df["extract_date"], errors="coerce", utc=True)
        df["materialite_score"] = pd.to_numeric(df["materialite_score"], errors="coerce")
        for col in ("categories", "companies_mentioned"):
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else None)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, PARQUET_OUT, compression="snappy")
        logger.info(f"  ✅ Parquet : {PARQUET_OUT}")
    except Exception as e:
        logger.error(f"  ✗ Parquet : {e}")

# ═════════════════════════════════════════════════════════════════════════════
# MÉTADONNÉES RUN
# ═════════════════════════════════════════════════════════════════════════════

def save_run_meta(source_path, source_hash, df_filtered, mapping, stats):
    doc_dist  = {}
    year_dist = {}
    if mapping["type_doc"] and mapping["type_doc"] in df_filtered.columns:
        doc_dist = df_filtered[mapping["type_doc"]].value_counts().to_dict()
    if "_year" in df_filtered.columns:
        year_dist = df_filtered["_year"].value_counts().sort_index().to_dict()

    meta = {
        "run_timestamp":    RUN_TS,
        "script_version":   "4.1",
        "source_file":      str(source_path.resolve()),
        "source_sha256":    source_hash,
        "date_range_from":  DATE_FROM.isoformat(),
        "date_range_to":    DATE_TO.isoformat(),
        "min_words":        MIN_WORDS_PER_PARAGRAPH,
        "max_words":        MAX_WORDS_PER_PARAGRAPH,
        "max_workers":      MAX_WORKERS,
        "batch_size":       BATCH_SIZE,
        "max_pdf_pages":    MAX_PDF_PAGES,
        "max_pdf_mb":       MAX_PDF_MB,
        **stats,
        "distribution_by_doc_type": {str(k): int(v) for k, v in doc_dist.items()},
        "distribution_by_year":     {str(k): int(v) for k, v in year_dist.items()},
        "output_csv":       str(CSV_OUT),
        "output_parquet":   str(PARQUET_OUT) if PARQUET_SUPPORT else None,
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✅ run_meta : {META_OUT}")


def log_final_report(stats):
    logger.info(f"\n{'═'*65}")
    logger.info(f"  RAPPORT FINAL v4.1")
    logger.info(f"{'═'*65}")
    logger.info(f"  Documents OK        : {stats['total_docs_ok']:>10,}")
    logger.info(f"  Documents en échec  : {stats['total_docs_fail']:>10,}")
    logger.info(f"  Taux de couverture  : {stats['coverage_rate_pct']:>9.2f}%")
    logger.info(f"  Paragraphes bruts   : {stats['total_paragraphs']:>10,}")
    logger.info(f"  CSV                 : {CSV_OUT}")
    logger.info(f"\n  Détail des échecs :")
    for err, n in sorted(stats["error_breakdown"].items(), key=lambda x: -x[1]):
        logger.info(f"    {err:<45} {n:>6,}")
    logger.info(f"\n  ⚙  Prochaine étape : build_annotation_sample.py (Phase 2)")
    logger.info(f"{'═'*65}")

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL CTRL+C — Sauvegarde checkpoint propre
# ═════════════════════════════════════════════════════════════════════════════

_done_urls_ref: set[str] = set()

def _sigint_handler(sig, frame):
    logger.warning(f"\n⚡ CTRL+C — Sauvegarde checkpoint...")
    save_checkpoint(_done_urls_ref)
    logger.warning(f"   CSV partiel : {CSV_OUT}")
    logger.warning(f"   Relancez avec --resume pour continuer.")
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="FraFin-Reasoning AMF Pipeline v4.1")
    p.add_argument("--input",     default=EXPORT_FILE)
    p.add_argument("--workers",   type=int,   default=MAX_WORKERS,
                   help=f"Workers parallèles (défaut {MAX_WORKERS}, max recommandé 4)")
    p.add_argument("--batch",     type=int,   default=BATCH_SIZE,
                   help=f"Taille des batches (défaut {BATCH_SIZE})")
    p.add_argument("--min-words", type=int,   default=MIN_WORDS_PER_PARAGRAPH)
    p.add_argument("--max-words", type=int,   default=MAX_WORDS_PER_PARAGRAPH)
    p.add_argument("--resume",    action="store_true",
                   help="Reprendre après un crash (utilise le checkpoint)")
    return p.parse_args()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global MAX_WORKERS, BATCH_SIZE, MIN_WORDS_PER_PARAGRAPH, MAX_WORDS_PER_PARAGRAPH
    global _done_urls_ref

    args = parse_args()
    MAX_WORKERS             = args.workers
    BATCH_SIZE              = args.batch
    MIN_WORDS_PER_PARAGRAPH = args.min_words
    MAX_WORDS_PER_PARAGRAPH = args.max_words

    logger.info("═" * 65)
    logger.info("  FraFin-Reasoning — AMF Pipeline v4.1 (Optimisé Ressources)")
    logger.info(f"  Workers : {MAX_WORKERS} | Batch : {BATCH_SIZE} "
                f"| Cooldown : {BATCH_COOLDOWN}s | PDF max : {MAX_PDF_PAGES}p/{MAX_PDF_MB}Mo")
    logger.info(f"  Paragraphes : {MIN_WORDS_PER_PARAGRAPH}–{MAX_WORDS_PER_PARAGRAPH} mots")
    logger.info("═" * 65)

    # [REC-01] Chargement du checkpoint si --resume
    done_urls: set[str] = set()
    if args.resume:
        done_urls = load_checkpoint()
        if not done_urls:
            logger.info("  Aucun checkpoint trouvé — run complet.")

    _done_urls_ref = done_urls   # référence pour SIGINT

    # Chargement source
    df, source_path = load_and_analyze(args.input)
    logger.info(f"\n  Hash SHA256 source...")
    source_hash = compute_file_hash(source_path)
    logger.info(f"  {source_hash}")

    mapping     = detect_columns(df)
    df_filtered = filter_metadata(df, mapping)

    if len(df_filtered) == 0:
        logger.error("Aucune ligne après filtrage.")
        sys.exit(1)

    # Ouverture CSV (append si --resume, sinon création)
    mode = "a" if args.resume and CSV_OUT.exists() else "w"
    if mode == "a":
        # Trouver le CSV le plus récent du répertoire data/
        existing = sorted(DATA_DIR.glob("frafin_raw_*.csv"), reverse=True)
        if existing:
            csv_target = existing[0]
            logger.info(f"  ♻  Reprise sur : {csv_target}")
        else:
            csv_target = CSV_OUT
            mode = "w"
    else:
        csv_target = CSV_OUT

    logger.info(f"\n  [4/5] Extraction par batches de {BATCH_SIZE} docs "
                f"({MAX_WORKERS} workers)...")

    with open(csv_target, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        stats = process_and_save(df_filtered, mapping, done_urls, f, writer)

    logger.info(f"\n  [5/5] Export Parquet...")
    export_parquet(csv_target)

    save_run_meta(source_path, source_hash, df_filtered, mapping, stats)
    log_final_report(stats)

    # Run complet → supprimer le checkpoint
    clear_checkpoint()


if __name__ == "__main__":
    main()