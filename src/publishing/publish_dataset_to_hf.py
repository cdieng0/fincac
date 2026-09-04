"""
publish_to_hf.py — Prépare et publie FReCAP sur le Hugging Face Hub
═══════════════════════════════════════════════════════════════════════════════════════
Convertit vos fichiers locaux au format attendu par le Dataset Viewer, vérifie
l'intégrité, puis pousse sur le Hub.

⚠️  Le script fait une VÉRIFICATION À BLANC par défaut (--dry-run implicite).
    Rien n'est publié tant que vous n'ajoutez pas --push.

Prérequis :
    pip install datasets huggingface_hub pandas pyarrow openpyxl
    huggingface-cli login

Usage :
    # 1. Vérification sans rien publier (recommandé en premier)
    python publish_to_hf.py --corpus data/frafin_raw_XXXX.parquet \\
                            --gold data/gold_150_annotated_clean_reformulated_without.xlsx \\
                            --repo VOTRE_USERNAME/FReCAP

    # 2. Publication réelle
    python publish_to_hf.py --corpus ... --gold ... --repo ... --push

    # Publier en privé d'abord (fortement conseillé pour une première fois)
    python publish_to_hf.py --corpus ... --gold ... --repo ... --push --private
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

EPOCH_ORDER = ["2010-2014", "2015-2019", "2020-2022", "2023-2026"]

# Colonnes finales publiées, dans l'ordre. Tout le reste est écarté :
# les colonnes internes (scores de stratification, quartiles) ne sont pas
# publiées car elles pourraient induire un biais d'ancrage chez de futurs
# annotateurs (cf. protocole d'annotation à l'aveugle du papier).
CORPUS_COLS = ["row_id", "paragraph_id", "document_id", "content", "date_envoi",
               "emetteur", "type_document", "epoch"]
GOLD_COLS = CORPUS_COLS + ["csrd_category", "esrs_subcategory", "chain_of_thought"]


def derive_epoch(year) -> str:
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return "unknown"
    if y <= 2014: return "2010-2014"
    if y <= 2019: return "2015-2019"
    if y <= 2022: return "2020-2022"
    return "2023-2026"


def normalise(df: pd.DataFrame, is_gold: bool) -> pd.DataFrame:
    """Renomme les colonnes hétérogènes des fichiers sources vers le schéma publié."""
    ren = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("content", "content — extrait à annoter") or "extrait à annoter" in cl:
            ren[c] = "content"
        elif cl in ("article_id", "document_id"):
            ren[c] = "document_id"
        elif cl in ("publish_date", "date_envoi", "date envoi amf"):
            ren[c] = "date_envoi"
        elif cl in ("author", "societe", "société", "emetteur"):
            ren[c] = "emetteur"
        elif cl in ("doc_type", "type_document", "type de document"):
            ren[c] = "type_document"
        elif cl in ("year_bucket", "période", "periode", "epoch"):
            ren[c] = "epoch"
        elif cl in ("chain_of_thought", "chain of thought", "raisonnement",
                    "chain_of_thought_human"):
            ren[c] = "chain_of_thought"
    df = df.rename(columns=ren)

    # Deux colonnes sources peuvent aboutir au même nom cible (p. ex. un fichier
    # Gold contenant à la fois `content` et `content — Extrait à annoter`).
    # On fusionne alors les homonymes en gardant, colonne par colonne, la
    # variante la mieux remplie — sinon df["content"] renverrait un DataFrame
    # et non une Series, ce qui casse tous les traitements en aval.
    if df.columns.duplicated().any():
        for name in df.columns[df.columns.duplicated()].unique():
            block = df.loc[:, df.columns == name]
            best = block.iloc[:, block.notna().sum().values.argmax()]
            df = df.loc[:, df.columns != name]
            df[name] = best
            print(f"  ℹ  Colonnes homonymes « {name} » fusionnées "
                  f"({block.shape[1]} variantes)")

    if "epoch" not in df.columns:
        year_src = next((c for c in ("_year", "année", "annee") if c in df.columns), None)
        if year_src:
            df["epoch"] = df[year_src].apply(derive_epoch)
        elif "date_envoi" in df.columns:
            df["epoch"] = pd.to_datetime(df["date_envoi"], errors="coerce").dt.year.apply(derive_epoch)

    wanted = GOLD_COLS if is_gold else CORPUS_COLS
    for c in wanted:
        if c not in df.columns:
            df[c] = None
    return df[wanted]


def clean(df: pd.DataFrame, dedupe: bool, name: str) -> pd.DataFrame:
    """
    Nettoyage minimal avant publication.

    `paragraph_id` est un hash SHA256 du CONTENU (pipeline d extraction) : deux
    paragraphes au texte strictement identique partagent donc le meme id. Ce
    n est pas une anomalie mais une propriete du corpus reglementaire, sature de
    boilerplate juridique repete entre documents et entre annees. On ajoute donc
    un `row_id` reellement unique et on documente `paragraph_id` comme hash de
    contenu, plutot que de supprimer des lignes legitimes.
    """
    n0 = len(df)

    empty = df["content"].isna() | (df["content"].astype(str).str.strip() == "")
    if empty.any():
        df = df[~empty].copy()
        print(f"  [{name}] {int(empty.sum())} ligne(s) au contenu vide supprimee(s)")

    n_dup_content = int(df["paragraph_id"].duplicated().sum())
    if n_dup_content:
        pct = n_dup_content / max(1, len(df)) * 100
        if dedupe:
            df = df.drop_duplicates(subset=["paragraph_id"], keep="first").copy()
            print(f"  [{name}] --dedupe : {n_dup_content} doublon(s) de contenu "
                  f"supprime(s) ({pct:.1f} %)")
        else:
            print(f"  [{name}] {n_dup_content} paragraphe(s) au contenu repete "
                  f"conserve(s) ({pct:.1f} %) — boilerplate reglementaire, "
                  f"voir --dedupe pour les retirer")

    df = df.reset_index(drop=True)
    # normalise() a pu pre-creer une colonne row_id vide (elle figure dans le
    # schema cible) : on la retire avant d inserer les identifiants reels.
    if "row_id" in df.columns:
        df = df.drop(columns=["row_id"])
    df.insert(0, "row_id", range(len(df)))
    print(f"  [{name}] {n0} -> {len(df)} lignes publiees")
    return df


def check(df: pd.DataFrame, name: str, is_gold: bool) -> bool:
    print(f"\n{'─'*62}\n  {name}\n{'─'*62}")
    print(f"  Lignes : {len(df):,}")
    ok = True

    n_empty = int(df["content"].isna().sum()
                  + (df["content"].astype(str).str.strip() == "").sum())
    if n_empty:
        print(f"  ⚠  {n_empty} paragraphe(s) au contenu vide")
        ok = False

    # L unicite se verifie sur row_id : paragraph_id est un hash de contenu,
    # legitimement repete pour le boilerplate (voir clean()).
    if "row_id" in df.columns and int(df["row_id"].duplicated().sum()):
        print(f"  ⚠  row_id dupliqué — identifiant de ligne non unique")
        ok = False

    n_dup_c = int(df["paragraph_id"].duplicated().sum())
    if n_dup_c:
        print(f"  ℹ  {n_dup_c} paragraphe(s) au contenu répété "
              f"({n_dup_c/max(1,len(df))*100:.1f} %) — normal (boilerplate)")

    if "epoch" in df.columns:
        print("  Répartition par époque :")
        for ep in EPOCH_ORDER:
            print(f"    {ep} : {(df['epoch'] == ep).sum():,}")
        n_unknown = (~df["epoch"].isin(EPOCH_ORDER)).sum()
        if n_unknown:
            print(f"  ⚠  {n_unknown} ligne(s) hors des quatre époques")
            ok = False

    if is_gold and "chain_of_thought" in df.columns:
        cot = df["chain_of_thought"].astype(str).str.strip()
        n_filled = int((cot.notna() & (cot != "") & (cot != "nan")).sum())
        lens = cot[cot.str.len() > 3].str.len()
        print(f"  Raisonnements experts : {n_filled}/{len(df)} renseignés"
              + (f", longueur moyenne {lens.mean():.0f} caractères" if len(lens) else ""))
        if n_filled < len(df):
            print(f"  ⚠  {len(df) - n_filled} annotation(s) sans raisonnement")

    if is_gold and "csrd_category" in df.columns:
        print("  Répartition des catégories :")
        for cat, n in df["csrd_category"].value_counts().items():
            print(f"    {cat} : {n}")

    print(f"  {'✅ Prêt' if ok else '❌ Problèmes détectés — corrigez avant publication'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Publie FINCAC40 sur le Hugging Face Hub")
    p.add_argument("--corpus", required=True, help="Parquet/CSV du corpus complet")
    p.add_argument("--gold", required=True, help="XLSX/CSV du Gold Standard annoté")
    p.add_argument("--repo", required=True, help="ex. votre-username/FINCAC")
    p.add_argument("--readme", default="README.md", help="Dataset card à téléverser")
    p.add_argument("--out", default="hf_build", help="Dossier de préparation local")
    p.add_argument("--push", action="store_true", help="Publie réellement (sinon : à blanc)")
    p.add_argument("--private", action="store_true", help="Crée le dépôt en privé")
    p.add_argument("--dedupe", action="store_true",
                   help="Supprime les paragraphes au contenu strictement identique "
                        "(~18 %% du corpus : boilerplate juridique). Par défaut ils sont "
                        "conservés, car un même texte à deux dates différentes est une "
                        "information réelle pour une étude temporelle.")
    args = p.parse_args()

    # ── Chargement ────────────────────────────────────────────────────────
    cp = Path(args.corpus)
    corpus = pd.read_parquet(cp) if cp.suffix == ".parquet" else pd.read_csv(cp, low_memory=False)
    gp = Path(args.gold)
    gold = pd.read_excel(gp) if gp.suffix in (".xlsx", ".xls") else pd.read_csv(gp)

    corpus = normalise(corpus, is_gold=False)
    gold = normalise(gold, is_gold=True)

    corpus = clean(corpus, args.dedupe, "corpus")
    gold = clean(gold, False, "gold")   # jamais de dedupe sur le Gold annote

    ok = check(corpus, "CORPUS", False) & check(gold, "GOLD STANDARD", True)

    # ── Exclusion du Gold hors du corpus publié ──────────────────────────
    # Le Gold est publié comme split d'évaluation distinct ; le laisser aussi
    # dans le corpus d'entraînement créerait une fuite pour tout utilisateur
    # qui entraînerait sur `corpus` puis évaluerait sur `gold`.
    before = len(corpus)
    gold_ids = set(gold["paragraph_id"].astype(str))
    corpus = corpus[~corpus["paragraph_id"].astype(str).isin(gold_ids)]
    print(f"\n  Gold retiré du corpus publié : {before - len(corpus)} ligne(s) "
          f"→ {len(corpus):,} paragraphes dans `corpus`")

    # ── Écriture locale ───────────────────────────────────────────────────
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "corpus").mkdir(parents=True)
    (out / "gold").mkdir(parents=True)
    corpus.to_parquet(out / "corpus" / "train.parquet", index=False)
    gold.to_parquet(out / "gold" / "test.parquet", index=False)

    readme = Path(args.readme)
    if readme.exists():
        shutil.copy(readme, out / "README.md")
        print(f"  Dataset card copiée : {readme}")
    else:
        print(f"  ⚠  {readme} introuvable — le dépôt n'aura pas de carte descriptive.")
        ok = False

    print(f"\n  Arborescence prête dans « {out}/ » :")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out)}  ({f.stat().st_size/1_048_576:.1f} Mo)")

    # ── Publication ───────────────────────────────────────────────────────
    if not args.push:
        print(f"\n{'═'*62}")
        print("  VÉRIFICATION À BLANC — rien n'a été publié.")
        print("  Relancez avec --push (et --private pour un premier essai).")
        print(f"{'═'*62}")
        return

    if not ok:
        print("\n❌ Des problèmes ont été détectés ci-dessus. Publication annulée.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset")
    print(f"\n✅ Publié : https://huggingface.co/datasets/{args.repo}")
    if args.private:
        print("   (dépôt privé — rendez-le public depuis Settings quand vous êtes prêt)")


if __name__ == "__main__":
    main()
