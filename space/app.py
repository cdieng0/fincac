"""
app.py — Espace de démonstration : effondrement d'une optimisation de préférence
═══════════════════════════════════════════════════════════════════════════════════════
Compare côte à côte le modèle de base et le modèle ORPO sur le même extrait.
L'utilisateur voit l'effondrement se produire plutôt que d'en lire la description.

Déploiement sur Hugging Face Spaces :
  1. Créer un Space (SDK: Gradio, Hardware: ZeroGPU si disponible, sinon T4)
  2. Y déposer ce fichier sous le nom app.py, plus requirements.txt
  3. Le Space se construit et démarre automatiquement

Si aucun GPU n'est disponible, l'app bascule automatiquement en mode « exemples
pré-calculés » : elle affiche les prédictions réelles enregistrées lors de l'évaluation
sur les 140 paragraphes du Gold Standard. La démonstration reste honnête et fonctionne.
"""

import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# HF Spaces expose SPACE_ID au format "utilisateur/nom-du-space". On en deduit
# l'identifiant du compte, ce qui evite tout placeholder a remplacer a la main
# (et donc tout lien mort si on oublie de le faire).
HF_USER = os.environ.get("SPACE_ID", "").split("/")[0] or "CID99"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", f"{HF_USER}/Mistral-7B-ORPO-CSRD")
PRECOMPUTED = Path("eval_predictions.csv")

SYSTEM_PROMPT = """\
Tu es un auditeur ESG senior, expert en réglementation européenne CSRD
et en normes ESRS, spécialisé dans l'analyse de documents réglementaires
AMF français.

Ta mission : classifier l'extrait ci-dessous selon la taxonomie CSRD/ESRS.

CONSIGNES STRICTES :
1. Détermine si le texte contient un enjeu de durabilité.
2. Si aucun enjeu -> csrd_category = "none".
3. Si un enjeu est présent -> identifie la catégorie ESRS principale.
4. Le raisonnement doit suivre 3 étapes : (1) nature de l'information,
   (2) analyse de matérialité ESG, (3) surprise pour les investisseurs.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases structurées>"
}
"""

FALLBACK_EXAMPLES = [
    ["La société annonce avoir procédé au rachat de 125 000 actions propres au cours du mois, "
     "dans le cadre du programme autorisé par l'assemblée générale, à un prix moyen de "
     "42,17 euros par action."],
    ["Le groupe a réduit ses émissions de gaz à effet de serre Scope 1 et 2 de 18 % sur "
     "l'exercice, dépassant l'objectif annuel de 12 % fixé dans son plan de transition "
     "climatique."],
]


def load_examples() -> list[list[str]]:
    """
    En mode hors-ligne, les exemples DOIVENT provenir du CSV d evaluation : sinon
    l utilisateur clique sur un exemple absent du jeu evalue et n obtient aucun
    resultat, ce qui donne l impression d une demo cassee.

    On selectionne en priorite les cas qui illustrent l effondrement :
      1. JSON invalide produit par le modele ORPO
      2. Faux positifs (annotation = none, mais ORPO predit une categorie CSRD)
      3. Cas ou les deux s accordent, pour le contraste
    """
    if not PRECOMPUTED.exists():
        return FALLBACK_EXAMPLES
    try:
        df = pd.read_csv(PRECOMPUTED)
    except Exception:
        return FALLBACK_EXAMPLES

    df = df[df["content"].astype(str).str.len().between(120, 700)]
    picked, seen = [], set()

    def take(subset, n):
        for _, r in subset.head(n).iterrows():
            t = str(r["content"]).strip()
            if t and t not in seen:
                seen.add(t)
                picked.append([t])

    gold_none = df["csrd_category"].astype(str).str.strip() == "none"
    pred_na = df["pred_category"].isna()
    pred_csrd = (~pred_na) & (df["pred_category"].astype(str).str.strip() != "none")

    take(df[pred_na], 2)                    # JSON invalide
    take(df[gold_none & pred_csrd], 3)      # faux positifs
    take(df[~gold_none & pred_csrd], 2)     # accords
    take(df, 2)                             # complement

    return picked[:8] or FALLBACK_EXAMPLES


_state = {"mode": None, "tok": None, "base": None, "orpo": None, "df": None}


# ════════════════════════════════════════════════════════════════════════════
# Chargement — GPU si disponible, sinon repli sur les prédictions enregistrées
# ════════════════════════════════════════════════════════════════════════════

def init():
    if _state["mode"]:
        return _state["mode"]
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("pas de GPU")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto")
        orpo = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=bnb, device_map="auto"),
            ADAPTER_REPO)
        _state.update(mode="live", tok=tok, base=base, orpo=orpo)
    except Exception as e:
        print(f"[init] Mode live indisponible ({e}) — bascule sur les prédictions enregistrées.")
        df = pd.read_csv(PRECOMPUTED) if PRECOMPUTED.exists() else None
        _state.update(mode="precomputed", df=df)
    return _state["mode"]


def _generate(model, text: str) -> str:
    import torch
    tok = _state["tok"]
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nExtrait : «{text}» [/INST]"
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=3000).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.0,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _format(raw: str) -> tuple[str, str]:
    """Retourne (catégorie affichée, bloc JSON formaté ou message d'erreur)."""
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            raise ValueError("aucun objet JSON complet")
        data = json.loads(raw[s:e])
        cat = str(data.get("csrd_category", "?"))
        return cat, json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as err:
        return "⚠️ JSON INVALIDE", f"Sortie brute non parsable ({err}) :\n\n{raw[:600]}"


def classify(text: str):
    if not text or not text.strip():
        m = "Sélectionnez un paragraphe dans le menu déroulant, ou collez un extrait."
        return "—", m, "—", m, ""

    mode = init()

    if mode == "precomputed":
        df = _state["df"]
        if df is None:
            m = ("**Fichier `eval_predictions.csv` absent du Space.** Sans lui, aucune "
                 "prédiction ne peut être affichée. Téléversez-le à la racine du Space "
                 "(Files → Add file), puis rechargez la page.")
            return "fichier manquant", m, "fichier manquant", m, m

        hit = df[df["content"].astype(str).str.strip().str[:80] == text.strip()[:80]]
        if hit.empty:
            m = ("⚠️ **Ce texte ne fait pas partie des 140 paragraphes évalués.**\n\n"
                 "Ce Space tourne sans GPU : il ne peut pas exécuter les modèles en direct, "
                 "il rejoue des prédictions déjà enregistrées. Seuls les paragraphes du "
                 "Gold Standard ont donc une réponse.\n\n"
                 "→ **Choisissez un paragraphe dans le menu déroulant ci-dessus.**\n\n"
                 "Pour soumettre vos propres textes, il faut activer un GPU "
                 "(Settings → Hardware → ZeroGPU) ou utiliser le script Colab du dépôt.")
            return ("hors du jeu évalué", m, "hors du jeu évalué", m, m)

        row = hit.iloc[0]
        gold = str(row.get("csrd_category", "?"))
        pred = row.get("pred_category")
        invalid = pd.isna(pred)
        pred = "⚠️ JSON INVALIDE" if invalid else str(pred)

        detail_gold = f"**Annotation experte (référence) : `{gold}`**"
        if invalid:
            detail_pred = ("**Le modèle ORPO n'a produit aucun JSON exploitable.**\n\n"
                           "Il émet une sortie malformée dans 30,7 % des cas, contre 0 % "
                           "pour le modèle de base.")
        else:
            detail_pred = f"**Prédiction du modèle ORPO : `{pred}`**"

        if invalid:
            verdict = ("🔴 **Sortie invalide.** Le fine-tuning a dégradé une capacité que le "
                       "modèle de base maîtrisait parfaitement : produire du JSON conforme.")
        elif gold != pred:
            verdict = (f"🔴 **Erreur du modèle ORPO** — référence `{gold}`, prédiction "
                       f"`{pred}`. Il sur-prédit les catégories CSRD : 64,9 % de faux "
                       f"positifs mesurés sur l'ensemble du jeu.")
        else:
            verdict = f"🟢 **Correct** — le modèle ORPO retrouve la catégorie `{gold}`."

        return gold, detail_gold, pred, detail_pred, verdict

    raw_base = _generate(_state["base"], text)
    raw_orpo = _generate(_state["orpo"], text)
    cat_b, json_b = _format(raw_base)
    cat_o, json_o = _format(raw_orpo)

    verdict = (
        "✅ Les deux modèles s'accordent." if cat_b == cat_o else
        f"⚠️ **Divergence** — base : `{cat_b}` · ORPO : `{cat_o}`. "
        "Le modèle ORPO sur-prédit les catégories CSRD (FPR de 64,9 % mesuré) "
        "et produit du JSON invalide dans 30,7 % des cas."
    )
    return cat_b, json_b, cat_o, json_o, verdict


# ════════════════════════════════════════════════════════════════════════════
# Interface
# ════════════════════════════════════════════════════════════════════════════

_EXAMPLES = load_examples()
_MODE = init()
_OFFLINE = _MODE == "precomputed"

# Libellés courts pour le menu déroulant : l utilisateur doit pouvoir choisir
# un paragraphe sans lire 700 caracteres.
_CHOICES = [(f"{i+1}. {e[0][:90]}…", e[0]) for i, e in enumerate(_EXAMPLES)]

with gr.Blocks(title="Effondrement d'une optimisation de préférence") as demo:
    gr.Markdown(
        """
        # Quand corriger un biais casse le modèle

        Un adaptateur ORPO a été entraîné sur Mistral-7B pour corriger un biais de récence
        en classification CSRD/ESRS. **Toutes les métriques d'entraînement indiquaient un
        succès.** Le taux de faux négatifs, la métrique cible, avait atteint sa valeur
        idéale de zéro.

        En réalité, le modèle s'était effondré : il avait cessé de prédire `none`, ce qui
        annule le faux-négatif mécaniquement sans rien corriger.

        | | Accuracy | Sorties invalides | FPR global |
        |---|---|---|---|
        | Modèle de base | **70,8 %** | **0 %** | — |
        | Après ORPO | 35,7 % | **30,7 %** | **64,9 %** |
        """
    )

    if _OFFLINE:
        gr.Markdown(
            """
            > ### ℹ️ Mode hors-ligne — comment utiliser cette démo
            >
            > Ce Space tourne **sans GPU**. Il ne peut donc pas exécuter Mistral-7B en direct,
            > et rejoue à la place les prédictions **réellement enregistrées** lors de
            > l'évaluation sur les 140 paragraphes annotés.
            >
            > **➡️ Choisissez un paragraphe dans le menu ci-dessous, puis cliquez sur
            > « Comparer ».** Saisir votre propre texte ne renverra rien dans ce mode.
            """
        )

    picker = gr.Dropdown(
        choices=_CHOICES,
        value=_CHOICES[0][1] if _CHOICES else None,
        label="Paragraphes disponibles (issus du Gold Standard annoté)",
        visible=_OFFLINE,
    )

    txt = gr.Textbox(
        label="Extrait analysé",
        lines=6,
        value=_EXAMPLES[0][0] if _EXAMPLES else "",
        placeholder="Collez un paragraphe issu d'un dépôt AMF…",
    )

    btn = gr.Button("Comparer les deux modèles", variant="primary", size="lg")
    verdict = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Référence\nAnnotation experte (Gold Standard)"
                        if _OFFLINE else "### Modèle de base\n`Mistral-7B-Instruct-v0.3`")
            cb = gr.Textbox(label="Catégorie", interactive=False)
            jb = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Après ORPO\n`Mistral-7B-ORPO-CSRD`")
            co = gr.Textbox(label="Catégorie", interactive=False)
            jo = gr.Markdown()

    picker.change(lambda v: v, inputs=picker, outputs=txt)
    btn.click(classify, inputs=txt, outputs=[cb, jb, co, jo, verdict])
    demo.load(classify, inputs=txt, outputs=[cb, jb, co, jo, verdict])

    gr.Markdown(
        """
        ---
        ⛔ **Le modèle ORPO n'est pas destiné à un usage réel.** Il est publié comme artefact
        de recherche documentant un mode d'échec de l'optimisation de préférence en régime de
        données rares. Pour une classification CSRD fonctionnelle, utilisez le modèle de base.

        📊 [Dataset FinCAC40](https://huggingface.co/datasets/CID99/FinCAC40) ·
        🤖 [Modèle](https://huggingface.co/CID99/Mistral-7B-ORPO-CSRD)
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
