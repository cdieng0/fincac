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
HF_USER = os.environ.get("SPACE_ID", "").split("/")[0] or "cheikhibra"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", f"{HF_USER}/Mistral-7B-ORPO-CSRD")
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USER}/FinCAC40")
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

EXAMPLES = [
    # accord correct — gold=E1, ORPO=E1 (2023-2026)
    ["Geoff West, vice-président exécutif en charge des achats pour STMicroelectronics, a déclaré : « Ce deuxième contrat d’achat d’électricité en Italie représente une nouvelle étape importante dans la réalisation de l’objectif de ST qui est d’atteindre la neutralité carbone dans ses activités (émissions des scopes 1 et 2, et une partie du scope 3) d’ici à 2027, notamment en s’approvisionnant à 100 % en énergies renouvelables d’ici à 2027. Les accords d’achat d’électricité joueront un rôle majeur dans notre transition."],
    # faux positif — gold=none, ORPO=ESRS2 (2020-2022)
    ["- les franchissements de seuils à la hausse déclarés par Agache Commandité résultent d’une réorganisation consistant en la transformation de la société Agache en une société en commandite par actions, avec la désignation de Monsieur Bernard Arnault en tant que gérant commandité d’Agache SCA et la création d’une société par actions simplifiée à capital variable (dont le capital social est détenu à parité par les cinq enfants de Monsieur Bernard Arnault), Agache Commandité SAS, associée commandité d’Agache SCA ;"],
    # json invalide — gold=none, ORPO=JSON invalide (2010-2014)
    ["Patrick Sayer Diplômé de l’École Polytechnique et de l’École des Mines de Paris, Patrick Sayer a notamment occupé les fonctions d’Associé-gérant de Lazard Frères et Cie à Paris et Managing Director de Lazard Frères & Co à New York. Il a participé à la création de Fonds Partenaires de 1989 à 1993. Il a ensuite contribué à la mise en place de la stratégie d’investissement de Gaz et Eaux devenue Eurazeo. Il est Président du Directoire d’Eurazeo depuis mai 2002."],
    # accord correct — gold=S1, ORPO=S1 (2023-2026)
    ["Nous sommes une entreprise humaine rassemblant un écosystème de 150 000 collaborateurs et de plus d'un million de partenaires dans plus de 100 pays au plus proche de nos clients et de nos parties prenantes. Nous plaçons la diversité et l'inclusion au cœur de tout ce que nous faisons, guidés par notre volonté profonde de contribuer à un futur durable pour tous."],
]

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
        return "—", "Saisissez un extrait.", "—", "Saisissez un extrait.", ""

    mode = init()

    if mode == "precomputed":
        df = _state["df"]
        note = ("**Mode hors-ligne.** Ce Space tourne sans GPU : les résultats affichés sont "
                "les prédictions **réellement enregistrées** lors de l'évaluation sur les 140 "
                "paragraphes du Gold Standard — pas des simulations. Utilisez les exemples "
                "ci-dessous : seuls les paragraphes de ce jeu peuvent être affichés dans ce mode.")
        if df is None:
            return "—", note, "—", note, note
        hit = df[df["content"].astype(str).str.strip().str[:80] == text.strip()[:80]]
        if hit.empty:
            msg = (note + "\n\n⚠️ **Cet extrait ne figure pas dans le jeu de 140 paragraphes "
                   "évalués.** En mode hors-ligne, seuls ces paragraphes ont des prédictions "
                   "enregistrées. Cliquez sur un exemple ci-dessous, ou activez un GPU dans "
                   "les paramètres du Space pour soumettre un texte libre.")
            return ("—", msg, "—", msg, msg)
        row = hit.iloc[0]
        gold = str(row.get("csrd_category", "?"))
        pred = row.get("pred_category")
        pred = "⚠️ JSON INVALIDE" if pd.isna(pred) else str(pred)
        return (gold, f"Annotation experte : **{gold}**",
                pred, f"Prédiction ORPO enregistrée : **{pred}**", note)

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

        Saisissez un extrait réglementaire ci-dessous et comparez les deux modèles.
        """
    )

    txt = gr.Textbox(label="Extrait de document réglementaire (français)",
                     lines=5, placeholder="Collez un paragraphe issu d'un dépôt AMF…")
    btn = gr.Button("Comparer les deux modèles", variant="primary")
    verdict = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Modèle de base\n`Mistral-7B-Instruct-v0.3`")
            cb = gr.Textbox(label="Catégorie", interactive=False)
            jb = gr.Code(label="Sortie", language="json")
        with gr.Column():
            gr.Markdown("### Après ORPO\n`Mistral-7B-ORPO-CSRD`")
            co = gr.Textbox(label="Catégorie", interactive=False)
            jo = gr.Code(label="Sortie", language="json")

    gr.Examples(examples=EXAMPLES, inputs=txt, label="Exemples issus du corpus")
    btn.click(classify, inputs=txt, outputs=[cb, jb, co, jo, verdict])

    gr.Markdown(
        f"""
        ---
        ⛔ **Le modèle ORPO n'est pas destiné à un usage réel.** Il est publié comme artefact
        de recherche documentant un mode d'échec de l'optimisation de préférence en régime de
        données rares. Pour une classification CSRD fonctionnelle, utilisez le modèle de base.

        📊 [Dataset FinCAC40](https://huggingface.co/datasets/{DATASET_REPO}) ·
        🤖 [Modèle](https://huggingface.co/{ADAPTER_REPO})
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
