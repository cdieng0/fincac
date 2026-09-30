"""
compare_base_vs_orpo_colab.py — Tester l'effondrement en direct, sur n'importe quel texte
═══════════════════════════════════════════════════════════════════════════════════════
Charge le modèle de base ET le modèle ORPO, les interroge sur le même extrait, et
affiche les deux réponses côte à côte.

À exécuter sur Google Colab (runtime GPU T4 suffit) ou toute machine avec GPU.
Indépendant de Hugging Face Spaces : c'est le moyen le plus direct de vérifier
l'effondrement avant ou après publication.

Sur Colab, dans une cellule :
    !pip install -q transformers peft bitsandbytes accelerate
    !python src/orpo/compare_base_vs_orpo_colab.py --adapter VOTRE_USERNAME/Mistral-7B-ORPO-CSRD

Ou pour tester un texte précis :
    !python src/orpo/compare_base_vs_orpo_colab.py --adapter ... --text "Votre paragraphe ici"
"""

import argparse
import json

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

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

# Textes de test conçus pour révéler les deux modes d'échec documentés :
# la sur-prédiction CSRD sur du contenu purement administratif, et le
# déclenchement de la catégorie refuge ESRS2 par le mot « risque ».
DEFAULT_TESTS = [
    ("Administratif pur — attendu : none",
     "La société annonce avoir procédé au rachat de 125 000 actions propres au cours du mois, "
     "dans le cadre du programme de rachat autorisé par l'assemblée générale du 14 mai 2023, "
     "à un prix moyen de 42,17 euros par action."),
    ("Climat explicite — attendu : E1",
     "Le groupe a réduit ses émissions de gaz à effet de serre Scope 1 et 2 de 18 % sur "
     "l'exercice, dépassant l'objectif annuel de 12 % fixé dans son plan de transition "
     "climatique, grâce à l'arrêt anticipé de deux unités de production au charbon."),
    ("Piège « risque » — attendu : none, déclenche souvent ESRS2",
     "Le présent communiqué contient des déclarations prospectives. Les facteurs de risque "
     "susceptibles d'affecter les résultats sont décrits dans le document d'enregistrement "
     "universel disponible sur le site de la société."),
]


def build_prompt(text: str) -> str:
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nExtrait : «{text}» [/INST]"


def parse(raw: str):
    """Retourne (catégorie, json_valide_bool)."""
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            return "⚠️ JSON INVALIDE", False
        d = json.loads(raw[s:e])
        return str(d.get("csrd_category", "?")), True
    except Exception:
        return "⚠️ JSON INVALIDE", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Dépôt HF ou chemin local de l'adaptateur ORPO")
    ap.add_argument("--text", default=None, help="Texte unique à tester")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if not torch.cuda.is_available():
        print("❌ Aucun GPU détecté. Sur Colab : Exécution → Modifier le type d'exécution → GPU")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print(f"Chargement du modèle de base ({BASE_MODEL})…")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")

    print(f"Chargement de l'adaptateur ORPO ({args.adapter})…")
    orpo = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"),
        args.adapter)

    def ask(model, text):
        inp = tok(build_prompt(text), return_tensors="pt",
                  truncation=True, max_length=3000).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    tests = [("Texte fourni", args.text)] if args.text else DEFAULT_TESTS

    n_diff = n_invalid = 0
    for label, text in tests:
        print(f"\n{'═'*70}\n  {label}\n{'═'*70}")
        print(f"  « {text[:110]}… »\n")

        cb, _ = parse(ask(base, text))
        co, ok_o = parse(ask(orpo, text))

        print(f"  Modèle de base : {cb}")
        print(f"  Après ORPO     : {co}")

        if cb != co:
            n_diff += 1
            print("  → DIVERGENCE")
        if not ok_o:
            n_invalid += 1
            print("  → Le modèle ORPO n'a pas produit de JSON exploitable")

    print(f"\n{'═'*70}")
    print(f"  Sur {len(tests)} test(s) : {n_diff} divergence(s), "
          f"{n_invalid} sortie(s) ORPO invalide(s)")
    print("  Références mesurées sur les 140 paragraphes du Gold Standard :")
    print("    accuracy 70,8 % (base) contre 35,7 % (ORPO)")
    print("    sorties invalides 0 % (base) contre 30,7 % (ORPO)")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
