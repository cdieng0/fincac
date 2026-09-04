"""
shared_prompts.py — Bloc taxonomie/tâche/few-shot CANONIQUE et UNIQUE
═══════════════════════════════════════════════════════════════════════════════════════
Source de vérité unique pour le contenu de prompt utilisé par :
  - benchmark_temporal_drift.py  (Section 4 — mesure du biais, FNR=0.667 sur 2010-2014)
  - build_orpo_pairs.py          (Phase 6a — collecte des erreurs réelles du modèle)
  - train_orpo_mistral7b.py      (Phase 6b — prompt d'inférence pour l'entraînement)

POURQUOI CE MODULE EXISTE :
    TAXONOMY_BLOCK et FEW_SHOT_EXAMPLES étaient auparavant dupliqués manuellement dans
    benchmark_temporal_drift.py et build_orpo_pairs.py. Cette duplication a dérivé deux
    fois sans que personne ne s'en aperçoive : (1) le texte de TAXONOMY_BLOCK a divergé
    (version détaillée vs condensée), et (2) build_orpo_pairs.py appelait la politique de
    base en 0-shot alors que le FNR=0.667 documenté en Section 4 a été mesuré en 3-shot.
    Résultat mesuré : la distribution de paires ORPO s'est inversée par rapport à
    l'hypothèse H1 (280 faux positifs contre seulement 24 faux négatifs de récence) —
    parce que Phase 6a mesurait en réalité un phénomène différent de celui de Section 4,
    pas parce que H1 serait invalidée.

    Ce module élimine la classe d'erreur "copier-coller qui diverge" en la rendant
    structurellement impossible : il n'existe plus qu'UNE SEULE définition de ces blocs,
    importée partout. Tout changement de prompt se fait ici, une fois, et se propage
    automatiquement aux 3 scripts.

CONTENU (extrait verbatim de benchmark_temporal_drift.py — la source de vérité
originale, celle qui a produit les résultats de Section 4) :
    TAXONOMY_BLOCK      — description des 12 catégories ESRS
    TASK_BLOCK          — instructions de tâche + format de sortie JSON
    FEW_SHOT_EXAMPLES   — les 3 exemples de référence (none/E1/E2/E5/S1 selon n_shot)
    build_system_prompt(n_shot) — assemble le system prompt complet, 0-shot ou n-shot
"""

VALID_CATEGORIES = [
    "none", "ESRS2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
]

TAXONOMY_BLOCK = """

CATÉGORIES DISPONIBLES (12) :

• none — Aucune pertinence CSRD
  Le paragraphe ne traite d'aucun enjeu de durabilité environnemental,
  social ou de gouvernance au sens CSRD. Inclut : opérations financières
  pures (résultats, dividendes, fusions-acquisitions sans volet ESG),
  mentions légales, formalités administratives, déclarations de
  transactions sur titres sans lien avec la durabilité.
  Sous-catégories : aucune.

• ESRS2 — Informations générales (gouvernance, stratégie, IRO)
  Gouvernance des enjeux de durabilité, modèle d'affaires et stratégie,
  processus d'identification des impacts/risques/opportunités (IRO),
  double matérialité au niveau de l'entité (hors enjeu sectoriel spécifique).
  Sous-catégories : ESRS2-GOV, ESRS2-SBM, ESRS2-IRO, ESRS2-MDR.

• E1 — Changement climatique
  Plan de transition climatique, émissions de gaz à effet de serre
  (Scope 1, 2, 3), consommation et mix énergétique, prix interne du
  carbone, effets financiers anticipés des risques climatiques physiques
  et de transition.
  Sous-catégories : E1-1 à E1-9.

• E2 — Pollution
  Pollution de l'air, de l'eau et des sols, substances préoccupantes,
  microplastiques, effets financiers anticipés liés à la pollution.
  Sous-catégories : E2-1 à E2-6.

• E3 — Eau et ressources marines
  Consommation d'eau, prélèvements en zones de stress hydrique,
  ressources marines, rejets dans l'eau.
  Sous-catégories : E3-1 à E3-5.

• E4 — Biodiversité et écosystèmes
  Plan de transition biodiversité, impacts sur les écosystèmes,
  sites sensibles, métriques de pression sur la biodiversité.
  Sous-catégories : E4-1 à E4-5.

• E5 — Utilisation des ressources et économie circulaire
  Flux entrants/sortants de ressources, déchets, recyclage,
  conception circulaire des produits.
  Sous-catégories : E5-1 à E5-6.

• S1 — Effectifs propres
  Caractéristiques des effectifs, négociation collective, diversité,
  rémunération adéquate, santé-sécurité, formation, équilibre vie
  professionnelle/personnelle, incidents et plaintes.
  Sous-catégories : S1-1 à S1-17.

• S2 — Travailleurs de la chaîne de valeur
  Conditions de travail chez les fournisseurs et sous-traitants,
  travail des enfants, travail forcé, dialogue avec les travailleurs
  de la chaîne de valeur.
  Sous-catégories : S2-1 à S2-5.

• S3 — Communautés affectées
  Droits des communautés locales, peuples autochtones, accès aux
  ressources, déplacement de populations, libertés civiles.
  Sous-catégories : S3-1 à S3-5.

• S4 — Consommateurs et utilisateurs finaux
  Sécurité des produits, protection de la vie privée, accès à
  l'information, pratiques commerciales responsables, inclusion.
  Sous-catégories : S4-1 à S4-5.

• G1 — Conduite des affaires
  Culture d'entreprise, protection des lanceurs d'alerte, lutte
  anti-corruption, relations fournisseurs, pratiques de paiement,
  influence politique et lobbying.
  Sous-catégories : G1-1 à G1-6.


SURPRISE DE MARCHÉ (market_surprise)
aucune   — Information de routine, pleinement anticipée
           (ex : publication périodique conforme au calendrier).
faible   — Information globalement attendue, avec des nuances mineures
           par rapport au consensus ou aux communications précédentes.
partielle — Information partiellement anticipée mais d'ampleur supérieure
           ou de nature différente de ce qui était attendu.
forte    — Information non anticipée, rupture avec la communication
           précédente ou le consensus de marché.

RÈGLE CRITIQUE : si le texte ne mentionne aucun enjeu ESG explicite ou implicite,
répondre csrd_category = "none". Ne pas forcer une catégorie CSRD sur un texte
purement financier ou administratif.
"""

TASK_BLOCK = """
Tu es un auditeur ESG senior, expert en réglementation européenne CSRD
(Corporate Sustainability Reporting Directive) et en normes ESRS,
spécialisé dans l'analyse de documents réglementaires AMF français
(Communiqués, Informations privilégiées, Rapports financiers,
Documents de référence, Déclarations de franchissement de seuil).

Ta mission : classifier l'extrait ci-dessous selon la taxonomie CSRD/ESRS.

CONSIGNES STRICTES :
1. Lis l'extrait en entier avant de répondre.
2. Détermine d'abord si le texte contient un enjeu de durabilité
   (environnemental, social ou de gouvernance au sens CSRD).
3. Si aucun enjeu de durabilité n'est présent → csrd_category = "none"
   et applique la règle de cohérence none.
4. Si un enjeu est présent → identifie la catégorie ESRS principale,
   la sous-catégorie la plus probable, et évalue la matérialité.
5. Le raisonnement (chain_of_thought) doit suivre EXACTEMENT cette
   structure en 3 étapes :
   (1) Nature de l'information : type de document, contenu factuel.
   (2) Analyse de matérialité ESG : le texte relève-t-il de la CSRD/ESG ?
       Si oui, quelle catégorie et pourquoi ? Si non, pourquoi ?
   (3) Surprise pour les investisseurs : cette information est-elle
       anticipée par le marché ? Quel niveau de surprise ?

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant
ni après, sans bloc de code markdown :

{
  "csrd_category": "<none|ESRS2|E1|E2|E3|E4|E5|S1|S2|S3|S4|G1>",
  "esrs_subcategory": "<code sous-catégorie ou chaîne vide si none>",
  "chain_of_thought": "<3 phrases : (1) nature du texte, (2) analyse matérialité ESG, (3) surprise investisseurs>"
}
"""

FEW_SHOT_EXAMPLES = [
    # #101 | 2023-2026 | STMICROELECTRONICS NV | none
    {

        "content": (
            "Pour de plus amples informations sur les engagements de ST dans les domaines "
            "de la transition énergétique et du changement climatique, cliquez ici."
        ),
        "answer": {
            "csrd_category": "none",
            "esrs_subcategory": "",
            "reasoning": (
                "Ce paragraphe, extrait d'un document « Informations privilégiées » de STMicroelectronics, "
                "évoque l'engagement du groupe vis-à-vis de l'Accord de Paris ou de la lutte contre le "
                "changement climatique. Le paragraphe ne relève pas de la CSRD/ESG : il s'agit d'une "
                "information administrative ou financière générique, sans mention d'un enjeu environnemental, "
                "social ou de gouvernance durable, et ne mentionne pas de dimension de durabilité. N'ayant pas "
                "de portée ESG, ce paragraphe ne constitue pas une surprise pour les investisseurs au sens de "
                "la matérialité de durabilité."
            ),
        },
    },

    # #23 | 2015-2019 | TOTAL S.A. | E1
    {


        "content": (
            "- L’Accord de Paris : Total reconnaît l'Accord de Paris comme une avancée majeure dans la lutte "
            "contre le réchauffement climatique et soutient les initiatives des États parties prenantes pour "
            "atteindre les objectifs de cet accord."
        ),
        "answer": {
            "csrd_category": "E1",
            "esrs_subcategory": "E1-1",
            "reasoning": (
                "Chez Total, cet extrait (Informations privilégiées) évoque l'engagement du groupe vis-à-vis "
                "de l'Accord de Paris ou de la lutte contre le changement climatique. Le paragraphe relève de "
                "la CSRD/ESG (catégorie E1 — Changement climatique), avec une matérialité notée 3/5 (modérée) "
                "car il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et "
                "d'impact (conséquences sur l'environnement ou la société) ; il mentionne explicitement la "
                "trajectoire climatique du groupe (réduction d'émissions, transition énergétique). Cette "
                "information ne représente pas une surprise pour les investisseurs : il s'agit d'une information "
                "de routine, pleinement anticipée par le marché."
            ),
        },
    },

    # #57 | 2023-2026 | TOTALENERGIES SE | E2
    {
        "content": (
            "En Inde, le gaz naturel jouera un rôle essentiel dans la transition énergétique. En tant que "
            "carburant alternatif plus propre pour les activités industrielles, la cuisson, et le transport, "
            "il contribue à réduire les émissions de gaz à effet de serre et la pollution, améliorant ainsi "
            "la qualité de l'air."
        ),
        "answer": {
            "csrd_category": "E2",
            "esrs_subcategory": "E2-1",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par TotalEnergies, le paragraphe aborde "
                "la réduction de la pollution générée par les activités du groupe. Le paragraphe relève de la "
                "CSRD/ESG (catégorie E2 — Pollution), avec une matérialité notée 4/5 (significative) car il "
                "traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement la réduction de "
                "la pollution générée par les activités du groupe. Du point de vue des investisseurs, la surprise "
                "est faible, car information globalement attendue, avec des nuances mineures par rapport au consensus."
            ),
        },
    },

    # #53 | 2015-2019 | VEOLIA ENVIRONNEMENT | E5
    {
        "content": (
            "Recyclage et Valorisation des Déchets / France Veolia renouvelle auprès du SMEDAR son contrat lié "
            "à l’exploitation de l’unité de valorisation VESTA à Rouen d’une durée de 6 ans et demi pour un "
            "montant de 116 millions d’euros Le Syndicat Mixte d’Elimination des Déchets de l’Arrondissement de "
            "Rouen (SMEDAR) a renouvelé sa confiance à Veolia, après mise en concurrence, pour l’exploitation de "
            "l’Usine de valorisation énergétique VESTA, dans le cadre d’un marché de 6 ans et demi pour un montant "
            "global de 116 M€. Points forts de ce nouveau marché : l’optimisation opérationnelle de l’usine et la "
            "fourniture au SMEDAR d’outils de contrôle lui permettant de maîtriser une unité désormais partie "
            "intégrante du mix énergétique territorial."
        ),
        "answer": {
            "csrd_category": "E5",
            "esrs_subcategory": "E5-2",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par Veolia, le paragraphe porte sur la "
                "gestion des déchets ou l'économie circulaire. Le paragraphe relève de la CSRD/ESG (catégorie E5 — "
                "Utilisation des ressources et économie circulaire), avec une matérialité notée 3/5 (modérée) car "
                "il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement l'économie circulaire "
                "et la gestion des déchets. Pour le marché, il s'agit d'une surprise partielle : information "
                "partiellement anticipée mais d'une ampleur ou d'une nature différente de ce qui était attendu."
            ),
        },
    },

    # #34 | 2023-2026 | SCHNEIDER ELECTRIC SE | S1
    {
        
        "content": (
            "Nous sommes une entreprise humaine rassemblant un écosystème de 150 000 collaborateurs et de plus "
            "d'un million de partenaires dans plus de 100 pays au plus proche de nos clients et de nos parties "
            "prenantes. Nous plaçons la diversité et l'inclusion au cœur de tout ce que nous faisons, guidés par "
            "notre volonté profonde de contribuer à un futur durable pour tous."
        ),
        "answer": {
            "csrd_category": "S1",
            "esrs_subcategory": "S1-3",
            "reasoning": (
                "Il s'agit d'un extrait de Schneider Electric (Acquisition ou cession des actions de l'émetteur) "
                "qui présente les engagements du groupe en matière de diversité, d'inclusion ou de conditions de "
                "travail des collaborateurs (texte dégradé par la reconnaissance OCR). Le paragraphe relève de la "
                "CSRD/ESG (catégorie S1 — Effectifs propres), avec une matérialité notée 2/5 (mineure) car il "
                "traduit un effet à la fois financier (risques/opportunités pour l'entreprise) et d'impact "
                "(conséquences sur l'environnement ou la société) ; il mentionne explicitement les conditions de "
                "travail, la diversité et l'inclusion au sein des effectifs. Du point de vue des investisseurs, "
                "la surprise est faible, car information globalement attendue, avec des nuances mineures par rapport "
                "au consensus."
            ),
        },
    },

    # #63 | 2020-2022 | BOUYGUES | S4
    {
      
        "content": (
            "transitions environnementale, industrielle et numérique, pour œuvrer en faveur d’une croissance plus "
            "durable et énergétiquement plus sobre, et enfin pour accompagner nos clients dans leur transition vers "
            "une activité bas carbone. Nous avons tous les atouts pour transformer cette acquisition en un succès "
            "pour l’ensemble des parties prenantes du Groupe, notamment ses collaborateurs, ses clients et ses "
            "actionnaires. »"
        ),
        "answer": {
            "csrd_category": "S4",
            "esrs_subcategory": "S4-4",
            "reasoning": (
                "Dans ce document « Informations privilégiées » publié par Bouygues, le paragraphe aborde la "
                "protection des consommateurs et des pratiques commerciales responsables. Le paragraphe relève de "
                "la CSRD/ESG (catégorie S4 — Consommateurs et utilisateurs finaux), avec une matérialité notée 1/5 "
                "(négligeable) car il traduit un effet à la fois financier (risques/opportunités pour l'entreprise) "
                "et d'impact (conséquences sur l'environnement ou la société) ; il mentionne explicitement la "
                "protection des consommateurs et des pratiques commerciales responsables. Cette information ne "
                "représente pas une surprise pour les investisseurs : il s'agit d'une information de routine, "
                "pleinement anticipée par le marché."
            ),
        },
    },

    # #38 | 2023-2026 | ENGIE | ESRS2
    {
        "categorie": "ESRS2",
        "content": (
            "- Enfin, la stratégie climatique du Groupe inclut désormais également l’adaptation au changement "
            "climatique afin d’assurer l'intégrité des actifs et des chaînes d'approvisionnement résilientes."
        ),
        "answer": {
            "csrd_category": "ESRS2",
            "esrs_subcategory": "ESRS2-SBM",
            "reasoning": (
                "Ce paragraphe, extrait d'un document « Informations privilégiées » de Engie, aborde la stratégie "
                "ou la gouvernance des enjeux de durabilité de l'entreprise. Le paragraphe relève de la CSRD/ESG "
                "(catégorie ESRS2 — Informations générales (gouvernance, stratégie, IRO)), avec une matérialité "
                "notée 3/5 (modérée) car il traduit un effet à la fois financier (risques/opportunités pour "
                "l'entreprise) et d'impact (conséquences sur l'environnement ou la société) ; il mentionne "
                "explicitement la stratégie ou la gouvernance des enjeux de durabilité de l'entreprise. Pour le "
                "marché, il s'agit d'une surprise partielle : information partiellement anticipée mais d'une ampleur "
                "ou d'une nature différente de ce qui était attendu."
            ),
        },
    },
]


def build_system_prompt(n_shot: int) -> str:
    """
    Construit le system prompt zero-shot ou few-shot — fonction IDENTIQUE à celle de
    benchmark_temporal_drift.py (copie exacte, cf. docstring du module).
    """
    import json as _json
    blocks = [TAXONOMY_BLOCK.strip(), ""]

    if n_shot > 0:
        blocks.append("EXEMPLES DE RÉFÉRENCE :\n")
        for i, ex in enumerate(FEW_SHOT_EXAMPLES[:n_shot], 1):
            blocks.append(f"Exemple {i} :")
            blocks.append(f"Extrait : «{ex['content']}»")
            blocks.append(f"Réponse : {_json.dumps(ex['answer'], ensure_ascii=False)}")
            blocks.append("")

    blocks.append(TASK_BLOCK.strip())
    return "\n".join(blocks)


def few_shot_signatures(n_shot: int) -> set:
    """Signatures (100 premiers caractères) des exemples few-shot, pour exclusion du pool/test set."""
    return {ex["content"][:80] for ex in FEW_SHOT_EXAMPLES[:n_shot]}
