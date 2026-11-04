# _____________________________________________________________________
# _____________________________________________________________________
# APPLICATION STREAMLIT — PRÉDICTION DE L'ÉTAT DE CONSOLIDATION DU SOL
# Propriétaire : Ir Moïse Bintu
# _____________________________________________________________________
# _____________________________________________________________________

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import sys

# 1. CONFIGURATION GÉNÉRALE
# _____________________________________________________________________
st.set_page_config(
    page_title="GeoPredict | Consolidation du sol",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELE_PATH = Path(__file__).resolve().parent

# 2. STYLE VISUEL
# _____________________________________________________________________

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --navy: #12233f;
            --blue: #246bfd;
            --blue-dark: #174fc2;
            --sky: #eaf2ff;
            --ink: #172033;
            --muted: #68738a;
            --line: #e7ebf3;
            --surface: #ffffff;
            --success: #087f5b;
            --warning: #b35c00;
        }

        .stApp {
            background: #f6f8fc;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12233f 0%, #1b3761 100%);
        }
        [data-testid="stSidebar"] * { color: #f5f8ff !important; }
        [data-testid="stSidebar"] .stCaption { color: #c8d5ec !important; }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 10px 0 28px;
        }
        .brand-mark {
            width: 46px;
            height: 46px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: linear-gradient(135deg, #4e8bff, #7ab3ff);
            color: white;
            font-size: 23px;
            box-shadow: 0 8px 22px rgba(52, 119, 255, .28);
        }
        .brand-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -.3px;
        }
        .brand-subtitle { font-size: 11px; color: #b9c9e5 !important; }

        .hero {
            padding: 30px 34px;
            border-radius: 24px;
            color: white;
            background: radial-gradient(circle at 88% 10%, rgba(92, 151, 255, .42), transparent 27%),
                        linear-gradient(120deg, #12233f 0%, #1e4d91 100%);
            box-shadow: 0 16px 34px rgba(23, 52, 97, .16);
            margin-bottom: 24px;
        }
        .eyebrow {
            color: #a9c7ff;
            text-transform: uppercase;
            letter-spacing: 1.6px;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(27px, 3vw, 40px);
            line-height: 1.1;
            margin: 0 0 12px;
            color: white;
        }
        .hero p { color: #d8e6ff; margin: 0; font-size: 15px; }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 21px;
            font-weight: 700;
            margin: 8px 0 4px;
            color: var(--navy);
        }
        .section-caption { color: var(--muted); font-size: 14px; margin-bottom: 16px; }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 22px 24px 8px;
            box-shadow: 0 8px 25px rgba(27, 46, 79, .05);
        }
        div[data-testid="stFormSubmitButton"] button,
        .stButton > button {
            border: 0;
            border-radius: 11px;
            background: linear-gradient(135deg, var(--blue), var(--blue-dark));
            color: white;
            font-weight: 700;
            min-height: 46px;
            box-shadow: 0 8px 16px rgba(36, 107, 253, .2);
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover { background: #123f9f; color: white; }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(27, 46, 79, .04);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--navy); }

        /* Les libellés des champs doivent rester visibles sur tous les thèmes. */
        [data-testid="stNumberInput"] label,
        [data-testid="stNumberInput"] label p {
            color: var(--navy) !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }
        [data-testid="stNumberInput"] input {
            color: var(--ink) !important;
            background: #ffffff !important;
        }

        /* Texte des encadrés d'information et d'avertissement */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {
        color: var(--ink) !important;
        }
        .result-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 25px rgba(27, 46, 79, .05);
            margin-top: 20px;
        }
        .result-label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .result-value {
            color: var(--navy);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 30px;
            font-weight: 700;
            margin-top: 5px;
        }
        .owner {
            border-top: 1px solid rgba(255,255,255,.15);
            margin-top: 28px;
            padding-top: 16px;
            font-size: 12px;
            color: #c8d5ec !important;
        }
        .owner strong { color: white !important; }
        .footer {
            text-align: center;
            color: #8a94a8;
            font-size: 12px;
            padding: 34px 0 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. CHARGEMENT DES MODÈLES — MIS EN CACHE
# _____________________________________________________________________

@st.cache_resource(show_spinner=False)
def charger_objets(chemin: str):
    dossier = Path(chemin)
    fichiers = {
        "model": dossier / "modele_rf.pkl",
        "yeo": dossier / "yeojohnson.pkl",
        "colonnes": dossier / "colonnes.pkl",
        "encoder": dossier / "label_encoder.pkl",
        "plages": dossier / "plages_entrainement.pkl",
    }

    obligatoires = [fichiers[k] for k in ("model", "yeo", "colonnes", "encoder")]
    manquants = [str(path) for path in obligatoires if not path.exists()]
    if manquants:
        raise FileNotFoundError("Fichiers obligatoires introuvables :\n" + "\n".join(manquants))

    plages = joblib.load(fichiers["plages"]) if fichiers["plages"].exists() else None
    return (
        joblib.load(fichiers["model"]),
        joblib.load(fichiers["yeo"]),
        joblib.load(fichiers["colonnes"]),
        joblib.load(fichiers["encoder"]),
        plages,
    )

# 4. BARRE LATÉRALE
# _____________________________________________________________________

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🪨</div>
            <div>
                <div class="brand-title">GeoPredict</div>
                <div class="brand-subtitle">Analyse géotechnique intelligente</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Navigation")
    st.caption("Tableau de prédiction")
    st.caption("Modèle : Random Forest")
    st.caption("Domaine : géotechnique")
    st.markdown(
        '<div class="owner">Application développée et détenue par<br><strong>Ir Moïse Bintu</strong></div>',
        unsafe_allow_html=True
    )

# 5. EN-TÊTE PRINCIPAL
# _____________________________________________________________________

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Plateforme d'aide à la décision</div>
        <h1>Prédiction de l'état de consolidation du sol</h1>
        <p>Renseignez les paramètres géotechniques afin d'obtenir une classification instantanée et lisible.</p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    model, yeo_johnson, colonnes, label_encoder, plages_entrainement = charger_objets(str(MODELE_PATH))
except Exception as erreur:
    st.error("Impossible de charger les fichiers du modèle.")
    st.code(str(erreur))
    st.info("Vérifie le chemin MODELE_PATH et la présence des cinq fichiers .pkl.")
    st.stop()

# 6. FORMULAIRE DE SAISIE
# _____________________________________________________________________

st.markdown('<div class="section-title">Paramètres d’entrée</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Entrez les caractéristiques géotechniques du sol à analyser. Veuillez renseigner chaque valeur dans l’unité indiquée.</div>',
    unsafe_allow_html=True
)

with st.expander("Comment choisir les valeurs à saisir ?", expanded=True):
    st.info(
        "Renseignez les caractéristiques géotechniques du sol à analyser. "
        "Utilisez les valeurs mesurées dans votre étude et respectez les unités indiquées "
        "pour chaque paramètre."
    )
    st.warning(
        "Les résultats fournis par cette application constituent une aide à l'analyse "
        "et doivent être interprétés en tenant compte du contexte géotechnique de l'étude."
    )

DESCRIPTION_PARAMETRES = {

    " LL (%)": {
        "nom": "Limite de liquidité - LL(%)"
    },

    "PL (%)": {
        "nom": "Limite de plasticité - PL(%)"
    },

    " w (%)": {
        "nom": "Teneur en eau naturelle - w(%)"
    },

    " LI": {
        "nom": "Indice de liquidité - LI"
    },

    "Profondeur (m)": {
        "nom": "Profondeur du prélèvement (m)"
    }

}

with st.form("form_prediction") :
    valeurs = []

    champs = st.columns(2, gap="large")

    for index, colonne in enumerate(colonnes):

        with champs[index % 2]:

            cle = str(colonne)

            details = DESCRIPTION_PARAMETRES.get(cle, {})
            nom = details.get("nom", cle)

            libelle = f"{index + 1}. {nom}"

            valeur = st.number_input(
                libelle,
                value=0.0,
                step=0.01,
                format="%.4f"
            )

            valeurs.append(valeur)

    st.markdown("<br>", unsafe_allow_html=True)

    lancer_prediction = st.form_submit_button(
        "Lancer la prédiction  →",
        use_container_width=True
    )

# PRÉDICTION ET AFFICHAGE DES RÉSULTATS
# _____________________________________________________________________

if lancer_prediction :
    try:
        nouvelle_donnee = pd.DataFrame([valeurs], columns=colonnes)

        # Contrôles de cohérence géotechnique.
        ll_col = next((c for c in colonnes if str(c).strip().upper().startswith(("LL", "W_L"))), None)
        pl_col = next((c for c in colonnes if str(c).strip().upper().startswith(("PL", "W_P"))), None)
        if ll_col is not None and pl_col is not None:
            if float(nouvelle_donnee.iloc[0][pl_col]) > float(nouvelle_donnee.iloc[0][ll_col]):
                st.error("La limite de plasticité PL doit être inférieure à la limite de liquidité LL.")
                st.stop()

        # Contrôle du domaine d'apprentissage si le fichier est disponible.
        if plages_entrainement is None:
            st.warning("Le fichier plages_entrainement.pkl est absent : le contrôle du domaine de validité n'est pas disponible.")
        else:
            hors_plage = []
            for colonne in colonnes:
                borne = plages_entrainement.get(colonne) if isinstance(plages_entrainement, dict) else None
                if borne is None:
                    continue
                minimum = float(borne["min"])
                maximum = float(borne["max"])
                valeur = float(nouvelle_donnee.iloc[0][colonne])
                if valeur < minimum or valeur > maximum:
                    hors_plage.append(f"{colonne} : {valeur:g} ; plage [{minimum:g} ; {maximum:g}]")
            if hors_plage:
                st.warning("La prédiction est suspendue : une ou plusieurs valeurs sont hors des plages d'apprentissage.")
                for message in hors_plage:
                    st.write("- " + message)
                st.stop()

        # Yeo-Johnson 
        donnee_transformee = yeo_johnson.transform(nouvelle_donnee)

        prediction = model.predict(donnee_transformee)[0]
        probabilites = model.predict_proba(donnee_transformee)[0]

        classe_brute = str(label_encoder.inverse_transform([prediction])[0]).strip().lower()

        libelles_classes = {
            "sur-consolide": "Sol surconsolidé",
            "surconsolide": "Sol surconsolidé",
            "surconsolidé": "Sol surconsolidé",
            "non-sur-consolide": "Sol non surconsolidé",
            "non-surconsolide": "Sol non surconsolidé",
            "non sur-consolide": "Sol non surconsolidé",
            "non surconsolide": "Sol non surconsolidé",
            "non surconsolidé": "Sol non surconsolidé",
            "0": "Sol non surconsolidé",
            "1": "Sol surconsolidé",
        }
        classe = libelles_classes.get(classe_brute, str(label_encoder.inverse_transform([prediction])[0]))
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Classe prédite</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{classe}</div>', unsafe_allow_html=True)
        st.caption("Résultat indicatif produit par le modèle Random Forest.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.info(
            "Cet outil ne remplace pas un essai œdométrique ni une étude géotechnique complète ; " \
            "il constitue uniquement une aide à l’estimation préliminaire de l’état de surconsolidation des sols fins."
        )

    except Exception as erreur:
        st.error("Une erreur est survenue pendant la prédiction.")
        st.exception(erreur)

with st.expander("Informations techniques et reproductibilité"):
    st.write(f"**Python :** {sys.version.split()[0]}")
    st.write(f"**Streamlit :** {st.__version__}")
    st.write(f"**scikit-learn :** {sklearn.__version__}")
    st.write(f"**pandas :** {pd.__version__}")
    st.write(f"**NumPy :** {np.__version__}")
    st.write(f"**joblib :** {joblib.__version__}")

st.markdown(
    '<div class="footer">© 2026 · GeoPredict · Propriétaire : <strong>Ir Moïse Bintu</strong></div>',
    unsafe_allow_html=True,
)

