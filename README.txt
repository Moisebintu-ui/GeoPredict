# GeoPredict — Prédiction de l’état de consolidation des sols fins

1. Présentation du projet

GeoPredict est une application développée avec Streamlit dans le cadre d’un mémoire de Master 2 en génie civil. Elle repose sur un modèle de classification par apprentissage automatique (Machine Learning), fondé sur l’algorithme Random Forest, pour estimer l’état de consolidation des sols fins à partir de paramètres géotechniques.

L’objectif est de proposer un outil d’aide à l’analyse préliminaire permettant de distinguer les sols surconsolidés des sols non surconsolidés, à partir de caractéristiques accessibles sans recourir directement aux paramètres dérivés de l’essai œdométrique comme variables d’entrée.

L’application effectue des contrôles sur les données saisies, applique le prétraitement associé au modèle entraîné et affiche la classe prédite.

2. Variables d’entrée

GeoPredict utilise les paramètres suivants :

- Limite de liquidité (LL)
- Limite de plasticité (PL)
- Teneur en eau naturelle (w)
- Indice de liquidité (LI)
- Profondeur du prélèvement

Les valeurs doivent être saisies dans les unités attendues par l’application. Des valeurs incorrectes ou situées en dehors des plages observées pendant l’entraînement peuvent empêcher la prédiction.

3. Définition des classes

La variable cible est le rapport de surconsolidation, OCR (Overconsolidation Ratio). La classification binaire utilise le seuil de 1,05 :

- Surconsolidé : OCR > 1,05 ;
- Non surconsolidé : OCR ≤ 1,05.

Dans cette étude, la classe non surconsolidée regroupe les sols sous-consolidés et normalement consolidés selon la définition opérationnelle retenue.

4. Données utilisées

Le développement du modèle s’appuie sur la base de données CLAY/10/7490, constituée de 7 490 échantillons d’argiles issus de 251 études.

Après sélection des observations disposant d’une valeur d’OCR et application des critères de traitement des données manquantes, le jeu de données final utilisé pour la classification comprend 1 602 observations.

Les performances obtenues dépendent de la qualité, de la représentativité et de la diversité des données utilisées pour l’apprentissage.

5. Organisation du dépôt

Le dépôt contient les fichiers nécessaires à l’utilisation de l’application :

	GeoPredict/
	├── App_Etude.py : code source de l’application Streamlit.
	├── requirements.txt : dépendances Python nécessaires à l’exécution.
	├── README.md
	├── modele_rf.pkl : modèle Random Forest entraîné.
	├── yeojohnson.pkl : objet de prétraitement utilisé par le modèle.
	├── colonnes.pkl : noms et ordre des variables d’entrée.
	├── label_encoder.pkl : correspondance entre les classes encodées et leurs libellés.
	└── plages_entrainement.pkl : plages de valeurs observées dans les données d’entraînement, utilisées pour contrôler les entrées.

6. Installation 

# Prérequis

	- Python installé sur l'ordinateur ;
	- Git, si le dépôt doit être cloné ;
	- une connexion Internet pour installer les dépendances.

## Installation

1. Télécharger ou cloner le dépôt du projet.
2. Ouvrir un terminal dans le dossier du projet, c’est-à-dire le dossier contenant `App_Etude.py` et `requirements.txt`.
3. Installer les dépendances avec la commande suivante :

   ```bash
   python -m pip install -r requirements.txt
   ```

7. Lancement de l’application

Depuis le dossier du projet, exécuter :

```bash
   streamlit run App_Etude.py
```

L'application s'ouvre dans le navigateur. L'utilisateur peut alors renseigner les paramètres géotechniques demandés pour obtenir la classe prédite.

Pour arrêter l’application, revenir dans le terminal et appuyer sur `Ctrl + C`.

8. Précautions d'utilisation et limites

Les résultats fournis par GeoPredict sont des estimations préliminaires. Ils ne remplacent pas les essais géotechniques de laboratoire, notamment l’essai œdométrique, ni l’interprétation d’un ingénieur.

La fiabilité des prédictions dépend des données utilisées pour entraîner le modèle et du domaine auquel elles s’appliquent. Les résultats doivent donc être interprétés avec prudence et confrontés aux observations et essais disponibles.

9. Cadre académique

Cette application a été développée dans le cadre d'un travail de recherche universitaire en génie civil portant sur l'application des techniques d'apprentissage automatique à la prédiction de l'état de surconsolidation des sols fins.

Auteur : Moïse Bintu
Niveau : Master 2 en géotechnique
Établissement : ULPGL

10. Licence et réutilisation

Les conditions de réutilisation du code, du modèle et des fichiers associés doivent être définies par l'auteur. En l'absence d'une licence explicite, aucune autorisation générale de réutilisation ou de redistribution ne doit être présumée.