# Projet_7 - Implémentez un modèle de scoring

L'objectif est de développer un modèle de scoring de la probabilité de défaut de paiement d'un client pour étayer la décision d'accorder ou non un prêt à un client potentiel.

## API FastAPI
- Code API : main.py
- Hébergée sur Azure webapp
- https://scoringmodeloc.azurewebsites.net/clients_pretrait
- Entrée : une ligne du dataframe correspondant à un client ainsi que tous les pré-traitements nécessaires
- Sortie : la prédiction en binaire, la prédiction en probabilité

## Dashboard Streamlit
- Code Dashboard : stream_lit.py
- Hébergée sur Azure webapp
- https://scoringcreditoc.azurewebsites.net/
- Fonctionnalités :
    - Sélection d'un client
    - Informations descriptives relatives à un client
    - Visualisation du dataframe et possibilité de le filtrer
    - Visualisation du score et interprétation de ce score
    - Importance locale des paramètres dans la décision du modèle
    - Comparaison des informations descriptives relatives à un client à l'ensemble des clients



