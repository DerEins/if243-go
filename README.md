# Auteurs

- Nolan Bizon : [nbizon@bordeaux-inp.fr](nbizon@bordeaux-inp.fr)
- Mathieu Dupoux : [mdupoux@bordeaux-inp.fr](mdupoux@bordeaux-inp.fr)

# Modèle choisi

## Heuristique

Notre modèle utilise une heuristique déterminée par le réseau de neurones [`model.keras`](./models/model.keras) qu'on a établi pour le TP d'IA. Celui-ci prend donc en entrée le plateau courant d'une partie et en évalue la probabilité de victoire de notre joueur.

## Élagage

Nous avons choisi d'utiliser l'algorithme d'élagage alpha-bêta couplé à l'heuristique évaluée par notre réseau de neurones pour notre modèle. Celui-ci est optimisé grâce à une table de hachage des plateaux déjà évalués, permettant à notre joueur de s'économiser l'évaluation des plateaux déjà évalués.

## Ouverture

Les évaluations de plateaux en début de jeu étant les plus longues (puisqu'il y beaucoup de coups légaux à tester), nous avons voulu exploiter la banque d'ouverture de professionnels fournie ([`openings.json`](./data/openings.json)) avec le sujet. À noter qu'on a augmenté cette base en appliquant les symétries et rotations possibles d'un plateau (8 fois plus de données) grâce au script [`openingDataDuplicator.py`](./utils/openingDataDuplicator.py). Ainsi, pour les dix premiers coups joués lors d'une partie, on regarde si le début de partie correspond à une ou des ouvertures de la banque :

- S'il y a plusieurs ouvertures possibles parmi cette banque, alors on choisi celle avec la meilleure évaluation par la même heuristique que notre alpha-bêta.
- S'il n'y a aucune ouverture correspondante dans notre banque, alors on choisi aléatoirement un des coups de celle-ci de la profondeur correspondante à la partie en cours.

# Structure de fichiers

```
├── data
│   ├── gnugo0-VS-gnugo0.json
│   ├── openings.json
│   └── scores.json
├── models
│   └── model.keras
├── utils
│   ├── trainingDataGenerator.py
│   └── openingDataDuplicator.py
├── myPlayer.py
├── gnugoPlayer.py
├── randomPlayer.py
├── GnuGo.py
├── Goban.py
├── helper.py
├── localGame.py
├── namedGame.py
├── playerInterface.py
├── visualGame.ipynb
└── README.md
```

# Pistes d'amélioration

## Génération de données d'entraînements

Nous avons essayé de générer nos propres données d'entraînements de notre modèle, l'objectif étant d'avoir des données pour des affrontement de GnuGo niveau 10 contre lui-même. Malheureusement, nous n'avons pas réussi à dupliquer les joueurs (notamment les processus `gnugo`), opération nécessaire pour dériver un plateaux en 100 parties différentes. Un de nos essais sur GnuGo niveau 0 est disponible dans [`gnugo0-VS-gnugo0.json`](./data/gnugo0-VS-gnugo0.json), fichier généré par [`trainingDataGenerator`](./utils/trainingDataGenerator.py).
