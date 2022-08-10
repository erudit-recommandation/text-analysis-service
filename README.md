# text-analysis-service

Service de [Nemo](https://github.com/erudit-recommandation/Nemo) qui permet l'analyse de texte.
# Support
Linux, non tester sur Windows (par contre avec Docker, il n'y a pas de problème)


# Installation et dépendances
Les dépendances sont dans le fichier `requirement.txt`, ainsi que le module `fr_core_news_sm` de `spacy` et les modules `punkt` et `stopwords`. Il suffit de lancer la commande `make install` pour tout installer.

Le répertoire [initialisation-service](https://github.com/erudit-recommandation/initialisation-service) permet de générer le fichier `gemsim` nécessaire au fonctionnement du service.

# Usage
L'application à deux modes de fonctionnement, développement et production, la seule différence entre les deux modes et l'utilisation du fichier d'environnement (`.env`, `.env_dev`)afin de garder le mot de passe privé.

Il y a plusieurs manières de lancer l'application, en serveur de flask afin d'utiliser les outils de développement en lançant `make run`.

La seconde est via docker en exécutant `make create-docker-debug` suivis de `run-docker-debug`.

La troisième est via docker pour le déploiement avec `make create-docker` suivis de  `run-docker`.

## Addresse

### GET `/`
page d'acceuil indique si le service est opérationnel

### POST `/gemsim`
se sert du modèle `gemsim` afin de retourner les index dans la base de données des textes similaire au texte envoyé.

#### Envoi

```json
{
    "text": string \\ texte de référence
    "n": int \\ nombre maximal de résultats
}
```

#### Retour

```json
{
    "{index}": int \\ l'index dans la base de données ainsi que le score de similitude 
}
```

### GET ET POST `/model`

Route qui gère le modèle `gemsim`, pour interagir il est  nécessaire d'utiliser un mot de passe qui est `SECRET` du fichier `.env`.
#### GET

retourne une page afin d'ajouter manuellement un nouveau modèle `gemsim`

#### POST

permets de publier un nouveau modèle `gemsim` via une requête HTTP

request files: [d2v.model, d2v.model.syn1neg.npy, d2v.model.wv.vectors.npy] 
request form: [password]
