# InnoFaso IA — Pipeline de Prédiction de Pannes

## Architecture

```
Excel (historique 2020-2025)
       │
       ▼
   db_loader.py  ←── PostgreSQL (vue v_ml_maintenance_mensuel)
       │                      (âge, stock, dispo, préventif)
       ▼
 train_model.py ──→ modele_panne_innofaso.pkl
       │
       ▼
 ia_service_complet.py ──→ Flask API :5001
       │
       ▼
 Backend Node.js ──→ Frontend React (jauges risque)
```

---

## 1. Source des données

### Excel (fiable, 478 lignes)
- Fichier : `INNOFASO_Historique_de_maintenance_2025.xlsx`
- Colonnes : Partenaire, Année, Mois, Équipement, Type maintenance, Durée, Description
- **214 correctifs / 478 lignes** soit ~45% de correctifs
- Couvre 2020–2025

### PostgreSQL (enrichissement, 474 lignes)
- Vue matérialisée `v_ml_maintenance_mensuel`
- Colonnes supplémentaires : âge équipement, taux disponibilité, stock pièces, respect planning préventif
- Actuellement non fusionné car les noms d'équipements Excel ne correspondent pas à ceux en base

---

## 2. Préparation des features (train_model.py)

### Features de base (toujours)
| Feature | Description |
|---------|-------------|
| `equipement_enc` | Équipement encodé (LabelEncoder) |
| `annee`, `mois_num`, `trimestre` | Temporelles |
| `mois_sin`, `mois_cos` | Saisonnalité (cyclique) |

### Features de lag (1, 2, 3 mois)
| Feature | Description |
|---------|-------------|
| `nb_correctif_lag1/2/3` | Nb de correctifs il y a 1/2/3 mois |
| `duree_lag1/2/3` | Durée totale d'arrêt il y a 1/2/3 mois |
| `rolling_nb_correctif_3m` | Somme glissante des correctifs sur 3 mois |
| `rolling_correctif_trend` | Tendance : lag1 - lag3 |
| `rolling_duree_3m` | Moyenne glissante des durées |

### Features PostgreSQL (quand disponible — pas encore actif)
Âge équipement, nb préventif réalisé/en retard, taux respect planning, taux dispo, stock pièces

---

## 3. Entraînement

### Algorithme
**XGBoost Classifier** avec GridSearchCV

### Grille d'hyperparamètres
| Paramètre | Valeurs testées |
|-----------|----------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5 |
| `learning_rate` | 0.05, 0.1 |
| `subsample` | 0.8, 1.0 |
| `colsample_bytree` | 0.8, 1.0 |
| `scale_pos_weight` | 1, ratio déséquilibre |

### Validation
- **Split temporel** : entraînement 2020-2023, test 2025
- **TimeSeriesSplit** (3 folds) pour la recherche d'hyperparamètres
- Année 2024 exclue (trou dans l'historique)

---

## 4. Résultats du dernier entraînement

```
Source        : Excel (PostgreSQL pas encore fusionné)
Modèle        : XGBClassifier
AUC-ROC       : 0.8724
Precision     : 0.8250  (quand on prédit correctif, on a raison 82% du temps)
Recall        : 0.5789  (on détecte 58% des vrais correctifs)
F1-score      : 0.6804

Correctif / Total :
  Train : 157 / 370 (42.4%)
  Test  : 57 / 108 (52.8%)

Faux positifs  : 7  (alertes inutiles)
Faux négatifs  : 24 (pannes non détectées)
```

### Interprétation
- **AUC 0.87** → nette amélioration vs l'ancien modèle (0.73)
- **Recall à 58%** → on rate encore 42% des correctifs réels, mais c'est un bon point de départ
- **Faux positifs : 7** → seulement 7 alertes inutiles sur 108 mois-test
- Le modèle est **exploitable en indicateur de tendance** mais pas encore en pilotage automatique

---

## 5. Seuils de risque

| Niveau | Seuil | Action |
|--------|-------|--------|
| 🔴 ÉLEVÉ | ≥ 0.60 | Planifier maintenance préventive |
| 🟠 MODÉRÉ | ≥ 0.38 | Surveiller de près |
| 🟢 FAIBLE | < 0.38 | Aucune action requise |

*Les seuils sont encore arbitraires (0.38, 0.60). Ils pourront être optimisés avec plus de données.*

---

## 6. Service Flask

### Lancement

```bash
cd innofaso-ia
python ia_service_complet.py
```

Puis ouvrir http://localhost:5001

### Clé API Groq (optionnelle)

Pour activer la classification LLM et le chatbot avancé :

1. Va sur https://console.groq.com/keys
2. Copie ta clé (`gsk_...`)
3. Colle-la dans `innofaso-ia/.env` :
   ```
   GROQ_API_KEY=gsk_ta_cle_ici
   ```
4. Relance le service

Sans clé, la classification utilise TF-IDF + SGD (8 catégories, fiable) et le bot répond en mode règles.

### Endpoints

| Route | Méthode | Description |
|-------|---------|-------------|
| `GET /` | - | Interface web de test (prédictions + NLP + Chat) |
| `GET /api/health` | GET | Santé du service |
| `GET /api/equipements` | GET | Liste des équipements |
| `GET /api/predict/{annee}/{mois}` | GET | Prédictions pour tous les équipements |
| `GET /api/predict/{annee}/{mois}/{equipement}` | GET | Prédiction pour un équipement |
| `POST /api/classify` | POST | Classifier une cause de panne |
| `POST /api/classify/batch` | POST | Classification par lot |
| `POST /api/bot/chat` | POST | Assistant maintenance conversationnel |

### Exemples

```bash
# Classifier une cause
curl -X POST http://localhost:5001/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Fuite d huile sur le verin"}'

# Parler au bot
curl -X POST http://localhost:5001/api/bot/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quel est le risque pour Filling machine ?"}'
```

---

## 7. Modules

| Fichier | Rôle |
|---------|------|
| `ia_service_complet.py` | Service Flask (prédictions + NLP + Bot) |
| `nlp_service.py` | Classification NLP (TF-IDF + SGD / Groq LLM) |
| `db_loader.py` | Chargement unifié PostgreSQL / Excel |
| `train_model.py` | Entraînement XGBoost + évaluation |
| `seed_base.py` | Injection des données Excel dans PostgreSQL |

---

## 8. Prochaines étapes

1. **Fusionner PostgreSQL** : uniformiser les noms d'équipements entre Excel et la base
2. **Prédiction durée d'arrêt** (régression)
3. **Détection d'anomalies** (Isolation Forest)
4. **Dashboard temps réel** des prédictions
