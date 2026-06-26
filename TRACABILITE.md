# Traçabilité — Implémentation IA v2

## Objectif

Reconstruire le système de prédiction de maintenance InnoFaso avec des données enrichies (PostgreSQL), un modèle plus performant (XGBoost), et des services IA supplémentaires.

---

## Étapes

### [x] Phase 1.1 — Analyse du modèle existant
- **Fait le :** 26/06/2026
- **Détail :** Analyse complète du modèle régression logistique (AUC=0.73, 22 features, données excel)
- **Livrable :** `report.md`

### [x] Phase 1.2 — Création de la vue matérialisée ML
- **Fait le :** 26/06/2026
- **Fichier :** `innofaso_backend/migrations/sql/029_vue_ml_materialized.sql`
- **Tables consolidées :** `equipements`, `maintenance_corrective`, `signalements_pannes`, `plannings_maintenance`, `intervention_ligne`, `intervention_quart`, `pieces_rechange`, `ligne_production`, `planning_semaine`, `planning_quart`
- **Commande :** `npm run migrate`
- **Résultat :** ✅ Vue créée avec 474 lignes de données consolidées sur 15 équipements × ~31 mois

### [x] Phase 2.1 — Pipeline d'entraînement XGBoost
- **Fait le :** 26/06/2026
- **Fichier :** `innofaso-ia/train_model.py`
- **Base :** PostgreSQL (vue matérialisée) avec repli Excel
- **Modules ajoutés :** `db_loader.py` (chargement unifié PostgreSQL/Excel), `.env` (configuration DB)
- **Features enrichies :** âge équipement, préventif, disponibilité, stock pièces, lags correctifs, rolling windows
- **Algo :** XGBoost Classifier avec GridSearchCV + TimeSeriesSplit
- **Validation :** Métriques globales + par équipement, matrice de confusion, feature importance

### [x] Phase 2.2 — Mise à jour du service Flask
- **Fait le :** 26/06/2026
- **Fichier :** `innofaso-ia/ia_service_complet.py`
- **Changement :** Chargement depuis PostgreSQL (via `db_loader.py`) avec repli Excel automatique
- **Version service :** 2.0
- **Nouvelles features exposées :** âge équipement, taux dispo, stock, respect planning préventif

### [ ] Phase 3.x — À définir
- Classification NLP des pannes
- Prédiction durée d'arrêt
- Détection d'anomalies
