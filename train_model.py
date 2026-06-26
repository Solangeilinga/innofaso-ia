"""
InnoFaso — Pipeline d'entraînement du modèle de prédiction de pannes
v2 : XGBoost avec évaluation rigoureuse
Lecture depuis PostgreSQL (vue matérialisée) avec repli Excel.

Usage : python train_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

import db_loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
TEST_YEARS = [2025]
VAL_YEARS = [2023]

LAG_MONTHS = [1, 2, 3]

FEATURES_POSTGRES = [
    'age_equipement_ans',
    'nb_preventif_planifie', 'nb_preventif_realise', 'nb_preventif_en_retard',
    'taux_respect_preventif_pct',
    'taux_disponibilite_moyen', 'taux_dispo_ligne',
    'nb_pieces_reference', 'stock_total_pieces', 'nb_pieces_sous_seuil',
    'duree_arret_moyenne_h', 'duree_maintenance_moyenne_h',
]

FEATURES_EXCEL = [
    'duree_totale_h', 'duree_max_h',
]

FEATURES_CYCLIC = ['mois_sin', 'mois_cos']

def get_lag_value(hist_eq, annee_cible, mois_cible, lag, col):
    m = mois_cible - lag
    a = annee_cible
    while m <= 0:
        m += 12
        a -= 1
    match = hist_eq[(hist_eq['annee'] == a) & (hist_eq['mois_num'] == m)]
    if len(match) > 0:
        val = match.iloc[0][col]
        return float(val) if pd.notna(val) else 0.0
    return 0.0

def _v(val, default=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

def generer_features(df, equipements, le_eq, source='postgres'):
    print("\n[3/9] Génération des features...")
    all_rows = []

    for eq in equipements:
        hist_eq = df[df['equipement_nom'] == eq].sort_values(['annee', 'mois_num']).copy()

        for _, row in hist_eq.iterrows():
            annee_cible = int(row['annee'])
            mois_cible = int(row['mois_num'])
            target = int(row['target'])

            features = {
                'equipement': eq,
                'equipement_enc': int(le_eq.transform([eq])[0]),
                'annee': annee_cible,
                'mois_num': mois_cible,
                'trimestre': (mois_cible - 1) // 3 + 1,
                'mois_sin': np.sin(2 * np.pi * mois_cible / 12),
                'mois_cos': np.cos(2 * np.pi * mois_cible / 12),
            }

            use_pg_features = source in ('postgres', 'hybrid')
            if use_pg_features:
                features['age_equipement_ans'] = float(_v(row.get('age_equipement_ans'), 0))
                features['nb_preventif_planifie'] = int(_v(row.get('nb_preventif_planifie'), 0))
                features['nb_preventif_realise'] = int(_v(row.get('nb_preventif_realise'), 0))
                features['nb_preventif_en_retard'] = int(_v(row.get('nb_preventif_en_retard'), 0))
                features['taux_respect_preventif_pct'] = float(_v(row.get('taux_respect_preventif_pct'), 0))
                features['taux_disponibilite_moyen'] = float(_v(row.get('taux_disponibilite_moyen'), 100))
                features['taux_dispo_ligne'] = float(_v(row.get('taux_dispo_ligne'), 100))
                features['nb_pieces_reference'] = int(_v(row.get('nb_pieces_reference'), 0))
                features['stock_total_pieces'] = int(_v(row.get('stock_total_pieces'), 0))
                features['nb_pieces_sous_seuil'] = int(_v(row.get('nb_pieces_sous_seuil'), 0))
                features['duree_arret_moyenne_h'] = float(_v(row.get('duree_arret_moyenne_h'), 0))
                features['duree_maintenance_moyenne_h'] = float(_v(row.get('duree_maintenance_moyenne_h'), 0))
            else:
                features['duree_totale_h'] = float(row.get('duree_totale_h', 0))
                features['duree_max_h'] = float(row.get('duree_max_h', 0))

            use_pg_features = source in ('postgres', 'hybrid')
            for lag in LAG_MONTHS:
                features[f'nb_correctif_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'nb_correctif')
                if use_pg_features:
                    features[f'duree_arret_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'duree_arret_total_h')
                    features[f'nb_interv_quart_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'nb_interv_quart')
                else:
                    features[f'duree_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'duree_totale_h')

            nb_corr_lags = sum(features.get(f'nb_correctif_lag{lag}', 0) for lag in LAG_MONTHS)
            features['rolling_nb_correctif_3m'] = nb_corr_lags
            features['rolling_correctif_trend'] = (
                features.get('nb_correctif_lag1', 0) - features.get('nb_correctif_lag3', 0)
            )

            if use_pg_features:
                duree_lags = sum(features.get(f'duree_arret_lag{lag}', 0) for lag in LAG_MONTHS)
                features['rolling_duree_arret_3m'] = duree_lags / len(LAG_MONTHS)
            else:
                duree_lags = sum(features.get(f'duree_lag{lag}', 0) for lag in LAG_MONTHS)
                features['rolling_duree_3m'] = duree_lags / len(LAG_MONTHS)

            features['target'] = target
            all_rows.append(features)

    return pd.DataFrame(all_rows)


def get_feature_cols(source):
    use_pg_features = source in ('postgres', 'hybrid')
    lags_corr = [f'nb_correctif_lag{lag}' for lag in LAG_MONTHS]
    lags_extra = [f'duree_arret_lag{lag}' for lag in LAG_MONTHS] if use_pg_features else [f'duree_lag{lag}' for lag in LAG_MONTHS]
    lags_interv = [f'nb_interv_quart_lag{lag}' for lag in LAG_MONTHS] if use_pg_features else []

    base = ['equipement_enc', 'mois_num', 'annee', 'trimestre'] + FEATURES_CYCLIC
    static = FEATURES_POSTGRES if use_pg_features else FEATURES_EXCEL
    rolling = ['rolling_nb_correctif_3m', 'rolling_correctif_trend']
    rolling += ['rolling_duree_arret_3m'] if use_pg_features else ['rolling_duree_3m']

    return base + static + lags_corr + lags_extra + lags_interv + rolling


def train_test_split_temporel(df_features, feature_cols):
    train_mask = ~df_features['annee'].isin(TEST_YEARS + [2024])
    test_mask = df_features['annee'].isin(TEST_YEARS)

    X_train = df_features[train_mask][feature_cols]
    y_train = df_features[train_mask]['target']
    X_test = df_features[test_mask][feature_cols]
    y_test = df_features[test_mask]['target']

    return X_train, y_train, X_test, y_test, train_mask, test_mask


def train_xgboost(X_train, y_train):
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, scale_pos],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    xgb_base = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
    )

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print(f"   Meilleurs paramètres : {grid.best_params_}")
    print(f"   Meilleur AUC-ROC (CV) : {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluer_modele(model, X_test, y_test, feature_cols):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    n_classes = cm.shape[0]
    if n_classes == 1:
        if list(y_test.unique()) == [0]:
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, cm[0, 0]]])

    print(f"\n   Accuracy  : {accuracy:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-score  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n   Matrice de confusion :")
    print(f"       TN={cm[0,0]:4d}   FP={cm[0,1]:4d}")
    print(f"       FN={cm[1,0]:4d}   TP={cm[1,1]:4d}")

    print(f"\n   Classification report :")
    for line in classification_report(y_test, y_pred, target_names=['Preventif', 'Correctif'],
                                       zero_division=0, labels=[0, 1]).split('\n'):
        print(f"   {line}")

    return y_prob, y_pred, {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': round(float(auc), 4),
    }, cm


def evaluer_par_equipement(df_test, y_prob, y_pred, X_test_index, equipements):
    print(f"\n[7/9] Métriques par équipement :")
    equip_metrics = []
    col_eq = 'equipement_nom' if 'equipement_nom' in df_test.columns else 'equipement'

    for eq in equipements:
        eq_mask = df_test[col_eq] == eq
        if eq_mask.sum() < 3:
            continue
        y_eq_true = df_test.loc[eq_mask, 'target']
        eq_idx = df_test.loc[eq_mask].index
        y_eq_pred_proba = pd.Series(y_prob, index=X_test_index).loc[eq_idx]
        y_eq_pred_class = pd.Series(y_pred, index=X_test_index).loc[eq_idx]

        if y_eq_true.nunique() < 2:
            continue

        try:
            eq_auc = roc_auc_score(y_eq_true, y_eq_pred_proba)
            eq_recall = recall_score(y_eq_true, y_eq_pred_class, zero_division=0)
            eq_precision = precision_score(y_eq_true, y_eq_pred_class, zero_division=0)
            eq_f1 = f1_score(y_eq_true, y_eq_pred_class, zero_division=0)
            n_correctif = int(y_eq_true.sum())
            n_total = len(y_eq_true)

            equip_metrics.append({
                'equipement': eq,
                'n_total': n_total,
                'n_correctif': n_correctif,
                'precision': round(eq_precision, 3),
                'recall': round(eq_recall, 3),
                'f1': round(eq_f1, 3),
                'auc': round(eq_auc, 3),
            })

            print(f"   {eq:35s} AUC={eq_auc:.3f}  Prec={eq_precision:.3f}  Recall={eq_recall:.3f}  F1={eq_f1:.3f}  ({n_correctif}/{n_total})")
        except Exception:
            continue

    return equip_metrics


def afficher_feature_importance(model, feature_cols):
    print(f"\n[8/9] Feature Importance :")
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"   Top 15 features :")
        for _, row in fi.head(15).iterrows():
            print(f"     {row['feature']:30s} : {row['importance']:.4f}")
        return fi
    return None


def sauvegarder(model, le_eq, feature_cols, metrics, equip_metrics, train_years, test_years,
                X_train, X_test, cm, source, grid_best_params=None):
    print(f"\n[9/9] Sauvegarde...")

    model_path = os.path.join(BASE_DIR, 'modele_panne_innofaso.pkl')
    le_path = os.path.join(BASE_DIR, 'label_encoder_equipement.pkl')
    features_path = os.path.join(BASE_DIR, 'features_liste.pkl')
    metrics_path = os.path.join(BASE_DIR, 'metrics.json')

    joblib.dump(model, model_path)
    print(f"   Modèle sauvegardé : {model_path}")

    joblib.dump(le_eq, le_path)
    joblib.dump(feature_cols, features_path)
    print(f"   Label encoder + features sauvegardés")

    METRICS = {
        'date_entrainement': datetime.now().isoformat(),
        'modele': type(model).__name__,
        'params': str(model.get_params()),
        'hyperparams_optimaux': str(grid_best_params),
        'source_donnees': source,
        'split_temporel': {
            'train_annees': train_years,
            'test_annees': test_years,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        },
        'metriques_globales': metrics,
        'matrice_confusion': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
        },
        'metriques_par_equipement': sorted(equip_metrics, key=lambda x: x['auc'], reverse=True),
    }

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(METRICS, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"   Métriques sauvegardées : {metrics_path}")

    return METRICS


def main():
    print("=" * 55)
    print("   InnoFaso — Pipeline d'entraînement v2")
    print("=" * 55)

    # 1. Chargement des données
    print("\n[1/9] Chargement des données...")
    try:
        df = db_loader.charger_donnees(source='auto')
    except RuntimeError as e:
        print(f"   [ERR] {e}")
        sys.exit(1)

    has_pg_features = 'age_equipement_ans' in df.columns
    source = 'hybrid' if has_pg_features and df['age_equipement_ans'].sum() > 0 else 'excel'
    source_label = {'hybrid': 'Excel + PostgreSQL (hybride)', 'excel': 'Excel (fallback)'}
    print(f"   Source : {source_label.get(source, source)}")
    print(f"   Lignes : {len(df)}")
    print(f"   Équipements : {df['equipement_nom'].nunique()}")
    print(f"   Target=1 (correctif) : {df['target'].sum()} / {len(df)} ({df['target'].mean()*100:.1f}%)")
    print(f"   Période : {int(df['annee'].min())}-{int(df['annee'].max())}")

    # 2. Encodage des équipements
    print("\n[2/9] Encodage des équipements...")
    equipements = sorted(df['equipement_nom'].unique())
    le_eq = LabelEncoder()
    le_eq.fit(equipements)
    print(f"   {len(equipements)} équipements encodés")

    # 3. Génération des features
    df_features = generer_features(df, equipements, le_eq, source=source)
    feature_cols = get_feature_cols(source)
    print(f"   {len(df_features)} lignes, {len(feature_cols)} features générées")

    # 4. Split temporel
    print("\n[4/9] Split temporel...")
    X_train, y_train, X_test, y_test, train_mask, test_mask = train_test_split_temporel(df_features, feature_cols)

    train_years = sorted(df_features[train_mask]['annee'].unique())
    test_years = sorted(df_features[test_mask]['annee'].unique())
    print(f"   Train : {len(X_train)} lignes ({train_years})")
    print(f"   Test  : {len(X_test)} lignes ({test_years})")
    print(f"   Target=1 train : {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"   Target=1 test  : {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

    # 5. Entraînement XGBoost
    print("\n[5/9] Recherche des meilleurs hyperparamètres XGBoost...")
    model = train_xgboost(X_train, y_train)

    # 6. Évaluation
    print("\n[6/9] Évaluation sur le test set...")
    y_prob, y_pred, metrics, cm = evaluer_modele(model, X_test, y_test, feature_cols)

    # 7. Métriques par équipement
    df_test = df_features[test_mask]
    equip_metrics = evaluer_par_equipement(df_test, y_prob, y_pred, X_test.index, equipements)

    # 8. Feature importance
    fi = afficher_feature_importance(model, feature_cols)

    # 9. Sauvegarde
    METRICS = sauvegarder(
        model, le_eq, feature_cols, metrics, equip_metrics,
        train_years, test_years, X_train, X_test, cm, source,
        grid_best_params=model.get_params()
    )

    # Résumé final
    print("\n" + "=" * 55)
    print("   RÉSUMÉ FINAL")
    print("=" * 55)
    src_label = {'hybrid': 'Excel + PostgreSQL (hybride)', 'postgres': 'PostgreSQL', 'excel': 'Excel'}
    print(f"""
   Source        : {src_label.get(source, source)}
   Modèle        : {type(model).__name__}
   AUC-ROC (test): {metrics['auc_roc']:.4f}
   Precision     : {metrics['precision']:.4f}
   Recall        : {metrics['recall']:.4f}
   F1-score      : {metrics['f1_score']:.4f}

   Correctif / Total :
     Train : {int(y_train.sum())} / {len(y_train)} ({y_train.mean()*100:.1f}%)
     Test  : {int(y_test.sum())} / {len(y_test)} ({y_test.mean()*100:.1f}%)

   Faux positifs : {int(cm[0, 1])} (alertes inutiles)
   Faux négatifs : {int(cm[1, 0])} (pannes non détectées)
""")


if __name__ == '__main__':
    import sys
    main()
