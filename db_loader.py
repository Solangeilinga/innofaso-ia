"""
InnoFaso — Chargement des données pour le ML
Mode hybride :
  - Base : Excel historique (cible `target` fiable, 416 correctifs)
  - Enrichissement : PostgreSQL (âge, stock, disponibilité) quand dispo
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'INNOFASO_Historique_de_maintenance_2025.xlsx')
SHEET_NAME = 'T_Maint_Innofaso'

EXCEL_COLUMNS = ['Partenaire', 'Annee', 'Mois', 'Equipement', 'Sous_equip',
                 'Type_maint', 'Duree_h', 'Description']

EQUIPEMENT_EN_FR = {
    'Air compressor':                'Compresseur air',
    'Carton sealer':                 'Soudeuse carton',
    'Conveyor':                      'Bande transporteuse',
    'Filling machine':               'Conditionneuse',
    'Finished Product Transfer Pump':'Pompe transfert PF',
    'Grinder - Vibroreactor':        'Vibroreacteur',
    'Main electrical panel':         'Armoire electrique principale',
    'Mixer':                         'Melangeur',
    'Nitrogen generator - Filter':   'Generateur azote / Filtre',
    'Oil tank - Oil melting tank':   'Fondoir a huile',
    'Others equipments':             'Autres equipements',
    'Packing machine':               'Emballeuse',
    'Pre-mixing hopper':             'Tremie de premelange',
    'Premixing heat treatment':      'Traitement thermique premelange',
    'Sachet Printer - Carton Printer':'Imprimeuse sachet / carton',
}

EQUIPEMENT_FR_EN = {v: k for k, v in EQUIPEMENT_EN_FR.items()}

MOIS_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    'jan': 1, 'jun': 6,
}

ML_VIEW = 'v_ml_maintenance_mensuel'


def get_db_conn():
    try:
        import psycopg2
        DATABASE_URL = os.getenv('DATABASE_URL')
        if DATABASE_URL:
            return psycopg2.connect(DATABASE_URL)
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            dbname=os.getenv('DB_NAME', 'innopro'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
        )
    except ImportError:
        return None
    except Exception as e:
        print(f"   [WARN] Connexion PostgreSQL impossible : {e}")
        return None


def charger_depuis_postgres():
    conn = get_db_conn()
    if conn is None:
        return None
    try:
        query = f"""
            SELECT
                equipement_nom,
                annee::int,
                mois_num::int,
                age_equipement_ans,
                nb_correctif::int,
                duree_arret_total_h,
                duree_arret_moyenne_h,
                duree_maintenance_total_h,
                duree_maintenance_moyenne_h,
                nb_preventif_planifie::int,
                nb_preventif_realise::int,
                nb_preventif_en_retard::int,
                taux_respect_preventif_pct,
                nb_interv_quart::int,
                duree_arret_quart_h,
                taux_disponibilite_moyen,
                duree_arret_ligne_h,
                taux_dispo_ligne,
                nb_quarts_planifies::int,
                nb_pieces_reference::int,
                stock_total_pieces::int,
                nb_pieces_sous_seuil::int
            FROM {ML_VIEW}
            ORDER BY equipement_nom, annee, mois_num
        """
        df = pd.read_sql(query, conn)
        df['equipement_nom'] = df['equipement_nom'].map(EQUIPEMENT_EN_FR).fillna(df['equipement_nom'])
        print(f"   [OK] PostgreSQL : {len(df)} lignes chargées depuis {ML_VIEW}")
        return df
    except Exception as e:
        print(f"   [WARN] Erreur PostgreSQL : {e}")
        return None
    finally:
        conn.close()


def charger_depuis_excel():
    if not os.path.exists(DATA_PATH):
        print(f"   [ERR] Excel introuvable : {DATA_PATH}")
        return None

    df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)
    df_raw.columns = EXCEL_COLUMNS

    df = df_raw.dropna(subset=['Annee']).copy()
    df['Annee'] = df['Annee'].astype(int)
    df['Duree_h'] = pd.to_numeric(df['Duree_h'], errors='coerce').fillna(0)
    df['Mois_num'] = df['Mois'].map(MOIS_MAP)
    df = df.dropna(subset=['Mois_num'])
    df['Mois_num'] = df['Mois_num'].astype(int)
    df = df[df['Type_maint'].isin(['Preventive', 'Corrective'])]

    hist = df.groupby(['Annee', 'Mois_num', 'Equipement']).agg(
        nb_interventions=('Duree_h', 'count'),
        duree_totale_h=('Duree_h', 'sum'),
        duree_max_h=('Duree_h', 'max'),
        nb_correctif=('Type_maint', lambda x: (x == 'Corrective').sum()),
    ).reset_index()
    hist['target'] = (hist['nb_correctif'] > 0).astype(int)
    hist.rename(columns={
        'Equipement': 'equipement_nom',
        'Annee': 'annee',
        'Mois_num': 'mois_num',
    }, inplace=True)

    hist['equipement_nom'] = hist['equipement_nom'].map(EQUIPEMENT_EN_FR).fillna(hist['equipement_nom'])

    n_corr = hist['target'].sum()
    print(f"   [OK] Excel : {len(hist)} lignes agrégées ({int(n_corr)} correctifs) — noms traduits en français")
    return hist


def enrichir_avec_postgres(df_excel, df_pg):
    cols_pg = [
        'age_equipement_ans',
        'duree_arret_total_h', 'duree_arret_moyenne_h',
        'duree_maintenance_total_h', 'duree_maintenance_moyenne_h',
        'nb_preventif_planifie', 'nb_preventif_realise', 'nb_preventif_en_retard',
        'taux_respect_preventif_pct',
        'nb_interv_quart', 'duree_arret_quart_h', 'taux_disponibilite_moyen',
        'duree_arret_ligne_h', 'taux_dispo_ligne',
        'nb_quarts_planifies',
        'nb_pieces_reference', 'stock_total_pieces', 'nb_pieces_sous_seuil',
    ]

    df = df_excel.merge(
        df_pg[['equipement_nom', 'annee', 'mois_num'] + cols_pg],
        on=['equipement_nom', 'annee', 'mois_num'],
        how='left',
    )

    for col in cols_pg:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    lignes_avec_enrichissement = df[cols_pg].sum().sum()
    if lignes_avec_enrichissement > 0:
        print(f"   [OK] Enrichi avec PostgreSQL : {cols_pg}")
    else:
        print(f"   [WARN] Aucun enrichissement PostgreSQL trouvé (colonnes à 0)")
    return df


def charger_donnees(source='auto'):
    """
    Stratégie :
    1. Charger l'Excel (base historique fiable pour la cible)
    2. Si PostgreSQL disponible, fusionner les features enrichies
    3. Retourne un DataFrame avec target + features PostgreSQL ou Excel
    """
    df_excel = charger_depuis_excel()
    if df_excel is None:
        raise RuntimeError("Fichier Excel introuvable — source de données requise")

    if source == 'excel':
        return df_excel

    if source in ('auto', 'hybrid'):
        df_pg = charger_depuis_postgres()
        if df_pg is not None:
            df = enrichir_avec_postgres(df_excel, df_pg)
            df['source'] = 'hybrid'
            return df
        print("   [WARN] Aucune donnée PostgreSQL — fallback Excel uniquement")

    df_excel['source'] = 'excel'
    return df_excel
