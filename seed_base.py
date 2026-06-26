"""
Seed la base PostgreSQL avec les données de l'Excel historique
pour activer l'entraînement hybride (Excel + PostgreSQL).

Usage : python seed_base.py
"""

import os
import sys
import uuid
from datetime import datetime, date, timedelta
from calendar import monthrange

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ── Connexion DB ─────────────────────────────────────────────
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    conn = psycopg2.connect(DATABASE_URL)
else:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'innopro'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', ''),
    )
conn.autocommit = False
cur = conn.cursor()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'INNOFASO_Historique_de_maintenance_2025.xlsx')

MOIS_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    'jan': 1, 'jun': 6,
}

# ── IDs existants ────────────────────────────────────────────
ADMIN_ID = 'fba3654e-c170-476f-94f7-cc2332a3a660'
LIGNE_IDS = {
    'L1': 'c3e56289-afb2-44b8-b9da-be7deda29242',
    'L2': '690168c6-a287-4824-83fe-a1d65d809400',
    'L4': '9afe84bb-e050-48b0-b249-56dbfb4e4483',
}
QUART_IDS = ['13a55b9b-1a13-403d-a073-debbd31c93d0',
             '697de1ab-6505-42f4-bd3c-81fdc7115fc7',
             'e34debf3-1ba3-4e75-9922-35f892001e41']

# ── Traduction EN → FR ────────────────────────────────────────
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

# ── Mapping équipements Excel → DB ────────────────────────────
EQUIPEMENT_MAP = {v: None for v in EQUIPEMENT_EN_FR.values()}

TYPE_MAP = {
    'Corrective': 'correctif',
    'Preventive': 'preventif',
    'Improvement': 'preventif',
}


def get_or_create_equipement(nom):
    if nom in EQUIPEMENT_MAP and EQUIPEMENT_MAP[nom] is not None:
        return EQUIPEMENT_MAP[nom]
    eq_id = str(uuid.uuid4())
    code = nom.upper().replace(' ', '_')[:20].replace('-', '_')
    cur.execute("""
        INSERT INTO equipements (id, code_ref, nom, actif, date_installation)
        VALUES (%s, %s, %s, TRUE, '2020-01-01')
        ON CONFLICT (code_ref) DO UPDATE SET nom = EXCLUDED.nom
        RETURNING id
    """, (eq_id, code, nom))
    eid = cur.fetchone()[0]
    EQUIPEMENT_MAP[nom] = eid
    return eid


def get_or_create_planning_semaine(annee, mois):
    premiere_semaine = date(annee, mois, 1).isocalendar()[1]
    debut = date(annee, 1, 1) + timedelta(weeks=premiere_semaine - 1)
    fin = debut + timedelta(days=6)
    cur.execute("""
        SELECT id FROM planning_semaine
        WHERE annee = %s AND mois = %s AND semaine_num = %s
        LIMIT 1
    """, (annee, mois, premiere_semaine))
    row = cur.fetchone()
    if row:
        return row[0]
    ps_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO planning_semaine (id, ligne_id, date_debut_semaine, date_fin_semaine,
                                      semaine_num, annee, mois, admin_id, statut)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'VALIDE')
    """, (ps_id, LIGNE_IDS['L1'], debut, fin, premiere_semaine, annee, mois, ADMIN_ID))
    return ps_id


def get_or_create_planning_jour(planning_semaine_id, jour_date):
    cur.execute("""
        SELECT id FROM planning_jour
        WHERE planning_semaine_id = %s AND date_jour = %s
        LIMIT 1
    """, (planning_semaine_id, jour_date))
    row = cur.fetchone()
    if row:
        return row[0]
    pj_id = str(uuid.uuid4())
    jour_semaine = jour_date.strftime('%A')
    cur.execute("""
        INSERT INTO planning_jour (id, planning_semaine_id, date_jour, jour_semaine)
        VALUES (%s, %s, %s, %s)
    """, (pj_id, planning_semaine_id, jour_date, jour_semaine))
    return pj_id


def get_or_create_planning_quart(planning_jour_id):
    cur.execute("""
        SELECT id FROM planning_quart
        WHERE planning_jour_id = %s LIMIT 1
    """, (planning_jour_id,))
    row = cur.fetchone()
    if row:
        return row[0]
    pq_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO planning_quart (id, planning_jour_id, quart_id, maintenancier_id)
        VALUES (%s, %s, %s, %s)
    """, (pq_id, planning_jour_id, QUART_IDS[0], ADMIN_ID))
    return pq_id


def create_signalement_panne(equipement_id):
    sp_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO signalements_pannes (id, signale_par_id, assigne_a_id,
                                          observation, date_panne, statut)
        VALUES (%s, %s, %s, 'Seed automatique depuis Excel', %s, 'confirme')
    """, (sp_id, ADMIN_ID, ADMIN_ID, date.today()))
    return sp_id


def seed_donnees():
    print("=" * 55)
    print("  InnoFaso — Seed des données Excel dans PostgreSQL")
    print("=" * 55)

    # 1. Charger l'Excel
    print("\n[1/4] Chargement de l'Excel...")
    df_raw = pd.read_excel(DATA_PATH, sheet_name='T_Maint_Innofaso', header=0)
    df_raw.columns = ['Partenaire', 'Annee', 'Mois_text', 'Equipement',
                      'Sous_equip', 'Type_maint', 'Duree_h', 'Description']
    df = df_raw.dropna(subset=['Annee']).copy()
    df['Annee'] = df['Annee'].astype(int)
    df['Mois_num'] = df['Mois_text'].map(MOIS_MAP)
    df = df.dropna(subset=['Mois_num'])
    df['Mois_num'] = df['Mois_num'].astype(int)
    df = df[df['Type_maint'].isin(['Corrective', 'Preventive', 'Improvement'])]
    df['Duree_h'] = pd.to_numeric(df['Duree_h'], errors='coerce').fillna(0)
    print(f"   {len(df)} lignes chargées")

    # Traduire les noms d'équipements en français
    df['Equipement'] = df['Equipement'].map(EQUIPEMENT_EN_FR).fillna(df['Equipement'])

    # 2. Créer / récupérer les équipements
    print("\n[2/4] Création des équipements...")
    equipements = df['Equipement'].unique()
    for eq in equipements:
        get_or_create_equipement(eq)
    print(f"   {len(equipements)} équipements dans la base")

    # 3. Insérer les maintenances correctives et préventives
    print("\n[3/4] Insertion des données...")
    count_correctif = 0
    count_preventif = 0
    count_signalements = 0
    count_skip = 0
    batch_size = 50
    batch = []

    for _, row in df.iterrows():
        annee = int(row['Annee'])
        mois = int(row['Mois_num'])
        equipement = row['Equipement']
        type_maint = row['Type_maint']
        duree = float(row['Duree_h'])
        cause = str(row['Description']) if pd.notna(row['Description']) else ''
        equipement_id = get_or_create_equipement(equipement)

        jour = min(15, monthrange(annee, mois)[1])
        jour_date = date(annee, mois, jour)

        try:
            ps_id = get_or_create_planning_semaine(annee, mois)
            pj_id = get_or_create_planning_jour(ps_id, jour_date)
            pq_id = get_or_create_planning_quart(pj_id)
        except Exception as e:
            count_skip += 1
            continue

        mc_id = str(uuid.uuid4())
        timestamp = datetime(annee, mois, jour, 8, 0, 0)

        if type_maint == 'Corrective':
            sp_id = create_signalement_panne(equipement_id)
            count_signalements += 1
            batch.append((
                mc_id, ps_id, equipement_id, None,
                ADMIN_ID, None, None, None, sp_id,
                duree, duree, cause, '',
                'termine', timestamp, timestamp
            ))
            count_correctif += 1
        else:
            batch.append((
                mc_id, ps_id, equipement_id, None,
                ADMIN_ID, None, None, None, None,
                0, duree, cause, '',
                'termine', timestamp, timestamp
            ))
            count_preventif += 1

        if len(batch) >= batch_size:
            execute_values(cur, """
                INSERT INTO maintenance_corrective
                (id, planning_semaine_id, equipement_id, equipement_libre,
                 executeur_id, co_executeur_id, verificateur_id, validateur_id,
                 signalement_panne_id, duree_arret, duree_maintenance, cause,
                 observations, statut, cree_le, modifie_le)
                VALUES %s
            """, batch, page_size=batch_size)
            batch = []

    if batch:
        execute_values(cur, """
            INSERT INTO maintenance_corrective
            (id, planning_semaine_id, equipement_id, equipement_libre,
             executeur_id, co_executeur_id, verificateur_id, validateur_id,
             signalement_panne_id, duree_arret, duree_maintenance, cause,
             observations, statut, cree_le, modifie_le)
            VALUES %s
        """, batch, page_size=batch_size)

    # 4. Insérer les plannings_maintenance (préventif)
    print("\n[4/4] Insertion des plannings préventifs...")
    count_pm = 0
    pm_batch = []
    preventif_rows = df[df['Type_maint'].isin(['Preventive', 'Improvement'])]

    for _, row in preventif_rows.iterrows():
        annee = int(row['Annee'])
        mois = int(row['Mois_num'])
        equipement = row['Equipement']
        equipement_id = get_or_create_equipement(equipement)
        jour = min(15, monthrange(annee, mois)[1])
        date_prevue = date(annee, mois, jour)

        pm_id = str(uuid.uuid4())
        pm_batch.append((
            pm_id, equipement_id, ADMIN_ID,
            date_prevue, None, 'REALISE', '',
            datetime(annee, mois, jour, 8, 0, 0)
        ))
        count_pm += 1

    FORMULAIRE_PREV_ID = 'aebc2cbb-c312-4a24-8bb6-f46bec21f0a2'
    pm_batch2 = []
    for pm in pm_batch:
        pm_batch2.append((
            pm[0], FORMULAIRE_PREV_ID, pm[1], pm[2],
            pm[3], pm[4], pm[5], pm[6], pm[7]
        ))
    if pm_batch2:
        execute_values(cur, """
            INSERT INTO plannings_maintenance
            (id, formulaire_type_id, equipement_id, technicien_id,
             date_prevue, date_realisee, statut, commentaire, cree_le)
            VALUES %s
        """, pm_batch2, page_size=100)

    # Commit
    conn.commit()
    print(f"\n   Résumé :")
    print(f"     Correctifs insérés : {count_correctif}")
    print(f"     Préventifs insérés  : {count_preventif}")
    print(f"     Signalements créés  : {count_signalements}")
    print(f"     Plannings prév.     : {count_pm}")
    print(f"     Lignes ignorées     : {count_skip}")
    print(f"\n   ✅ Base mise à jour avec succès !")

    # Refresh de la vue matérialisée
    cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY v_ml_maintenance_mensuel")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM v_ml_maintenance_mensuel")
    n = cur.fetchone()[0]
    print(f"   🔄 Vue matérialisée rafraîchie : {n} lignes")


if __name__ == '__main__':
    try:
        seed_donnees()
    except Exception as e:
        conn.rollback()
        print(f"\n   ❌ Erreur : {e}")
        raise
    finally:
        cur.close()
        conn.close()
