"""
InnoFaso — Microservice IA (Flask)
Prédiction de pannes correctives par équipement

Lancer: python ia_service_complet.py
Tester: http://localhost:5001
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ── Charger le modèle ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, 'modele_panne_innofaso.pkl'))
le_eq    = joblib.load(os.path.join(BASE_DIR, 'label_encoder_equipement.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'features_liste.pkl'))

EQUIPEMENTS = list(le_eq.classes_)

MOIS_NOMS = {
    1:'Janvier', 2:'Février', 3:'Mars', 4:'Avril',
    5:'Mai', 6:'Juin', 7:'Juillet', 8:'Août',
    9:'Septembre', 10:'Octobre', 11:'Novembre', 12:'Décembre'
}

# ── Charger l'historique au démarrage ─────────────────────────
DATA_PATH = os.path.join(BASE_DIR, 'INNOFASO_Historique_de_maintenance_2025.xlsx')

def charger_historique():
    df_raw = pd.read_excel(DATA_PATH, sheet_name='T_Maint_Innofaso', header=0)
    df_raw.columns = ['Partenaire','Annee','Mois','Equipement','Sous_equip',
                      'Type_maint','Duree_h','Description']
    df = df_raw.dropna(subset=['Annee']).copy()
    df['Annee'] = df['Annee'].astype(int)
    df['Duree_h'] = pd.to_numeric(df['Duree_h'], errors='coerce').fillna(0)
    mois_map = {
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,
        'jan':1,'jun':6
    }
    df['Mois_num'] = df['Mois'].map(mois_map)
    df = df.dropna(subset=['Mois_num'])
    df['Mois_num'] = df['Mois_num'].astype(int)
    df = df[df['Type_maint'].isin(['Preventive','Corrective'])]

    hist = df.groupby(['Annee','Mois_num','Equipement']).agg(
        nb_interventions = ('Duree_h','count'),
        duree_totale_h   = ('Duree_h','sum'),
        duree_max_h      = ('Duree_h','max'),
        nb_correctif     = ('Type_maint', lambda x: (x=='Corrective').sum()),
    ).reset_index()
    hist['pct_correctif'] = hist['nb_correctif'] / hist['nb_interventions']
    return hist

print("⏳ Chargement de l'historique...")
HISTORIQUE = charger_historique()
print(f"✅ Historique chargé : {len(HISTORIQUE)} lignes agrégées")

# ── Préparer les features pour la prédiction ─────────────────
def preparer_features(annee_cible, mois_cible):
    lignes = []
    for eq in EQUIPEMENTS:
        hist_eq = HISTORIQUE[HISTORIQUE['Equipement']==eq].sort_values(['Annee','Mois_num'])

        def get_lag(lag):
            m = mois_cible - lag
            a = annee_cible
            while m <= 0:
                m += 12; a -= 1
            row = hist_eq[(hist_eq['Annee']==a) & (hist_eq['Mois_num']==m)]
            return row.iloc[0] if len(row) > 0 else None

        lags = [get_lag(i) for i in range(1, 4)]

        def v(lr, col):
            return float(lr[col]) if lr is not None and col in lr.index else 0.0

        row = {
            'Equipement':          eq,
            'Equipement_enc':      int(le_eq.transform([eq])[0]),
            'Mois_num':            mois_cible,
            'Annee':               annee_cible,
            'trimestre':           (mois_cible - 1) // 3 + 1,
            'mois_sin':            np.sin(2 * np.pi * mois_cible / 12),
            'mois_cos':            np.cos(2 * np.pi * mois_cible / 12),
            'duree_lag1':          v(lags[0], 'duree_totale_h'),
            'duree_lag2':          v(lags[1], 'duree_totale_h'),
            'duree_lag3':          v(lags[2], 'duree_totale_h'),
            'nb_interv_lag1':      v(lags[0], 'nb_interventions'),
            'nb_interv_lag2':      v(lags[1], 'nb_interventions'),
            'nb_interv_lag3':      v(lags[2], 'nb_interventions'),
            'nb_corr_lag1':        v(lags[0], 'nb_correctif'),
            'nb_corr_lag2':        v(lags[1], 'nb_correctif'),
            'nb_corr_lag3':        v(lags[2], 'nb_correctif'),
            'pct_corr_lag1':       v(lags[0], 'pct_correctif'),
            'pct_corr_lag2':       v(lags[1], 'pct_correctif'),
            'pct_corr_lag3':       v(lags[2], 'pct_correctif'),
            'duree_max_lag1':      v(lags[0], 'duree_max_h'),
            'duree_max_lag2':      v(lags[1], 'duree_max_h'),
            'rolling_nb_corr_3m':  sum(v(lags[i], 'nb_correctif') for i in range(3)),
            'rolling_duree_3m':    np.mean([v(lags[i], 'duree_totale_h') for i in range(3)]),
        }
        lignes.append(row)

    return pd.DataFrame(lignes)

def niveau_risque(proba):
    if proba >= 0.60: return 'ÉLEVÉ'
    if proba >= 0.38: return 'MODÉRÉ'
    return 'FAIBLE'

def couleur_risque(proba):
    if proba >= 0.60: return 'red'
    if proba >= 0.38: return 'orange'
    return 'green'

# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Interface web de test"""
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>InnoFaso — IA Prédiction Pannes</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; }
  header { background: #0f172a; color: white; padding: 16px 32px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { background: #22c55e; color: white; font-size: 10px;
                padding: 2px 8px; border-radius: 20px; font-weight: 600; }
  main { max-width: 900px; margin: 32px auto; padding: 0 16px; }
  .card { background: white; border-radius: 12px; border: 1px solid #e2e8f0;
          padding: 24px; margin-bottom: 20px; }
  h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #0f172a; }
  .row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
  label { display: block; font-size: 12px; color: #64748b; margin-bottom: 4px; }
  select, input { padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 8px;
                  font-size: 14px; background: white; }
  button { background: #0f172a; color: white; border: none; padding: 9px 24px;
           border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 500; }
  button:hover { background: #1e293b; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  #result { margin-top: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; background: #f1f5f9;
       font-weight: 500; color: #64748b; font-size: 11px; text-transform: uppercase; }
  td { padding: 10px 12px; border-top: 1px solid #f1f5f9; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
           font-size: 11px; font-weight: 600; }
  .badge-red    { background: #fee2e2; color: #dc2626; }
  .badge-orange { background: #ffedd5; color: #ea580c; }
  .badge-green  { background: #dcfce7; color: #16a34a; }
  .bar { height: 8px; border-radius: 4px; background: #f1f5f9; width: 120px; display: inline-block; }
  .bar-inner { height: 100%; border-radius: 4px; }
  .loading { text-align: center; padding: 32px; color: #94a3b8; }
  .meta { font-size: 12px; color: #94a3b8; margin-bottom: 12px; }
  .endpoints { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .ep { background: #f8fafc; border-radius: 8px; padding: 12px;
        border: 1px solid #e2e8f0; font-size: 12px; }
  .ep code { font-family: monospace; color: #6366f1; display: block; margin-top: 4px; }
  .ep p { color: #64748b; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
  .stat { background: white; border-radius: 10px; border: 1px solid #e2e8f0; padding: 14px; text-align: center; }
  .stat-num { font-size: 24px; font-weight: 700; color: #0f172a; }
  .stat-lbl { font-size: 11px; color: #94a3b8; margin-top: 2px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>⚡ InnoFaso — IA Prédiction de Pannes</h1>
  </div>
  <span>v1.0 · ONLINE</span>
</header>
<main>
  <div class="stats">
    <div class="stat"><div class="stat-num">15</div><div class="stat-lbl">Équipements</div></div>
    <div class="stat"><div class="stat-num">1 884</div><div class="stat-lbl">Interventions</div></div>
    <div class="stat"><div class="stat-num">5 ans</div><div class="stat-lbl">Historique</div></div>
    <div class="stat"><div class="stat-num">0.73</div><div class="stat-lbl">AUC-ROC</div></div>
  </div>

  <div class="card">
    <h2>🔮 Prédire les pannes d'un mois</h2>
    <div class="row">
      <div>
        <label>Mois</label>
        <select id="mois">
          <option value="1">Janvier</option><option value="2">Février</option>
          <option value="3">Mars</option><option value="4">Avril</option>
          <option value="5">Mai</option><option value="6">Juin</option>
          <option value="7" selected>Juillet</option><option value="8">Août</option>
          <option value="9">Septembre</option><option value="10">Octobre</option>
          <option value="11">Novembre</option><option value="12">Décembre</option>
        </select>
      </div>
      <div>
        <label>Année</label>
        <input type="number" id="annee" value="2026" min="2025" max="2030" style="width:100px">
      </div>
      <button onclick="predire()" id="btn">Prédire</button>
    </div>
    <div id="result"></div>
  </div>

  <div class="card">
    <h2>📡 Endpoints API disponibles</h2>
    <div class="endpoints">
      <div class="ep">
        <p>Prédictions d'un mois</p>
        <code>GET /api/predict/{annee}/{mois}</code>
      </div>
      <div class="ep">
        <p>Prédiction un équipement</p>
        <code>GET /api/predict/{annee}/{mois}/{equipement}</code>
      </div>
      <div class="ep">
        <p>Équipements disponibles</p>
        <code>GET /api/equipements</code>
      </div>
      <div class="ep">
        <p>Santé du service</p>
        <code>GET /api/health</code>
      </div>
    </div>
  </div>
</main>

<script>
async function predire() {
  const mois  = document.getElementById('mois').value;
  const annee = document.getElementById('annee').value;
  const btn   = document.getElementById('btn');
  const div   = document.getElementById('result');

  btn.disabled = true;
  div.innerHTML = '<div class="loading">⏳ Calcul en cours...</div>';

  try {
    const r = await fetch(`/api/predict/${annee}/${mois}`);
    const data = await r.json();

    const moisNoms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                      7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'};
    const nb_eleve  = data.predictions.filter(p => p.risque === 'ÉLEVÉ').length;
    const nb_modere = data.predictions.filter(p => p.risque === 'MODÉRÉ').length;

    let html = `
      <div class="meta" style="margin-top:16px">
        Prédictions pour <strong>${moisNoms[mois]} ${annee}</strong> —
        <span style="color:#dc2626">${nb_eleve} risque(s) élevé(s)</span>,
        ${nb_modere} modéré(s)
      </div>
      <table>
        <thead><tr>
          <th>Équipement</th><th>Probabilité</th><th>Risque</th><th>Recommandation</th>
        </tr></thead>
        <tbody>`;

    for (const p of data.predictions) {
      const pct   = (p.probabilite_panne * 100).toFixed(1);
      const cls   = p.risque === 'ÉLEVÉ' ? 'red' : p.risque === 'MODÉRÉ' ? 'orange' : 'green';
      const fill  = p.couleur === 'red' ? '#dc2626' : p.couleur === 'orange' ? '#ea580c' : '#16a34a';
      const reco  = p.risque === 'ÉLEVÉ'  ? '⚠️ Planifier maintenance préventive' :
                    p.risque === 'MODÉRÉ' ? '👁 Surveiller de près' :
                                           '✅ Aucune action requise';
      html += `
        <tr>
          <td style="font-weight:500">${p.equipement}</td>
          <td>
            <div class="bar"><div class="bar-inner" style="width:${pct}%;background:${fill}"></div></div>
            <span style="margin-left:8px;font-weight:600">${pct}%</span>
          </td>
          <td><span class="badge badge-${cls}">${p.risque}</span></td>
          <td style="color:#64748b;font-size:12px">${reco}</td>
        </tr>`;
    }
    html += '</tbody></table>';
    div.innerHTML = html;
  } catch(e) {
    div.innerHTML = `<div style="color:red;padding:16px">Erreur: ${e.message}</div>`;
  }
  btn.disabled = false;
}
// Lancer au chargement
predire();
</script>
</body>
</html>
"""
    return html

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'InnoFaso IA — Prédiction de pannes',
        'version': '1.0',
        'nb_equipements': len(EQUIPEMENTS),
        'nb_lignes_historique': len(HISTORIQUE)
    })

@app.route('/api/equipements')
def get_equipements():
    return jsonify({'equipements': EQUIPEMENTS})

@app.route('/api/predict/<int:annee>/<int:mois>')
def predict_mois(annee, mois):
    if not (1 <= mois <= 12):
        return jsonify({'error': 'Mois invalide (1-12)'}), 400
    if annee < 2020:
        return jsonify({'error': 'Année trop ancienne'}), 400

    X = preparer_features(annee, mois)
    probas = model.predict_proba(X[features])[:,1]

    predictions = []
    for eq, proba in zip(EQUIPEMENTS, probas):
        predictions.append({
            'equipement':        eq,
            'probabilite_panne': round(float(proba), 4),
            'probabilite_pct':   round(float(proba) * 100, 1),
            'risque':            niveau_risque(proba),
            'couleur':           couleur_risque(proba),
        })

    predictions.sort(key=lambda x: x['probabilite_panne'], reverse=True)
    return jsonify({
        'annee': annee,
        'mois':  mois,
        'mois_nom': MOIS_NOMS[mois],
        'predictions': predictions,
        'resume': {
            'nb_eleve':   sum(1 for p in predictions if p['risque'] == 'ÉLEVÉ'),
            'nb_modere':  sum(1 for p in predictions if p['risque'] == 'MODÉRÉ'),
            'nb_faible':  sum(1 for p in predictions if p['risque'] == 'FAIBLE'),
        }
    })

@app.route('/api/predict/<int:annee>/<int:mois>/<path:equipement>')
def predict_equipement(annee, mois, equipement):
    if equipement not in EQUIPEMENTS:
        return jsonify({'error': f'Équipement inconnu. Disponibles: {EQUIPEMENTS}'}), 404

    X = preparer_features(annee, mois)
    row = X[X['Equipement'] == equipement]
    proba = float(model.predict_proba(row[features])[0][1])

    # Historique des 6 derniers mois pour contexte
    hist_eq = HISTORIQUE[HISTORIQUE['Equipement'] == equipement].sort_values(
        ['Annee','Mois_num']).tail(6)
    historique_recent = hist_eq[['Annee','Mois_num','nb_interventions',
                                  'duree_totale_h','nb_correctif']].to_dict('records')

    return jsonify({
        'equipement':        equipement,
        'annee':             annee,
        'mois':              mois,
        'mois_nom':          MOIS_NOMS[mois],
        'probabilite_panne': round(proba, 4),
        'probabilite_pct':   round(proba * 100, 1),
        'risque':            niveau_risque(proba),
        'couleur':           couleur_risque(proba),
        'historique_recent': historique_recent,
    })

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  InnoFaso IA — Service de prédiction de pannes")
    print("="*55)
    print("  Interface web : http://localhost:5001")
    print("  API JSON      : http://localhost:5001/api/predict/2026/7")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)