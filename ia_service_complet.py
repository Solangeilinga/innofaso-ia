"""
InnoFaso — Microservice IA (Flask)
Prédiction de pannes correctives par équipement
Données depuis PostgreSQL (vue matérialisée) avec repli Excel.

Lancer: python ia_service_complet.py
Tester: http://localhost:5001
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from dotenv import load_dotenv
load_dotenv()

import json
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

import db_loader

try:
    from nlp_service import classify_panne
    NLP_AVAILABLE = True
except Exception as e:
    print(f"   [WARN] NLP service non disponible : {e}")
    NLP_AVAILABLE = False

    def classify_panne(text, use_llm=False):
        return {'categorie': 'N/A', 'confiance': 0.0, 'mode': 'unavailable'}

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, 'modele_panne_innofaso.pkl'))
le_eq    = joblib.load(os.path.join(BASE_DIR, 'label_encoder_equipement.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'features_liste.pkl'))

EQUIPEMENTS = list(le_eq.classes_)
LAG_MONTHS = [1, 2, 3]

MOIS_NOMS = {
    1:'Janvier', 2:'Février', 3:'Mars', 4:'Avril',
    5:'Mai', 6:'Juin', 7:'Juillet', 8:'Août',
    9:'Septembre', 10:'Octobre', 11:'Novembre', 12:'Décembre'
}


def charger_historique():
    try:
        df = db_loader.charger_donnees(source='auto')
        print(f"   Source : {'PostgreSQL' if 'age_equipement_ans' in df.columns else 'Excel'}")
        return df
    except RuntimeError as e:
        print(f"   [ERR] {e}")
        return pd.DataFrame()


print("[...] Chargement de l'historique...")
HISTORIQUE = charger_historique()
SOURCE_TYPE = 'postgres' if 'age_equipement_ans' in HISTORIQUE.columns else 'excel'
print(f"[OK] Historique chargé : {len(HISTORIQUE)} lignes depuis {'PostgreSQL' if SOURCE_TYPE == 'postgres' else 'Excel'}")


def _v(val, default=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

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


def preparer_features(annee_cible, mois_cible):
    lignes = []
    for eq in EQUIPEMENTS:
        hist_eq = HISTORIQUE[HISTORIQUE['equipement_nom'] == eq].sort_values(['annee', 'mois_num'])

        row = {
            'equipement': eq,
            'equipement_enc': int(le_eq.transform([eq])[0]),
            'mois_num': mois_cible,
            'annee': annee_cible,
            'trimestre': (mois_cible - 1) // 3 + 1,
            'mois_sin': np.sin(2 * np.pi * mois_cible / 12),
            'mois_cos': np.cos(2 * np.pi * mois_cible / 12),
        }

        if SOURCE_TYPE == 'postgres':
            derniere = hist_eq.iloc[-1] if len(hist_eq) > 0 else None
            if derniere is not None:
                row['age_equipement_ans'] = float(_v(derniere.get('age_equipement_ans'), 0))
                row['nb_preventif_planifie'] = int(_v(derniere.get('nb_preventif_planifie'), 0))
                row['nb_preventif_realise'] = int(_v(derniere.get('nb_preventif_realise'), 0))
                row['nb_preventif_en_retard'] = int(_v(derniere.get('nb_preventif_en_retard'), 0))
                row['taux_respect_preventif_pct'] = float(_v(derniere.get('taux_respect_preventif_pct'), 0))
                row['taux_disponibilite_moyen'] = float(_v(derniere.get('taux_disponibilite_moyen'), 100))
                row['taux_dispo_ligne'] = float(_v(derniere.get('taux_dispo_ligne'), 100))
                row['nb_pieces_reference'] = int(_v(derniere.get('nb_pieces_reference'), 0))
                row['stock_total_pieces'] = int(_v(derniere.get('stock_total_pieces'), 0))
                row['nb_pieces_sous_seuil'] = int(_v(derniere.get('nb_pieces_sous_seuil'), 0))
                row['duree_arret_moyenne_h'] = float(_v(derniere.get('duree_arret_moyenne_h'), 0))
                row['duree_maintenance_moyenne_h'] = float(_v(derniere.get('duree_maintenance_moyenne_h'), 0))
            else:
                for col in ['age_equipement_ans', 'taux_respect_preventif_pct',
                            'taux_disponibilite_moyen', 'taux_dispo_ligne',
                            'duree_arret_moyenne_h', 'duree_maintenance_moyenne_h']:
                    row[col] = 0.0
                for col in ['nb_preventif_planifie', 'nb_preventif_realise',
                            'nb_preventif_en_retard', 'nb_pieces_reference',
                            'stock_total_pieces', 'nb_pieces_sous_seuil']:
                    row[col] = 0
        else:
            row['duree_totale_h'] = 0.0
            row['duree_max_h'] = 0.0

        for lag in LAG_MONTHS:
            row[f'nb_correctif_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'nb_correctif')
            if SOURCE_TYPE == 'postgres':
                row[f'duree_arret_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'duree_arret_total_h')
                row[f'nb_interv_quart_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'nb_interv_quart')
            else:
                row[f'duree_lag{lag}'] = get_lag_value(hist_eq, annee_cible, mois_cible, lag, 'duree_totale_h')

        nb_corr_lags = sum(row.get(f'nb_correctif_lag{lag}', 0) for lag in LAG_MONTHS)
        row['rolling_nb_correctif_3m'] = nb_corr_lags
        row['rolling_correctif_trend'] = (
            row.get('nb_correctif_lag1', 0) - row.get('nb_correctif_lag3', 0)
        )

        if SOURCE_TYPE == 'postgres':
            duree_vals = [row.get(f'duree_arret_lag{lag}', 0) for lag in LAG_MONTHS]
            row['rolling_duree_arret_3m'] = np.mean(duree_vals)
        else:
            duree_vals = [row.get(f'duree_lag{lag}', 0) for lag in LAG_MONTHS]
            row['rolling_duree_3m'] = np.mean(duree_vals)

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
  <span>v2.0 · ONLINE</span>
</header>
<main>
  <div class="stats">
    <div class="stat"><div class="stat-num">""" + str(len(EQUIPEMENTS)) + """</div><div class="stat-lbl">Équipements</div></div>
    <div class="stat"><div class="stat-num">""" + str(len(HISTORIQUE)) + """</div><div class="stat-lbl">Lignes historiques</div></div>
    <div class="stat"><div class="stat-num">""" + ('PostgreSQL' if SOURCE_TYPE == 'postgres' else 'Excel') + """</div><div class="stat-lbl">Source données</div></div>
    <div class="stat"><div class="stat-num">XGBoost</div><div class="stat-lbl">Modèle</div></div>
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
    <h2>🏷️ Classifier une cause</h2>
    <div class="row">
      <div style="flex:1">
        <label>Texte de la cause</label>
        <input type="text" id="classifyText" style="width:100%" placeholder="Ex: Fuite d'huile sur le vérin">
      </div>
      <button onclick="classify()" id="btnCls">Classifier</button>
    </div>
    <div id="classifyResult" style="margin-top:12px;font-size:14px"></div>
  </div>

  <div class="card">
    <h2>💬 Assistant Maintenance</h2>
    <div style="margin-bottom:8px;max-height:200px;overflow-y:auto;background:#f8fafc;border-radius:8px;padding:12px" id="chatBox">
      <div style="color:#94a3b8;font-size:12px">Pose une question sur la maintenance, les équipements ou les pannes.</div>
    </div>
    <div class="row">
      <div style="flex:1">
        <input type="text" id="chatInput" style="width:100%" placeholder="Ex: Quel est le risque pour Filling machine en juillet ?"
               onkeydown="if(event.key==='Enter') chat()">
      </div>
      <button onclick="chat()" id="btnChat">Envoyer</button>
    </div>
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
        <p>Classifier une cause</p>
        <code>POST /api/classify</code>
      </div>
      <div class="ep">
        <p>Classifier (batch)</p>
        <code>POST /api/classify/batch</code>
      </div>
      <div class="ep">
        <p>Assistant maintenance</p>
        <code>POST /api/bot/chat</code>
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
async function classify() {
  const text = document.getElementById('classifyText').value.trim();
  const btn  = document.getElementById('btnCls');
  const div  = document.getElementById('classifyResult');
  if (!text) { div.innerHTML = '<span style="color:#dc2626">Veuillez entrer un texte</span>'; return; }
  btn.disabled = true;
  div.innerHTML = '<span style="color:#94a3b8">Classification en cours...</span>';
  try {
    const r = await fetch('/api/classify', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}) });
    const d = await r.json();
    const pct = (d.confiance * 100).toFixed(0);
    const cls = d.confiance > 0.7 ? 'green' : d.confiance > 0.4 ? 'orange' : 'red';
    div.innerHTML = `
      <span style="font-weight:600">${d.categorie}</span>
      <span class="badge badge-${cls}">${pct}%</span>
      <span style="font-size:11px;color:#94a3b8;margin-left:8px">mode: ${d.mode}</span>`;
  } catch(e) { div.innerHTML = `<span style="color:#dc2626">Erreur: ${e.message}</span>`; }
  btn.disabled = false;
}

let chatHistory = [];

async function chat() {
  const input = document.getElementById('chatInput');
  const box   = document.getElementById('chatBox');
  const btn   = document.getElementById('btnChat');
  const msg   = input.value.trim();
  if (!msg) return;
  box.innerHTML += `<div style="margin:4px 0"><b>Vous:</b> ${msg}</div>`;
  input.value = '';
  btn.disabled = true;
  box.innerHTML += `<div style="margin:4px 0;color:#94a3b8">Assistant réfléchit...</div>`;
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch('/api/bot/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:msg, history:chatHistory}) });
    const d = await r.json();
    box.innerHTML = box.innerHTML.replace('<div style="margin:4px 0;color:#94a3b8">Assistant réfléchit...</div>', '');
    box.innerHTML += `<div style="margin:4px 0;color:#0f172a"><b>Assistant:</b> ${d.reponse}</div>`;
    chatHistory.push({role:'user', content:msg});
    chatHistory.push({role:'assistant', content:d.reponse});
    if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
  } catch(e) {
    box.innerHTML = box.innerHTML.replace('<div style="margin:4px 0;color:#94a3b8">Assistant réfléchit...</div>', '');
    box.innerHTML += `<div style="margin:4px 0;color:#dc2626">Erreur: ${e.message}</div>`;
  }
  box.scrollTop = box.scrollHeight;
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
        'version': '2.0',
        'source_donnees': SOURCE_TYPE,
        'nb_equipements': len(EQUIPEMENTS),
        'nb_lignes_historique': len(HISTORIQUE),
        'modele': type(model).__name__,
    })


@app.route('/api/equipements')
def get_equipements():
    return jsonify({'equipements': EQUIPEMENTS})


@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Champ "text" requis'}), 400
    text = data['text']
    use_llm = data.get('use_llm', False)
    result = classify_panne(text, use_llm=use_llm)
    return jsonify(result)


@app.route('/api/classify/batch', methods=['POST'])
def api_classify_batch():
    data = request.get_json(silent=True)
    if not data or 'texts' not in data:
        return jsonify({'error': 'Champ "texts" requis (liste de chaines)'}), 400
    texts = data['texts']
    use_llm = data.get('use_llm', False)
    results = [classify_panne(t, use_llm=use_llm) for t in texts]
    return jsonify({'results': results})


def build_contexte_app():
    equipements_str = '\n'.join(f'  - {e}' for e in EQUIPEMENTS)
    try:
        etats = {}
        conn = db_loader.get_db_conn()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT nom, etat FROM equipements WHERE actif = true ORDER BY nom")
            for row in cur.fetchall():
                etats[row[0]] = row[1]
            cur.close()
            conn.close()
    except Exception:
        etats = {}
    etats_str = ''
    if etats:
        lignes = []
        for e in EQUIPEMENTS:
            s = etats.get(e, 'inconnu')
            lbl = {'OPERATIONNEL': '✅ OK', 'EN_PANNE': '🔴 Panne', 'EN_MAINTENANCE': '🟡 Maintenance'}.get(s, s)
            lignes.append(f'  - {e} → {lbl}')
        etats_str = '\n'.join(lignes)
    nb_histo = len(HISTORIQUE)
    return (
        f"Tu es l'assistant maintenance de l'application InnoFaso, "
        f"un systeme de gestion de maintenance industrielle.\n\n"
        f"L'application gere {len(EQUIPEMENTS)} equipements :\n{equipements_str}\n\n"
        + (f"États actuels des equipements :\n{etats_str}\n\n" if etats_str else "")
        + f"La base historique contient {nb_histo} enregistrements de maintenance.\n\n"
        f"L'utilisateur peut :\n"
        f"- Decrire une cause de panne → l'IA la classifie (Mecanique, Electrique, Hydraulique/Fuite, "
        f"Instrumentation/Capteur, Nettoyage/Obstruction, Operateur/Utilisation, Preventif planifie, Autre)\n"
        f"- Consulter les predictions de pannes par equipement et par mois\n"
        f"- Diagnostiquer un probleme sur un equipement\n\n"
        f"Reponds en francais de maniere concise, technique et en t'appuyant sur les donnees "
        f"de l'application. Si l'utilisateur decrit un symptome, propose de le classer "
        f"automatiquement. Si l'utilisateur demande une prediction pour un equipement specifique, "
        f"donne les dernieres tendances disponibles. Sois toujours pertinent par rapport "
        f"aux equipements listes ci-dessus."
    )


GROQ_API_KEY = os.getenv('GROQ_API_KEY')

@app.route('/api/bot/chat', methods=['POST'])
def bot_chat():
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': 'Champ "message" requis'}), 400
    message = data['message'].strip()
    history = data.get('history', [])

    system_prompt = build_contexte_app()

    msg_lower = message.lower()

    # Detection: demande d'API explicite → rediriger
    if any(w in msg_lower for w in ['api', 'endpoint', 'curl', 'postman']):
        equipements_str = ', '.join(EQUIPEMENTS[:5]) + ', ...'
        return jsonify({
            'reponse': (
                f"Endpoints disponibles:\n"
                f"- GET /api/predict/ANNEE/MOIS → predictions par equipement\n"
                f"- POST /api/classify → classer une cause (body: {{\"text\": \"...\"}})\n"
                f"- POST /api/bot/chat → ce chatbot\n"
                f"Equipements: {equipements_str}"
            ),
            'mode': 'regle',
        })

    # Detection: description d'une cause de panne → classifier automatiquement
    cause_keywords = ['fuite', 'casse', 'rouille', 'usure', 'surchauffe', 'bruit', 'vibration',
                      'bloque', 'dechire', 'courcircuit', 'surcharge', 'oxydation', 'frottement',
                      'obstruction', 'bouchon', 'nettoyage', 'capteur', 'electrique', 'mecanique',
                      'hydraulique', 'pneumatique']
    if any(w in msg_lower for w in cause_keywords) and len(message) > 10:
        try:
            result = classify_panne(message)
            categorie = result.get('categorie', 'Autre')
            confiance = result.get('confiance', 0)
            conf_pct = f"{confiance * 100:.0f}%"
            suggestions = {
                'Mecanique': 'Verifier l\'usure des pieces, graisser, remplacer si necessaire.',
                'Electrique': 'Verifier le cablage, les fusibles, et les contacteurs.',
                'Hydraulique/Fuite': 'Identifier la source de la fuite, remplacer joint ou flexible.',
                'Instrumentation/Capteur': 'Nettoyer ou recalibrer le capteur, verifier la liaison.',
                'Nettoyage/Obstruction': 'Nettoyer l\'equipement, enlever les obstructions.',
                'Operateur/Utilisation': 'Former l\'operateur, verifier la procedure.',
                'Preventif planifie': 'Operation de maintenance preventive prevue.',
                'Autre': 'Diagnostic non standard, contacter le service technique.',
            }
            action = suggestions.get(categorie, '')
            return jsonify({
                'reponse': (
                    f"Cause classée : **{categorie}** (confiance {conf_pct})\n\n"
                    f"Action suggérée : {action}\n\n"
                    f"Utilise le widget *Classifier une cause de panne* sur la page Equipements "
                    f"pour tester d'autres descriptions."
                ),
                'mode': 'classify',
            })
        except Exception:
            pass

    # Detection: question sur un equipement specifique
    equipement_trouve = None
    for eq in EQUIPEMENTS:
        if eq.lower() in msg_lower:
            equipement_trouve = eq
            break
    if equipement_trouve and any(w in msg_lower for w in ['panne', 'risque', 'probabilite', 'pred']):
        return jsonify({
            'reponse': (
                f"Consulte les predictions pour **{equipement_trouve}** directement sur le tableau de bord "
                f"ou via l'API. Tu peux aussi decrire une cause specifique et je la classerai."
            ),
            'mode': 'regle',
        })

    # Mode LLM si cle disponible
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-10:]:
                messages.append(h)
            messages.append({"role": "user", "content": message})
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            reponse = response.choices[0].message.content
            return jsonify({'reponse': reponse, 'mode': 'llm'})
        except Exception as e:
            pass

    # Fallback conversationnel
    eq_list = ', '.join(EQUIPEMENTS[:8]) + '...'
    salutations = ['bonjour', 'salut', 'hello', 'bonsoir', 'coucou']
    if any(w in msg_lower for w in salutations):
        return jsonify({
            'reponse': (
                f"Bonjour ! Je suis l'assistant maintenance InnoFaso.\n\n"
                f"Je connais {len(EQUIPEMENTS)} equipements : {eq_list}\n\n"
                f"Je peux :\n"
                f"- Classer une cause de panne (décrivez le symptôme)\n"
                f"- Donner les tendances de pannes par equipement\n"
                f"- Repondre aux questions sur la maintenance\n\n"
                f"Que puis-je faire pour vous ?"
            ),
            'mode': 'fallback',
        })

    return jsonify({
        'reponse': (
            f"Je n'ai pas de reponse specifique a cette question. "
            f"Les equipements suivis sont : {eq_list}\n"
            f"Essayez de :\n"
            f"- Decrire une cause de panne (ex: \"bruit anormal compresseur\")\n"
            f"- Demander les predictions pour un equipement\n"
            f"- Poser une question sur la maintenance d'un equipement specifique"
        ),
        'mode': 'fallback',
    })


@app.route('/api/bot/chat/stream', methods=['POST'])
def bot_chat_stream():
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': 'Champ "message" requis'}), 400
    message = data['message'].strip()
    history = data.get('history', [])

    def generate():
        msg_lower = message.lower()

        # Detection: demande d'API
        if any(w in msg_lower for w in ['api', 'endpoint', 'curl', 'postman']):
            equipements_str = ', '.join(EQUIPEMENTS[:5]) + ', ...'
            txt = (
                f"Endpoints disponibles:\n"
                f"- GET /api/predict/ANNEE/MOIS → predictions par equipement\n"
                f"- POST /api/classify → classer une cause (body: {{\"text\": \"...\"}})\n"
                f"- POST /api/bot/chat → ce chatbot\n"
                f"Equipements: {equipements_str}"
            )
            yield f"data: {{\"token\": {json.dumps(txt)}}}\n\n"
            yield "data: {\"done\": true, \"mode\": \"regle\"}\n\n"
            return

        # Detection: cause de panne
        cause_keywords = ['fuite', 'casse', 'rouille', 'usure', 'surchauffe', 'bruit', 'vibration',
                          'bloque', 'dechire', 'courcircuit', 'surcharge', 'oxydation', 'frottement',
                          'obstruction', 'bouchon', 'nettoyage', 'capteur', 'electrique', 'mecanique',
                          'hydraulique', 'pneumatique']
        if any(w in msg_lower for w in cause_keywords) and len(message) > 10:
            try:
                result = classify_panne(message) if NLP_AVAILABLE else {'categorie': 'N/A', 'confiance': 0.0}
                categorie = result.get('categorie', 'Autre')
                confiance = result.get('confiance', 0)
                conf_pct = f"{confiance * 100:.0f}%"
                suggestions = {
                    'Mecanique': 'Verifier l\'usure des pieces, graisser, remplacer si necessaire.',
                    'Electrique': 'Verifier le cablage, les fusibles, et les contacteurs.',
                    'Hydraulique/Fuite': 'Identifier la source de la fuite, remplacer joint ou flexible.',
                    'Instrumentation/Capteur': 'Nettoyer ou recalibrer le capteur, verifier la liaison.',
                    'Nettoyage/Obstruction': 'Nettoyer l\'equipement, enlever les obstructions.',
                    'Operateur/Utilisation': 'Former l\'operateur, verifier la procedure.',
                    'Preventif planifie': 'Operation de maintenance preventive prevue.',
                    'Autre': 'Diagnostic non standard, contacter le service technique.',
                }
                action = suggestions.get(categorie, '')
                txt = (
                    f"Cause classée : **{categorie}** (confiance {conf_pct})\n\n"
                    f"Action suggérée : {action}\n\n"
                    f"Widget *Classifier* sur la page Equipements pour tester d'autres causes."
                )
                yield f"data: {{\"token\": {json.dumps(txt)}}}\n\n"
                yield "data: {\"done\": true, \"mode\": \"classify\"}\n\n"
                return
            except Exception:
                pass

        # Detection: equipement specifique
        equipement_trouve = None
        for eq in EQUIPEMENTS:
            if eq.lower() in msg_lower:
                equipement_trouve = eq
                break
        if equipement_trouve and any(w in msg_lower for w in ['panne', 'risque', 'probabilite', 'pred']):
            txt = (
                f"Consulte les predictions pour **{equipement_trouve}** sur le tableau de bord "
                f"ou via l'API. Decris une cause specifique et je la classerai."
            )
            yield f"data: {{\"token\": {json.dumps(txt)}}}\n\n"
            yield "data: {\"done\": true, \"mode\": \"regle\"}\n\n"
            return

        # Mode LLM streaming
        if GROQ_API_KEY:
            try:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                system_prompt = build_contexte_app()
                msgs = [{"role": "system", "content": system_prompt}]
                for h in history[-10:]:
                    msgs.append(h)
                msgs.append({"role": "user", "content": message})
                stream = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=msgs,
                    temperature=0.3,
                    max_tokens=500,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ''
                    if token:
                        yield f"data: {{\"token\": {json.dumps(token)}}}\n\n"
                yield "data: {\"done\": true, \"mode\": \"llm\"}\n\n"
                return
            except Exception as e:
                yield f"data: {{\"token\": {json.dumps(f'Erreur LLM: {str(e)}')}}}\n\n"
                yield "data: {\"done\": true, \"mode\": \"error\"}\n\n"
                return

        # Fallback
        eq_list = ', '.join(EQUIPEMENTS[:8]) + '...'
        salutations = ['bonjour', 'salut', 'hello', 'bonsoir', 'coucou']
        if any(w in msg_lower for w in salutations):
            txt = (
                f"Bonjour ! Je suis l'assistant maintenance InnoFaso.\n\n"
                f"Je connais {len(EQUIPEMENTS)} equipements : {eq_list}\n\n"
                f"Je peux :\n"
                f"- Classer une cause de panne (decrirez le symptome)\n"
                f"- Donner les tendances de pannes par equipement\n"
                f"- Repondre aux questions sur la maintenance\n\n"
                f"Que puis-je faire pour vous ?"
            )
            yield f"data: {{\"token\": {json.dumps(txt)}}}\n\n"
            yield "data: {\"done\": true, \"mode\": \"fallback\"}\n\n"
            return

        txt = (
            f"Je n'ai pas de reponse specifique. "
            f"Les equipements suivis sont : {eq_list}\n"
            f"Essayez de :\n"
            f"- Decrire une cause de panne (ex: \"bruit anormal compresseur\")\n"
            f"- Demander les predictions pour un equipement\n"
            f"- Poser une question sur la maintenance d'un equipement specifique"
        )
        yield f"data: {{\"token\": {json.dumps(txt)}}}\n\n"
        yield "data: {\"done\": true, \"mode\": \"fallback\"}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/predict/<int:annee>/<int:mois>')
def predict_mois(annee, mois):
    if not (1 <= mois <= 12):
        return jsonify({'error': 'Mois invalide (1-12)'}), 400
    if annee < 2020:
        return jsonify({'error': 'Année trop ancienne'}), 400

    X = preparer_features(annee, mois)
    probas = model.predict_proba(X[features])[:, 1]

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
    row = X[X['equipement'] == equipement]
    proba = float(model.predict_proba(row[features])[0][1])

    hist_eq = HISTORIQUE[HISTORIQUE['equipement_nom'] == equipement].sort_values(
        ['annee', 'mois_num']).tail(6)

    historique_recent = hist_eq[['annee', 'mois_num', 'nb_correctif']].to_dict('records')

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
    port = int(os.getenv('FLASK_PORT', '5001'))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    print("\n" + "=" * 55)
    print("  InnoFaso IA v2 — Service de prédiction de pannes")
    print(f"  Source données : {'PostgreSQL' if SOURCE_TYPE == 'postgres' else 'Excel (fallback)'}")
    print("=" * 55)
    print(f"  Interface web : http://localhost:{port}")
    print(f"  API JSON      : http://localhost:{port}/api/predict/2026/7")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
