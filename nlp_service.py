"""
InnoFaso — NLP Classification des causes de pannes
Modes :
  1. TF-IDF + SGD classifier (fallback local)
  2. LLM (Groq) si GROQ_API_KEY est défini dans .env
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

CATEGORIES = [
    'Mecanique',
    'Electrique',
    'Hydraulique_Fuite',
    'Instrumentation_Capteur',
    'Nettoyage_Obstruction',
    'Operateur_Utilisation',
    'Preventif_planifie',
    'Autre',
]

CATEGORIES_LABELS = {
    'Mecanique': 'Mécanique',
    'Electrique': 'Électrique',
    'Hydraulique_Fuite': 'Hydraulique / Fuite',
    'Instrumentation_Capteur': 'Instrumentation / Capteur',
    'Nettoyage_Obstruction': 'Nettoyage / Obstruction',
    'Operateur_Utilisation': 'Opérateur / Utilisation',
    'Preventif_planifie': 'Préventif planifié',
    'Autre': 'Autre',
}

KEYWORDS = {
    'Mecanique': [
        'roulement', 'courroie', 'engrenage', 'arbre', 'pignon', 'chaine',
        'usure', 'frottement', 'vibration', 'alignement', 'desserrage',
        'casse', 'fissure', 'deformation', 'jeu', 'grippage', 'blocage',
        'support', 'ressort', 'poulie', 'palier', 'accouplement',
        'axe', 'bague', 'changement', 'remplacement', 'reparation mecanique',
        'usure mecanique', 'defaut mecanique',
        'roulement moteur', 'roulement defectueux', 'usure roulement',
        'changement roulement', 'remplacement roulement',
        'jeu mecanique', 'bruit mecanique', 'vibration mecanique',
    ],
    'Electrique': [
        'moteur', 'electrique', 'cable', 'fusible', 'disjoncteur',
        'contacteur', 'relais', 'transformateur', 'variateur',
        'surchauffe moteur', 'panne electrique', 'court-circuit',
        'surtension', 'isolation', 'bobinage', 'charbon', 'balai',
        'resistance', 'thermique', 'defaut electrique', 'probleme electrique',
    ],
    'Hydraulique_Fuite': [
        'fuite', 'hydraulique', 'pneumatique', 'verin', 'pompe',
        'flexible', 'joint', 'raccord', 'soupape', 'distributeur',
        'huile', 'pression', 'debit', 'fuite huile', 'fuite eau',
        'fuite air', 'garniture', 'circuit hydraulique',
        'basse pression', 'surpression', 'vidange', 'niveau huile',
        'appoint huile', 'clapet', 'limiteur',
    ],
    'Instrumentation_Capteur': [
        'capteur', 'sonde', 'detecteur', 'thermostat', 'manometre',
        'pressostat', 'thermocouple', 'pt100', 'transmetteur',
        'regulateur', 'automate', 'ihm', 'afficheur',
        'compteur', 'debitmetre', 'niveau', 'temperature',
        'capteur defectueux', 'sonde defectueuse', 'calibration',
        'etalonnage', 'derive', 'mesure', 'signal',
        'probleme sonde', 'defaut capteur', 'panne capteur',
    ],
    'Nettoyage_Obstruction': [
        'nettoyage', 'obstruction', 'bouchon', 'colmatage',
        'encrassement', 'bouche', 'salete', 'depot', 'cale',
        'nettoyer', 'desobstruction', 'debourrage',
        'obstrue', 'filtre bouche', 'filtre colmate',
        'nettoyage filtre', 'nettoyage pompe', 'lavage',
        'rincage', 'decrassage', 'depoussierage',
        'obstruction pompe', 'pompe obstruée', 'pompe bouchee',
        'nettoyage obstruction', 'debourrage pompe',
    ],
    'Operateur_Utilisation': [
        'erreur operateur', 'mauvaise manipulation', 'defaut reglage',
        'reglage', 'parametrage', 'utilisation', 'conduite',
        'mauvaise utilisation', 'fausse manoeuvre', 'reglage incorrect',
        'erreur manipulation', 'defaut conduite', 'surcharge',
        'mal regle', 'mal positionne', 'dereglement',
    ],
    'Preventif_planifie': [
        'maintenance hebdomadaire', 'maintenance mensuelle',
        'maintenance trimestrielle', 'maintenance semestrielle',
        'maintenance annuelle', 'visite', 'inspection',
        'controle', 'graissage', 'lubrification',
        'maintenance preventive', 'entretien', 'revision',
        'hebdomadaire', 'mensuel', 'trimestriel',
        'controle temps', 'appoint', 'nettoyage preventif',
        'planifie', 'programme', 'periodique',
    ],
}


class PanneClassifier:
    """Classifieur de causes de pannes en 8 categories"""

    def __init__(self):
        self.pipeline = None
        self._init_fallback()

    def _init_fallback(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                max_features=5000,
                sublinear_tf=True,
            )),
            ('clf', SGDClassifier(
                loss='modified_huber',
                penalty='l2',
                alpha=1e-4,
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                tol=1e-3,
            )),
        ])
        self._train_synthetic()

    def _train_synthetic(self):
        texts = []
        labels = []
        for cat, mots in KEYWORDS.items():
            for mot in mots:
                texts.append(mot)
                labels.append(cat)
                texts.append(mot.replace(' ', ' '))
                labels.append(cat)
        self.pipeline.fit(texts, labels)

    def classify(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'categorie': 'Autre',
                'categorie_code': 'Autre',
                'confiance': 0.0,
                'mode': 'fallback',
            }
        try:
            proba = self.pipeline.predict_proba([text])[0]
            idx = np.argmax(proba)
            confidence = float(proba[idx])
            cat_code = self.pipeline.classes_[idx]
            cat_label = CATEGORIES_LABELS.get(cat_code, cat_code)
            return {
                'categorie': cat_label,
                'categorie_code': cat_code,
                'confiance': round(confidence, 4),
                'mode': 'tfidf_sgd',
            }
        except Exception:
            return {
                'categorie': 'Autre',
                'categorie_code': 'Autre',
                'confiance': 0.0,
                'mode': 'fallback',
            }

    def classify_with_llm(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'categorie': 'Autre',
                'categorie_code': 'Autre',
                'confiance': 0.0,
                'mode': 'fallback',
            }
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return self.classify(text)
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            cats_str = ', '.join(f'{k}={v}' for k, v in CATEGORIES_LABELS.items())
            prompt = (
                "Tu es un expert en maintenance industrielle.\n"
                f"Categorise la cause de panne suivante parmi : {cats_str}\n\n"
                f"Cause: \"{text}\"\n\n"
                "Reponds STRICTEMENT au format: CODE|CONFIANCE\n"
                "Exemple: Mecanique|0.95\n"
                "Exemple: Preventif_planifie|0.99"
            )
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )
            result = response.choices[0].message.content.strip()
            parts = result.split('|')
            cat_code = parts[0].strip()
            confidence = float(parts[1]) if len(parts) > 1 else 1.0
            if cat_code not in CATEGORIES:
                return self.classify(text)
            cat_label = CATEGORIES_LABELS.get(cat_code, cat_code)
            return {
                'categorie': cat_label,
                'categorie_code': cat_code,
                'confiance': round(confidence, 4),
                'mode': 'llm',
            }
        except Exception:
            return self.classify(text)


CLASSIFIER = PanneClassifier()


def classify_panne(text, use_llm=False):
    if use_llm:
        return CLASSIFIER.classify_with_llm(text)
    return CLASSIFIER.classify(text)
