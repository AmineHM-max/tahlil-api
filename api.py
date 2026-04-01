"""
TAHLIL TAÂLIM — API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import uvicorn
import os

# ─────────────────────────────────────────────────────────────
#  Chargement du modèle
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = "modele_tahlil.pkl"
COLUMNS_PATH = "colonnes.json"

pipeline = None
COLONNES_ATTENDUES = []
FEATURE_NAMES = []

try:
    pipeline = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        COLONNES_ATTENDUES = json.load(f)
    
    # Extraction des noms de features pour l'explicabilité
    num_cols = ['note_bac', 'moyenne_s1', 'nb_echecs_partiels', 'taux_absenteisme']
    cat_cols = ['region_origine', 'filiere_bac']
    bin_cols = ['statut_bourse', 'premier_emploi']
    
    cat_encoder = pipeline.named_steps['preprocessing'].transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(cat_cols).tolist()
    FEATURE_NAMES = num_cols + cat_features + bin_cols
    
except Exception as e:
    print(f"Erreur initialisation : {e}")

# ─────────────────────────────────────────────────────────────
#  Schémas
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="TAHLIL TAÂLIM API")

class StudentInput(BaseModel):
    note_bac: float
    moyenne_s1: float
    nb_echecs_partiels: int
    taux_absenteisme: float
    statut_bourse: int
    premier_emploi: int
    region_origine: str
    filiere_bac: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    top_features: dict  # Contiendra les %
    advice: str

# ─────────────────────────────────────────────────────────────
#  Logique métier simplifiée
# ─────────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob > 0.7: return "Élevé"
    if prob > 0.4: return "Modéré"
    return "Faible"

# ─────────────────────────────────────────────────────────────
#  Endpoint Principal
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    # Préparation données
    df_input = pd.DataFrame([student.dict()])[COLONNES_ATTENDUES]

    # 1. Calcul Probabilité
    prob = float(pipeline.predict_proba(df_input)[0][1])
    pred = 1 if prob > 0.5 else 0

    # 2. Calcul des Contributions en POURCENTAGE
    # On récupère les coefficients et les données transformées
    X_trans = pipeline.named_steps['preprocessing'].transform(df_input)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    
    coefs = pipeline.named_steps['model'].coef_[0]
    raw_contributions = X_trans[0] * coefs
    
    # On ne garde que les contributions positives (ce qui pousse vers le décrochage)
    # ou on prend la valeur absolue pour voir l'importance relative
    abs_contributions = np.abs(raw_contributions)
    total_impact = np.sum(abs_contributions)
    
    # Calcul du % (Contribution de chaque feature / Total des impacts)
    feature_impacts = []
    for name, val in zip(FEATURE_NAMES, abs_contributions):
        percentage = (val / total_impact) * 100 if total_impact > 0 else 0
        feature_impacts.append((name, round(percentage, 2)))

    # Trier et prendre le Top 3
    feature_impacts.sort(key=lambda x: x[1], reverse=True)
    top_3 = {item[0]: f"{item[1]}%" for item in feature_impacts[:3]}

    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 4),
        risk_level=get_risk_level(prob),
        top_features=top_3,
        advice="Analyse basée sur les facteurs académiques et l'assiduité."
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)