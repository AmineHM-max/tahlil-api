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
    top_features: dict
    advice: str


def get_risk_level(prob):
    if prob > 0.7: return "Élevé"
    if prob > 0.4: return "Modéré"
    return "Faible"

def clean_feature_label(name: str) -> str:
    """
    Transforme 'filiere_bac_SM' → 'SM'
    Garde les autres noms intacts (num + bin features).
    Exclut complètement les features region_origine_*.
    """
    if name.startswith("filiere_bac_"):
        return name.replace("filiere_bac_", "")
    return name

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

    # 2. Poids de la régression logistique (au lieu des contributions)
    # On utilise directement les coefs du modèle — valeur absolue = importance
    coefs = pipeline.named_steps['model'].coef_[0]

    feature_weights = []
    for name, coef in zip(FEATURE_NAMES, coefs):
        # Exclure toutes les features région (biais éthique)
        if name.startswith("region_origine_"):
            continue
        clean_name = clean_feature_label(name)
        feature_weights.append((clean_name, abs(coef)))

    # Normaliser en % sur les features conservées
    total = sum(w for _, w in feature_weights)
    feature_weights = [
        (name, round((w / total) * 100, 2) if total > 0 else 0.0)
        for name, w in feature_weights
    ]

    # Top 3
    feature_weights.sort(key=lambda x: x[1], reverse=True)
    top_3 = {name: f"{pct}%" for name, pct in feature_weights[:3]}
    return PredictionResponse(
        prediction=pred,
        probability=round(prob,4),
        risk_level=risk,
        top_features=top_3,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
