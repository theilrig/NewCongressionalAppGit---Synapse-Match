from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
import parselmouth
from parselmouth.praat import call
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import Dict

from app.core.database import get_db
from app.models import models

router = APIRouter(prefix="/api/v1", tags=["VoiceAnalysis"])

BASE = Path(__file__).resolve().parent       
MODELS_DIR = BASE / "models"
MODEL_PATH = MODELS_DIR / "parkinsons_voice_model.pkl"
SCALER_PATH = MODELS_DIR / "parkinsons_voice_scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print("Voice analysis model actually loaded")
except Exception as e:
    print(f"COULDNT LOAD MODELLL AAAAH: {e}")
    model = None
    scaler = None
    feature_columns = None


def extract_voice_features(audio_path: str) -> Dict[str, float]:
    """
    Extract voice features from audio file using Parselmouth (Praat).
    Returns a dictionary of features.
    """
    try:
        snd = parselmouth.Sound(audio_path)
        
        pitch = snd.to_pitch(time_step=None, pitch_floor=75, pitch_ceiling=500)
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        mean_F0 = call(pitch, "Get mean", 0, 0, "Hertz")
        max_F0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_F0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        
        local_jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        local_shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        hnr = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        mean_hnr = call(hnr, "Get mean", 0, 0)
        
        features = {
            'MDVP:Fo(Hz)': mean_F0 if not np.isnan(mean_F0) else 0,
            'MDVP:Fhi(Hz)': max_F0 if not np.isnan(max_F0) else 0,
            'MDVP:Flo(Hz)': min_F0 if not np.isnan(min_F0) else 0,
            'MDVP:Jitter(%)': local_jitter * 100 if not np.isnan(local_jitter) else 0,
            'MDVP:Shimmer': local_shimmer if not np.isnan(local_shimmer) else 0,
            'HNR': mean_hnr if not np.isnan(mean_hnr) else 0
        }
        
        return features
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")


@router.post("/analyze_voice")
async def analyze_voice(
    file: UploadFile = File(...),
    profile_id: int = None,
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded voice audio file for Parkinson's disease indicators.
    
    Parameters:
    - file: .wav audio file
    - profile_id: Optional health profile ID to associate with this analysis
    
    Returns:
    - prediction: "Healthy" or "Likely Parkinson's"
    - confidence: Prediction confidence (0-1)
    - features: Extracted voice features
    """
    
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(
            status_code=503, 
            detail="Voice analysis model not loaded. Please contact administrator."
        )
    
    if not file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only .wav audio files are supported"
        )
    
    temp_audio_path = f"/tmp/{file.filename}"
    try:
        with open(temp_audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        extracted_features = extract_voice_features(temp_audio_path)
        
        feature_dict = {}
        for col in feature_columns:
            if col in extracted_features:
                feature_dict[col] = extracted_features[col]
            else:
                feature_dict[col] = 0
        
        sample_df = pd.DataFrame([[feature_dict[col] for col in feature_columns]], 
                                 columns=feature_columns)
        
        sample_scaled = scaler.transform(sample_df)
        
        prediction = model.predict(sample_scaled)[0]
        prediction_proba = model.predict_proba(sample_scaled)[0]
        
        confidence = float(prediction_proba[prediction])
        
        result = "Likely Parkinson's" if prediction == 1 else "Healthy"
        
        os.remove(temp_audio_path)
        
        return {
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "features": extracted_features,
            "status": "success"
        }
        
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.get("/voice_analysis_status")
async def check_voice_analysis_status():
    """Check if voice analysis model is loaded and ready"""
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_columns_loaded": feature_columns is not None,
        "ready": all([model is not None, scaler is not None, feature_columns is not None])
    }