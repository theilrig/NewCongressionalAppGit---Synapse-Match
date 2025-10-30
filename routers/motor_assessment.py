# app/routers/motor_assessment.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
import joblib
import os
import math
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["MotorAssessment"])

BASE = Path(__file__).resolve().parent 
MODELS_DIR = BASE / "models"
SPIRAL_MODEL_PATH = MODELS_DIR / "spiral_model.pkl"
SPIRAL_SCALER_PATH = MODELS_DIR / "spiral_scaler.pkl"
WAVE_MODEL_PATH = MODELS_DIR / "wave_model.pkl"
WAVE_SCALER_PATH = MODELS_DIR / "wave_scaler.pkl"
TAP_MODEL_PATH = MODELS_DIR / "tap_model.pkl"
TAP_SCALER_PATH = MODELS_DIR / "tap_scaler.pkl"

try:
    spiral_model = joblib.load(SPIRAL_MODEL_PATH)
    spiral_scaler = joblib.load(SPIRAL_SCALER_PATH)
    wave_model = joblib.load(WAVE_MODEL_PATH)
    wave_scaler = joblib.load(WAVE_SCALER_PATH)
    tap_model = joblib.load(TAP_MODEL_PATH)
    tap_scaler = joblib.load(TAP_SCALER_PATH)
    print("Motor assessment models loaded successfully")
except Exception as e:
    print(f"Motor assessment models not found: {e}")
    print(f"   Looking in: {MODELS_DIR}")
    spiral_model = wave_model = tap_model = None
    spiral_scaler = wave_scaler = tap_scaler = None


class TouchPoint(BaseModel):
    x: float
    y: float
    timestamp: int
    pressure: float


class TapEvent(BaseModel):
    target: str
    x: float
    y: float
    timestamp: int


class SpiralRequest(BaseModel):
    points: List[TouchPoint]
    duration: float
    screen_width: float
    screen_height: float


class WaveRequest(BaseModel):
    points: List[TouchPoint]
    duration: float
    screen_width: float
    screen_height: float


class TapRequest(BaseModel):
    taps: List[TapEvent]
    duration: float



def extract_spiral_features(points: List[TouchPoint], duration: float, screen_width: float, screen_height: float) -> Dict[str, float]:
    """Extract features from spiral drawing"""
    if len(points) < 2:
        raise HTTPException(status_code=400, detail="Not enough points")
    
    timestamps = np.array([p.timestamp for p in points])
    x_coords = np.array([p.x for p in points])
    y_coords = np.array([p.y for p in points])
    pressures = np.array([p.pressure for p in points])
    
    x_coords = x_coords / screen_width
    y_coords = y_coords / screen_height
    
    time_deltas = np.diff(timestamps) / 1000.0
    
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    velocities = distances / np.maximum(time_deltas, 0.001)  
    
    accelerations = np.diff(velocities) / np.maximum(time_deltas[:-1], 0.001)
    
    jerks = np.diff(accelerations) / np.maximum(time_deltas[:-2], 0.001)
    
    center_x, center_y = 0.5, 0.5  
    reference_points = []
    for t in np.linspace(0, 10 * np.pi, 500):
        r = t * 0.01  
        ref_x = center_x + r * np.cos(t)
        ref_y = center_y + r * np.sin(t)
        reference_points.append((ref_x, ref_y))
    
    deviations = []
    for i in range(len(x_coords)):
        min_dist = float('inf')
        for ref_x, ref_y in reference_points:
            dist = np.sqrt((x_coords[i] - ref_x)**2 + (y_coords[i] - ref_y)**2)
            min_dist = min(min_dist, dist)
        deviations.append(min_dist)
    
    if len(velocities) > 10:
        freqs, power_spectrum = signal.welch(velocities, fs=1/np.mean(time_deltas))
        tremor_band = (freqs >= 4) & (freqs <= 12)  # Parkinson's tremor: 4-12 Hz
        tremor_power = np.sum(power_spectrum[tremor_band])
    else:
        tremor_power = 0
    
    pause_threshold = 0.01
    num_pauses = np.sum(velocities < pause_threshold)
    
    pressure_std = np.std(pressures) if len(pressures) > 1 else 0
    
    segment_size = len(points) // 3
    if segment_size > 0:
        early_distances = distances[:segment_size].mean() if segment_size < len(distances) else 0
        late_distances = distances[-segment_size:].mean() if segment_size < len(distances) else 0
        micrographia_score = (early_distances - late_distances) / (early_distances + 0.001)
    else:
        micrographia_score = 0
    
    features = {
        'duration': duration,
        'num_points': len(points),
        'avg_velocity': np.mean(velocities) if len(velocities) > 0 else 0,
        'std_velocity': np.std(velocities) if len(velocities) > 0 else 0,
        'max_velocity': np.max(velocities) if len(velocities) > 0 else 0,
        'min_velocity': np.min(velocities) if len(velocities) > 0 else 0,
        'avg_acceleration': np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0,
        'avg_jerk': np.mean(np.abs(jerks)) if len(jerks) > 0 else 0,
        'avg_deviation': np.mean(deviations),
        'max_deviation': np.max(deviations),
        'tremor_power': float(tremor_power),
        'num_pauses': int(num_pauses),
        'pressure_std': float(pressure_std),
        'micrographia_score': float(micrographia_score),
        'drawing_smoothness': 1.0 / (1.0 + np.mean(np.abs(jerks))) if len(jerks) > 0 else 0,
    }
    
    return features


def extract_wave_features(points: List[TouchPoint], duration: float, screen_width: float, screen_height: float) -> Dict[str, float]:
    """Extract features from wave pattern drawing"""
    if len(points) < 2:
        raise HTTPException(status_code=400, detail="Not enough points")
    
    timestamps = np.array([p.timestamp for p in points])
    x_coords = np.array([p.x for p in points])
    y_coords = np.array([p.y for p in points])
    pressures = np.array([p.pressure for p in points])
    
    x_coords = x_coords / screen_width
    y_coords = y_coords / screen_height
    
    time_deltas = np.diff(timestamps) / 1000.0
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    velocities = distances / np.maximum(time_deltas, 0.001)

    reference_y = []
    for x in x_coords:
        ref_y = 0.5 + 0.15 * np.sin(x * 10)  
        reference_y.append(ref_y)
    reference_y = np.array(reference_y)
    
    y_deviations = np.abs(y_coords - reference_y)
    
    y_diff = np.diff(y_coords)
    wave_smoothness = 1.0 / (1.0 + np.std(np.diff(y_diff))) if len(y_diff) > 1 else 0
    
    features = {
        'duration': duration,
        'num_points': len(points),
        'avg_velocity': np.mean(velocities) if len(velocities) > 0 else 0,
        'std_velocity': np.std(velocities) if len(velocities) > 0 else 0,
        'avg_y_deviation': np.mean(y_deviations),
        'max_y_deviation': np.max(y_deviations),
        'wave_smoothness': float(wave_smoothness),
        'pressure_std': float(np.std(pressures)),
        'x_progression': (x_coords[-1] - x_coords[0]) / screen_width,  
    }
    
    return features


def extract_tap_features(taps: List[TapEvent], duration: float) -> Dict[str, float]:
    """Extract features from tap test"""
    if len(taps) < 2:
        return {
            'num_taps': len(taps),
            'taps_per_second': len(taps) / duration,
            'avg_interval': 0,
            'std_interval': 0,
            'min_interval': 0,
            'max_interval': 0,
            'num_errors': 0,
            'error_rate': 0,
            'rhythm_score': 0,
            'num_freezes': 0,
            'slowing_score': 0,
        }
    
    timestamps = np.array([t.timestamp for t in taps])
    targets = [t.target for t in taps]
    
    intervals = np.diff(timestamps) / 1000.0
    
    num_errors = 0
    for i in range(len(targets) - 1):
        if targets[i] == targets[i + 1]:
            num_errors += 1
    
    interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
    rhythm_score = 1.0 / (1.0 + interval_cv)
    
    freeze_threshold = np.mean(intervals) + 2 * np.std(intervals) if len(intervals) > 0 else 1.0
    num_freezes = np.sum(intervals > freeze_threshold)
    
    if len(intervals) > 4:
        first_half = intervals[:len(intervals)//2]
        second_half = intervals[len(intervals)//2:]
        slowing_score = np.mean(second_half) - np.mean(first_half)
    else:
        slowing_score = 0
    
    features = {
        'num_taps': len(taps),
        'taps_per_second': len(taps) / duration,
        'avg_interval': float(np.mean(intervals)) if len(intervals) > 0 else 0,
        'std_interval': float(np.std(intervals)) if len(intervals) > 0 else 0,
        'min_interval': float(np.min(intervals)) if len(intervals) > 0 else 0,
        'max_interval': float(np.max(intervals)) if len(intervals) > 0 else 0,
        'num_errors': int(num_errors),
        'error_rate': num_errors / len(taps) if len(taps) > 0 else 0,
        'rhythm_score': float(rhythm_score),
        'num_freezes': int(num_freezes),
        'slowing_score': float(slowing_score),
    }
    
    return features



@router.post("/analyze_spiral")
async def analyze_spiral(request: SpiralRequest):
    """Analyze spiral drawing for Parkinson's indicators"""
    try:
        features = extract_spiral_features(
            request.points,
            request.duration,
            request.screen_width,
            request.screen_height
        )
        
        if spiral_model and spiral_scaler:
            feature_vector = np.array([[
                features['duration'],
                features['num_points'],
                features['avg_velocity'],
                features['std_velocity'],
                features['max_velocity'],
                features['min_velocity'],
                features['avg_acceleration'],
                features['avg_jerk'],
                features['avg_deviation'],
                features['max_deviation'],
                features['tremor_power'],
                features['num_pauses'],
                features['pressure_std'],
                features['micrographia_score'],
                features['drawing_smoothness'],
            ]])
            
            feature_vector_scaled = spiral_scaler.transform(feature_vector)
            prediction = spiral_model.predict(feature_vector_scaled)[0]
            confidence_proba = spiral_model.predict_proba(feature_vector_scaled)[0]
            confidence = float(confidence_proba[prediction] * 100)
            
            result_text = "Likely Parkinson's" if prediction == 1 else "Healthy"
        else:
            score = 0
            if features['avg_velocity'] < 0.5:
                score += 20
            if features['tremor_power'] > 0.1:
                score += 25
            if features['avg_deviation'] > 0.05:
                score += 20
            if features['num_pauses'] > 5:
                score += 15
            if features['micrographia_score'] > 0.1:
                score += 20
            
            confidence = min(score, 100)
            result_text = "Likely Parkinson's" if confidence > 50 else "Healthy"
        
        return {
            "prediction": result_text,
            "confidence": confidence,
            "metrics": features,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing spiral: {str(e)}")


@router.post("/analyze_wave")
async def analyze_wave(request: WaveRequest):
    """Analyze wave pattern for Parkinson's indicators"""
    try:
        features = extract_wave_features(
            request.points,
            request.duration,
            request.screen_width,
            request.screen_height
        )
        
        if wave_model and wave_scaler:
            feature_vector = np.array([[
                features['duration'],
                features['num_points'],
                features['avg_velocity'],
                features['std_velocity'],
                features['avg_y_deviation'],
                features['max_y_deviation'],
                features['wave_smoothness'],
                features['pressure_std'],
                features['x_progression'],
            ]])
            
            feature_vector_scaled = wave_scaler.transform(feature_vector)
            prediction = wave_model.predict(feature_vector_scaled)[0]
            confidence_proba = wave_model.predict_proba(feature_vector_scaled)[0]
            confidence = float(confidence_proba[prediction] * 100)
            
            result_text = "Likely Parkinson's" if prediction == 1 else "Healthy"
        else:
            score = 0
            if features['avg_velocity'] < 0.5:
                score += 25
            if features['avg_y_deviation'] > 0.1:
                score += 30
            if features['wave_smoothness'] < 0.5:
                score += 25
            if features['std_velocity'] > 0.3:
                score += 20
            
            confidence = min(score, 100)
            result_text = "Likely Parkinson's" if confidence > 50 else "Healthy"
        
        return {
            "prediction": result_text,
            "confidence": confidence,
            "metrics": features,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing wave: {str(e)}")


@router.post("/analyze_taps")
async def analyze_taps(request: TapRequest):
    """Analyze tap test for Parkinson's indicators"""
    try:
        features = extract_tap_features(request.taps, request.duration)
        
        if tap_model and tap_scaler:
            feature_vector = np.array([[
                features['num_taps'],
                features['taps_per_second'],
                features['avg_interval'],
                features['std_interval'],
                features['min_interval'],
                features['max_interval'],
                features['num_errors'],
                features['error_rate'],
                features['rhythm_score'],
                features['num_freezes'],
                features['slowing_score'],
            ]])
            
            feature_vector_scaled = tap_scaler.transform(feature_vector)
            prediction = tap_model.predict(feature_vector_scaled)[0]
            confidence_proba = tap_model.predict_proba(feature_vector_scaled)[0]
            confidence = float(confidence_proba[prediction] * 100)
            
            result_text = "Likely Parkinson's" if prediction == 1 else "Healthy"
        else:
            score = 0
            if features['taps_per_second'] < 3:
                score += 30
            if features['rhythm_score'] < 0.5:
                score += 25
            if features['num_errors'] > 3:
                score += 20
            if features['num_freezes'] > 2:
                score += 25
            
            confidence = min(score, 100)
            result_text = "Likely Parkinson's" if confidence > 50 else "Healthy"
        
        return {
            "prediction": result_text,
            "confidence": confidence,
            "metrics": features,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing taps: {str(e)}")


@router.get("/motor_assessment_status")
async def check_status():
    """Check if motor assessment models are loaded"""
    return {
        "spiral_model_loaded": spiral_model is not None,
        "wave_model_loaded": wave_model is not None,
        "tap_model_loaded": tap_model is not None,
        "ready": all([spiral_model is not None, wave_model is not None, tap_model is not None]),
        "models_directory": str(MODELS_DIR)
    }