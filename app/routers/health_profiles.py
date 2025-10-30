from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import numpy as np

from app.core.database import get_db
from app.models import models
from app.models import schemas

router = APIRouter(prefix="/api/v1", tags=["HealthProfiles"])

NUMERIC_FIELDS = ["age", "height_cm", "weight_kg", "bmi", "stress_level", "sleep_hours"]
CATEGORICAL_FIELDS = ["gender", "smoker", "exercise_freq", "diet_quality", "alcohol_consumption", "chronic_disease"]
PARKINSONS_NUMERIC_FIELDS = ["age", "height_cm", "weight_kg", "bmi"]
PARKINSONS_CATEGORICAL_FIELDS = ["gender", "handedness"]


@router.post("/health_profiles", response_model=schemas.HealthProfileResponse)
def create_health_profile(payload: schemas.HealthProfileCreate, db: Session = Depends(get_db)):
    db_profile = models.HealthProfile(**payload.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile


def _as_vector_from_model(obj, fields):
    """Convert object to numeric vector using specified fields"""
    return np.array([float(getattr(obj, f, 0)) for f in fields], dtype=float)


def _categorical_mismatch_score(target, candidate, fields):
    """Calculate categorical mismatch as fraction of mismatches"""
    mismatches = 0
    for f in fields:
        if str(getattr(target, f, "")).lower() != str(getattr(candidate, f, "")).lower():
            mismatches += 1
    return mismatches / max(1, len(fields)) if fields else 0


@router.post("/match", response_model=List[schemas.HealthProfileResponse])
def match_profiles(
    profile: Optional[schemas.HealthProfileCreate] = None,
    saved_profile_id: Optional[int] = Query(None),
    top_k: int = Query(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Match against health profiles in database"""
    candidates = db.query(models.HealthProfile).all()
    if not candidates:
        raise HTTPException(status_code=404, detail="No candidate profiles available in database.")

    if saved_profile_id is not None:
        target = db.query(models.HealthProfile).filter(models.HealthProfile.id == saved_profile_id).first()
        if not target:
            raise HTTPException(status_code=404, detail="Saved profile not found.")
    elif profile is not None:
        class Tmp:
            pass
        target = Tmp()
        for k, v in profile.dict().items():
            setattr(target, k, v)
    else:
        raise HTTPException(status_code=400, detail="Provide profile body or saved_profile_id query parameter.")

    numeric_matrix = np.array([_as_vector_from_model(p, NUMERIC_FIELDS) for p in candidates], dtype=float)
    mins = numeric_matrix.min(axis=0)
    maxs = numeric_matrix.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)

    target_vec = _as_vector_from_model(target, NUMERIC_FIELDS)
    target_norm = (target_vec - mins) / ranges
    candidates_norm = (numeric_matrix - mins) / ranges

    dists = np.linalg.norm(candidates_norm - target_norm, axis=1)
    cat_penalties = np.array([_categorical_mismatch_score(target, p, CATEGORICAL_FIELDS) for p in candidates], dtype=float)

    final_scores = 0.8 * dists + 0.2 * cat_penalties
    idx_sorted = np.argsort(final_scores)[:top_k]

    matches = [candidates[i] for i in idx_sorted]
    return matches


@router.post("/match_parkinsons", response_model=List[schemas.ParkinsonsPatientResponse])
def match_parkinsons(
    saved_profile_id: int = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """
    Match a saved health profile against Parkinson's dataset.
    Automatically called after health profile matching.
    """
    user_profile = db.query(models.HealthProfile).filter(models.HealthProfile.id == saved_profile_id).first()
    if not user_profile:
        raise HTTPException(status_code=404, detail="Saved profile not found.")

    parkinsons_patients = db.query(models.ParkinsonsPatient).all()
    if not parkinsons_patients:
        raise HTTPException(status_code=404, detail="No Parkinson's patients in database.")

    parkinsons_numeric = np.array(
        [_as_vector_from_model(p, PARKINSONS_NUMERIC_FIELDS) for p in parkinsons_patients],
        dtype=float
    )
    user_numeric = _as_vector_from_model(user_profile, PARKINSONS_NUMERIC_FIELDS)

    mins = parkinsons_numeric.min(axis=0)
    maxs = parkinsons_numeric.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)

    user_norm = (user_numeric - mins) / ranges
    parkinsons_norm = (parkinsons_numeric - mins) / ranges

    dists = np.linalg.norm(parkinsons_norm - user_norm, axis=1)
    cat_penalties = np.array(
        [_categorical_mismatch_score(user_profile, p, PARKINSONS_CATEGORICAL_FIELDS) for p in parkinsons_patients],
        dtype=float
    )

    final_scores = 0.8 * dists + 0.2 * cat_penalties
    idx_sorted = np.argsort(final_scores)[:top_k]

    results = []
    for idx in idx_sorted:
        patient = parkinsons_patients[idx]
        similarity = 1 / (1 + final_scores[idx])
        
        response_dict = {
            "id": patient.id,
            "study_id": patient.study_id,
            "condition": patient.condition,
            "disease_comment": patient.disease_comment,
            "age_at_diagnosis": patient.age_at_diagnosis,
            "age": patient.age,
            "height_cm": patient.height_cm,
            "weight_kg": patient.weight_kg,
            "bmi": patient.bmi,
            "gender": patient.gender,
            "handedness": patient.handedness,
            "appearance_in_kinship": patient.appearance_in_kinship,
            "appearance_in_first_grade_kinship": patient.appearance_in_first_grade_kinship,
            "effect_of_alcohol_on_tremor": patient.effect_of_alcohol_on_tremor,
            "similarity_score": round(similarity * 100, 2)  # Return as percentage
        }
        results.append(schemas.ParkinsonsPatientResponse(**response_dict))

    return results
