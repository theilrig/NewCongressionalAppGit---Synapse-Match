from sqlalchemy import Column, Integer, String, Float, Boolean
from app.models.base import Base

class HealthProfile(Base):
    __tablename__ = "health_profiles"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    height_cm = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    smoker = Column(String, nullable=False)
    exercise_freq = Column(String, nullable=False)
    diet_quality = Column(String, nullable=False)
    alcohol_consumption = Column(String, nullable=False)
    chronic_disease = Column(String, nullable=False)
    stress_level = Column(Integer, nullable=False)
    sleep_hours = Column(Float, nullable=False)

class ParkinsonsPatient(Base):
    __tablename__ = "parkinsons_patients"

    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(String, nullable=True)
    condition = Column(String, nullable=False)  
    disease_comment = Column(String, nullable=True)
    age_at_diagnosis = Column(Integer, nullable=True)
    age = Column(Integer, nullable=False)
    height_cm = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    gender = Column(String, nullable=False)
    handedness = Column(String, nullable=True)
    appearance_in_kinship = Column(String, nullable=True)
    appearance_in_first_grade_kinship = Column(String, nullable=True)
    effect_of_alcohol_on_tremor = Column(String, nullable=True)
