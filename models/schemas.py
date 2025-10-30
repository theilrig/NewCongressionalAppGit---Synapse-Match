from pydantic import BaseModel

class HealthProfileBase(BaseModel):
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    bmi: float
    smoker: str
    exercise_freq: str
    diet_quality: str
    alcohol_consumption: str
    chronic_disease: str
    stress_level: int
    sleep_hours: float

class HealthProfileCreate(HealthProfileBase):
    pass

class HealthProfileResponse(HealthProfileBase):
    id: int

    class Config:
        orm_mode = True

class ParkinsonsPatientBase(BaseModel):
    study_id: str
    condition: str
    disease_comment: str
    age_at_diagnosis: int
    age: int
    height_cm: float
    weight_kg: float
    bmi: float
    gender: str
    handedness: str
    appearance_in_kinship: str
    appearance_in_first_grade_kinship: str
    effect_of_alcohol_on_tremor: str

class ParkinsonsPatientResponse(ParkinsonsPatientBase):
    id: int
    similarity_score: float  

    class Config:
        orm_mode = True

