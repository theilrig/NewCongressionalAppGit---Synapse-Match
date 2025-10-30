import pandas as pd
from sqlalchemy.orm import Session
from app.core.database import engine
from app.models.models import HealthProfile

CSV_PATH = "/Users/abbasraza/Downloads/synthetic_health_lifestyle_dataset.csv"

print("Reading CSV data")
df = pd.read_csv(CSV_PATH)

session = Session(bind=engine)

print("Inserting data into health_profiles table")
for _, row in df.iterrows():
    profile = HealthProfile(
        age=int(row["Age"]),
        gender=str(row["Gender"]),
        height_cm=float(row["Height_cm"]),
        weight_kg=float(row["Weight_kg"]),
        bmi=float(row["BMI"]),
        smoker=str(row["Smoker"]),
        exercise_freq=str(row["Exercise_Freq"]),
        diet_quality=str(row["Diet_Quality"]),
        alcohol_consumption=str(row["Alcohol_Consumption"]),
        chronic_disease=str(row["Chronic_Disease"]),
        stress_level=int(row["Stress_Level"]),
        sleep_hours=float(row["Sleep_Hours"]),
    )
    session.add(profile)

session.commit()
session.close()
print("CSV data inserted finally jesus allah")
