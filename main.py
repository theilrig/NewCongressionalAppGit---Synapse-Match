from fastapi import FastAPI
from app.routers import health_profiles, voice_analysis, motor_assessment

app = FastAPI(title="Health Matcher API", version="0.3")

# Include routers
app.include_router(health_profiles.router)
app.include_router(voice_analysis.router)
app.include_router(motor_assessment.router)

@app.get("/")
def read_root():
    return {"message": "Health Matcher backend is live 🚀"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.3"}