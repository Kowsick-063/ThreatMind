from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.network import router as network_router


app = FastAPI(
    title="ThreatMind API",
    description="Predictive Cyber Attack World Model with Counterfactual Defense Planning",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(network_router)


@app.get("/")
def root():
    return {
        "project": "ThreatMind",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "threatmind-backend",
    }