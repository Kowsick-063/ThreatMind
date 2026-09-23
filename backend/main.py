from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.counterfactual import router as counterfactual_router
from backend.api.decision import router as decision_router
from backend.api.explain import router as explain_router
from backend.api.forecast import router as forecast_router
from backend.api.network import router as network_router
from backend.api.incidents import router as incidents_router
from backend.api.soc import router as soc_router


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
app.include_router(counterfactual_router)
app.include_router(explain_router)
app.include_router(forecast_router)
app.include_router(decision_router)
app.include_router(soc_router)
app.include_router(incidents_router)


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