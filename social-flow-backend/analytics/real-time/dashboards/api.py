from fastapi import FastAPI
from .metrics import compute_kpis

app = FastAPI(title="Real-Time Dashboards API")

# In-memory events store for demonstration
EVENTS = []


@app.get("/metrics")
def get_metrics():
    return compute_kpis(EVENTS)


@app.post("/event")
def add_event(value: float):
    EVENTS.append(value)
    return {"status": "ok", "count": len(EVENTS)}
