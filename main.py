# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import predict_fun   # tu módulo con predict_interval

app = FastAPI(
    title="Demand Forecasting API",
    description="Un endpoint para pedir p10/p50/p90",
    version="1.0"
)

class Query(BaseModel):
    year:    int
    week:    int
    product: int
    deposit: int

@app.post("/predict_interval")
def predict(q: Query):
    try:
        res = predict_fun.predict_interval(
            year=q.year,
            week=q.week,
            product=q.product,
            deposit=q.deposit
        )
    except ValueError as e:
        # por ejemplo "No hay histórico para..."
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "p10":     res["p10"],
        "p50":     res["p50"],
        "p90":     res["p90"]
    }
