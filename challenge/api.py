from fastapi import FastAPI, Depends, HTTPException, Request, Body
from pydantic import BaseModel
from .model import DelayModel  # Assuming your model file is named "model.py"
import pandas as pd
from typing import List, Dict

app = FastAPI()

delay_model = DelayModel()  # Initialize your model

# Define a Pydantic model for request data validation
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> dict:
    try:
        data = await request.json()
        df = pd.DataFrame(data["flights"])
        processed_data = delay_model.preprocess(df)  # Here, is_training is False by default
        predictions = delay_model.predict(processed_data)
        return {"predict": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))