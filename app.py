from fastapi import FastAPI
from pydantic import BaseModel
from model import train_model
import numpy as np

# Define expected input structure
class InputData(BaseModel):
    x: list[float]

app = FastAPI()
X, y, model = train_model()

@app.post("/predict")
async def predict(data: InputData):
    x = np.array(data.x).reshape(1, -1)
    pred = model.predict(x)
    print(f"Received request: {data.x} → Prediction: {int(pred[0])}")
    return {"prediction": int(pred[0])}