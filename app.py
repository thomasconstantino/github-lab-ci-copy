from fastapi import FastAPI, Request
from model import train_model
import numpy as np

app = FastAPI()
X, y, model = train_model()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    x = np.array(data["x"]).reshape(1, -1)
    pred = model.predict(x)
    return {"prediction": int(pred[0])}
