from fastapi import FastAPI, Path, Body
import uvicorn
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Sample FastAPI",
              description="For education", version="1.0")

model = joblib.load('c45_classifier.pkl')

class Input(BaseModel):
    pregnancies : float = Field(0, example=1) 
    glocose: float = Field(0, example=0) 
    blood_pressure: float = Field(0, example=0) 
    skin_thickness: float = Field(0, example=0)
    insulin: float = Field(0)
    diabetes_ped_func: float = Field(0, example=0)
    age: float = Field(..., example=30)
    bmi: float = Field(..., example=13)
    
@app.post("/prediction")
def prediction(data: Input):
    data = dict(data)
    data_arr = np.array([v for (_,v)in data.items()]) 
    result = model.predict(data_arr.reshape(1,-1))
    result = {"status": "OK", "result": int(result)}
    return result


if __name__ == "__main__":
	uvicorn.run("main:app", port=5001, log_level="info", reload=True, host="0.0.0.0")