import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from banknotes import BankNote 


# create object and load model
app = FastAPI()
classifier = joblib.load('classifier_model.joblib')

# index route
@app.get("/")
def home():
    return {"message" : "Hello World !"}

@app.get("/{name}")
def get_name(name:str):
    return {"message" : f"Welcome {name} !"}

@app.post("/predict")
def predict(data:BankNote):
    data = data.dict()
    print(data)
    variance = data["variance"]
    print(variance)
    skewness = data["skewness"]
    print(skewness)
    curtosis = data["curtosis"]
    print(curtosis)
    entropy = data["entropy"]
    print(entropy)
    print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    print("Prediction..........")
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if (prediction[0] > 0.5):
        prediction = "Fake note"
    else:
        prediction = "Genuine note"
    

    return {
        'prediction' : prediction
    }

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8000)

