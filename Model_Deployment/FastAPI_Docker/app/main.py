import uvicorn
from fastapi import FastAPI
import pickle # loading the pre-trained model saved in the app/wine.pkl file
import numpy as np # for tensor manipulation
from pydantic import BaseModel # way to represent a data point. You can do this by creating a class the subclasses from pydantic's BaseModel and listing each attribute along with its corresponding type.

class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


app = FastAPI(title="Predicting wine by sanjeet's model")

# load the classfier
@app.get('/')
def load_clf():
    with open("wine.pkl", "rb") as model:
        global clf
        clf = pickle.load(model)

# create route
# This function will be run when you visit the /predict endpoint of the server and it expects a Wine data point.

'''
This function is actually very straightforward, first you will convert the information within
 the Wine object into a numpy array of shape (1, 13) and then use the predict method of the 
 classifier to make a prediction for the data point. Notice that the prediction must be casted 
 into a list using the tolist method.
'''

@app.post('/predict')
def make_prediction(data:Wine):
    data_points = np.array([
        [
            data.alcohol,            
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.od280_od315_of_diluted_datas,
            data.proline,
        ]
    ])
    pred = clf.predict(data_points).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}

if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)