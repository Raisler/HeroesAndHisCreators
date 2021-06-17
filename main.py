from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
from actions import vect
from pydantic import BaseModel


class Story(BaseModel):
    story: str


app = FastAPI()
model = pickle.load(open('./finalized_model.sav', 'rb'))


@app.post("/predict/")
def predict(story: Story):
    """
    Returns a prediction of some sentence
    """
    to_predict = vect.transform([story.story])

    prediction = model.predict(to_predict)
    prediction = prediction[0]
    return {'prediction':prediction}


@app.get("/hello-world")
def hello_world():
    """
    Returns Hello World
    """
    return {"Hello": "World"}




