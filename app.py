from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib


app = FastAPI(title= "Default Prediction")

model = load(pathlib.Path("models/XGB_V_fold2.xgb"))

class InputData(BaseModel):

    