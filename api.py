import numpy as np
import pickle
from pydantic import BaseModel
import pandas as pd 

# loading the saved model
loaded_model = pickle.load(open('credit_scoring_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

class CreditScore(BaseModel) : 
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: int
    NAME_EDUCATION_TYPE: int
    NAME_FAMILY_STATUS: int
    NAME_HOUSING_TYPE: int
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    CNT_FAM_MEMBERS: float


from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def root():
    return {"Response":"Wokring Great!"}

@app.post('/predict')
async def credit_scoring(item:CreditScore) : 
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    scaled_input = loaded_scaler.transform(df)
    results = loaded_model.predict(scaled_input)
    if results == 0 :
        return "No Risk"
    elif results == 1 :
        return "High Risk"