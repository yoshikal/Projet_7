from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import joblib
from typing import Dict, Any
import pandas as pd
from predict import pretreatment
import json

imput = joblib.load('pkl/imputer.pkl')
dic_le = joblib.load('pkl/dic_le.pkl')
enc_ohe = joblib.load('pkl/enc_ohe.pkl')
imput2 = joblib.load('pkl/imputer2.pkl')
scaler = joblib.load('pkl/scaler.pkl')
best_clf = joblib.load('pkl/LGBM.pkl')

app = FastAPI()

@app.get("/ping")
def pong():
    return {"blabla"}

@app.post("/clients")
async def create_client(client: Dict[Any, Any]):
    client_dict = client.dict()
    return client_dict

@app.post("/clients_pretrait")
async def create_client_pretrait(client: Dict[Any, Any]):

    client_dict = jsonable_encoder(client)

    for key, value in client_dict.items():
        client_dict[key] = [value]

    single_instance = pd.DataFrame.from_dict(client_dict)

    y_pred, y_prob, X, Xcol = pretreatment(single_instance, imput, dic_le, enc_ohe, imput2, scaler, best_clf)

    #print(y_prob)

    output = {}

    #ids = single_instance["SK_ID_CURR"].to_list()
    
    #for i, id in enumerate(ids):
     #   output[id] = y_prob[i] 

    #output['predict_proba'] = y_prob[0]

    output['predict_proba'] = y_prob[0]

    df = pd.DataFrame(X)
    X = df.to_json(orient='index')
    output['X'] = X

    print(Xcol)
    df2 = pd.DataFrame(columns=Xcol)
    #print(df2)
    Xcol = df2.to_json(orient='split')
    output['Xcol'] = Xcol

    #best_estimator = best_clf.best_estimator_
    #df3 = pd.DataFrame(columns=best_estimator)
    #print(df3)
    #best_estimator = df3.to_json(orient='split')


    #output['best_estimator'] = best_estimator

    return output
