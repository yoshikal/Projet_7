from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import joblib
from typing import Union

import pandas as pd

from predict import pretreatment

imput = joblib.load('pkl/imputer.pkl')
dic_le = joblib.load('pkl/dic_le.pkl')
enc_ohe = joblib.load('pkl/enc_ohe.pkl')
imput2 = joblib.load('pkl/imputer2.pkl')
scaler = joblib.load('pkl/scaler.pkl')
best_clf = joblib.load('pkl/LGBM.pkl')

class Client(BaseModel):
    SK_ID_CURR: int
    TARGET: int
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Union[float, None] = None
    AMT_GOODS_PRICE: Union[float, None] = None
    NAME_TYPE_SUITE: Union[str, None] = None
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: Union[float, None] = None
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: Union[str, None] = None
    CNT_FAM_MEMBERS: Union[float, None] = None
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: Union[float, None] = None
    EXT_SOURCE_2: Union[float, None] = None
    EXT_SOURCE_3: Union[float, None] = None
    APARTMENTS_AVG: Union[float, None] = None
    BASEMENTAREA_AVG: Union[float, None] = None
    YEARS_BEGINEXPLUATATION_AVG: Union[float, None] = None
    YEARS_BUILD_AVG: Union[float, None] = None
    COMMONAREA_AVG: Union[float, None] = None
    ELEVATORS_AVG: Union[float, None] = None
    ENTRANCES_AVG: Union[float, None] = None
    FLOORSMAX_AVG: Union[float, None] = None
    FLOORSMIN_AVG: Union[float, None] = None
    LANDAREA_AVG: Union[float, None] = None
    LIVINGAPARTMENTS_AVG: Union[float, None] = None
    LIVINGAREA_AVG: Union[float, None] = None
    NONLIVINGAPARTMENTS_AVG: Union[float, None] = None
    NONLIVINGAREA_AVG: Union[float, None] = None
    APARTMENTS_MODE: Union[float, None] = None
    BASEMENTAREA_MODE: Union[float, None] = None
    YEARS_BEGINEXPLUATATION_MODE: Union[float, None] = None
    YEARS_BUILD_MODE: Union[float, None] = None
    COMMONAREA_MODE: Union[float, None] = None
    ELEVATORS_MODE: Union[float, None] = None
    ENTRANCES_MODE: Union[float, None] = None
    FLOORSMAX_MODE: Union[float, None] = None
    FLOORSMIN_MODE: Union[float, None] = None
    LANDAREA_MODE: Union[float, None] = None
    LIVINGAPARTMENTS_MODE: Union[float, None] = None
    LIVINGAREA_MODE: Union[float, None] = None
    NONLIVINGAPARTMENTS_MODE: Union[float, None] = None
    NONLIVINGAREA_MODE: Union[float, None] = None
    APARTMENTS_MEDI: Union[float, None] = None
    BASEMENTAREA_MEDI: Union[float, None] = None
    YEARS_BEGINEXPLUATATION_MEDI: Union[float, None] = None
    YEARS_BUILD_MEDI: Union[float, None] = None
    COMMONAREA_MEDI: Union[float, None] = None
    ELEVATORS_MEDI: Union[float, None] = None
    ENTRANCES_MEDI: Union[float, None] = None
    FLOORSMAX_MEDI: Union[float, None] = None
    FLOORSMIN_MEDI: Union[float, None] = None
    LANDAREA_MEDI: Union[float, None] = None
    LIVINGAPARTMENTS_MEDI: Union[float, None] = None
    LIVINGAREA_MEDI: Union[float, None] = None
    NONLIVINGAPARTMENTS_MEDI: Union[float, None] = None
    NONLIVINGAREA_MEDI: Union[float, None] = None
    FONDKAPREMONT_MODE: Union[str, None] = None
    HOUSETYPE_MODE: Union[str, None] = None
    TOTALAREA_MODE: Union[float, None] = None
    WALLSMATERIAL_MODE: Union[str, None] = None
    EMERGENCYSTATE_MODE: Union[str, None] = None
    OBS_30_CNT_SOCIAL_CIRCLE: Union[float, None] = None
    DEF_30_CNT_SOCIAL_CIRCLE: Union[float, None] = None
    OBS_60_CNT_SOCIAL_CIRCLE: Union[float, None] = None
    DEF_60_CNT_SOCIAL_CIRCLE: Union[float, None] = None
    DAYS_LAST_PHONE_CHANGE: Union[float, None] = None
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_MON: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Union[float, None] = None

app = FastAPI()

@app.get("/ping")
def pong():
    return {"blabla"}

@app.post("/clients")
async def create_client(client: Client):
    client_dict = client.dict()
    return client_dict

@app.post("/clients_pretrait")
async def create_client_pretrait(client: Client):

    client_dict = jsonable_encoder(client)

    print(client_dict)

    for key, value in client_dict.items():
        client_dict[key] = [value]
     # answer_dict = {k:[v] for (k,v) in jsonable_encoder(answer).items()}
    single_instance = pd.DataFrame.from_dict(client_dict)

    y_pred, y_prob = pretreatment(single_instance, imput, dic_le, enc_ohe, imput2, scaler, best_clf)

    output = {}

    ids = single_instance["SK_ID_CURR"].to_list()
    
    for i, id in enumerate(ids):
        output[id] = y_prob[i] 

    print(output)

    return output


