# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
from numpy import dtype

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

dtypes_dict = {
 'NAME_CONTRACT_TYPE': dtype('O'),
 'CODE_GENDER': dtype('O'),
 'FLAG_OWN_CAR': dtype('O'),
 'FLAG_OWN_REALTY': dtype('O'),
 'CNT_CHILDREN': dtype('int64'),
 'AMT_INCOME_TOTAL': dtype('float64'),
 'AMT_CREDIT': dtype('float64'),
 'AMT_ANNUITY': dtype('float64'),
 'AMT_GOODS_PRICE': dtype('float64'),
 'NAME_TYPE_SUITE': dtype('O'),
 'NAME_INCOME_TYPE': dtype('O'),
 'NAME_EDUCATION_TYPE': dtype('O'),
 'NAME_FAMILY_STATUS': dtype('O'),
 'NAME_HOUSING_TYPE': dtype('O'),
 'REGION_POPULATION_RELATIVE': dtype('float64'),
 'DAYS_BIRTH': dtype('int64'),
 'DAYS_EMPLOYED': dtype('int64'),
 'DAYS_REGISTRATION': dtype('float64'),
 'DAYS_ID_PUBLISH': dtype('int64'),
 'OWN_CAR_AGE': dtype('float64'),
 'FLAG_MOBIL': dtype('int64'),
 'FLAG_EMP_PHONE': dtype('int64'),
 'FLAG_WORK_PHONE': dtype('int64'),
 'FLAG_CONT_MOBILE': dtype('int64'),
 'FLAG_PHONE': dtype('int64'),
 'FLAG_EMAIL': dtype('int64'),
 'OCCUPATION_TYPE': dtype('O'),
 'CNT_FAM_MEMBERS': dtype('float64'),
 'REGION_RATING_CLIENT': dtype('int64'),
 'REGION_RATING_CLIENT_W_CITY': dtype('int64'),
 'WEEKDAY_APPR_PROCESS_START': dtype('O'),
 'HOUR_APPR_PROCESS_START': dtype('int64'),
 'REG_REGION_NOT_LIVE_REGION': dtype('int64'),
 'REG_REGION_NOT_WORK_REGION': dtype('int64'),
 'LIVE_REGION_NOT_WORK_REGION': dtype('int64'),
 'REG_CITY_NOT_LIVE_CITY': dtype('int64'),
 'REG_CITY_NOT_WORK_CITY': dtype('int64'),
 'LIVE_CITY_NOT_WORK_CITY': dtype('int64'),
 'ORGANIZATION_TYPE': dtype('O'),
 'EXT_SOURCE_1': dtype('float64'),
 'EXT_SOURCE_2': dtype('float64'),
 'EXT_SOURCE_3': dtype('float64'),
 'APARTMENTS_AVG': dtype('float64'),
 'BASEMENTAREA_AVG': dtype('float64'),
 'YEARS_BEGINEXPLUATATION_AVG': dtype('float64'),
 'YEARS_BUILD_AVG': dtype('float64'),
 'COMMONAREA_AVG': dtype('float64'),
 'ELEVATORS_AVG': dtype('float64'),
 'ENTRANCES_AVG': dtype('float64'),
 'FLOORSMAX_AVG': dtype('float64'),
 'FLOORSMIN_AVG': dtype('float64'),
 'LANDAREA_AVG': dtype('float64'),
 'LIVINGAPARTMENTS_AVG': dtype('float64'),
 'LIVINGAREA_AVG': dtype('float64'),
 'NONLIVINGAPARTMENTS_AVG': dtype('float64'),
 'NONLIVINGAREA_AVG': dtype('float64'),
 'APARTMENTS_MODE': dtype('float64'),
 'BASEMENTAREA_MODE': dtype('float64'),
 'YEARS_BEGINEXPLUATATION_MODE': dtype('float64'),
 'YEARS_BUILD_MODE': dtype('float64'),
 'COMMONAREA_MODE': dtype('float64'),
 'ELEVATORS_MODE': dtype('float64'),
 'ENTRANCES_MODE': dtype('float64'),
 'FLOORSMAX_MODE': dtype('float64'),
 'FLOORSMIN_MODE': dtype('float64'),
 'LANDAREA_MODE': dtype('float64'),
 'LIVINGAPARTMENTS_MODE': dtype('float64'),
 'LIVINGAREA_MODE': dtype('float64'),
 'NONLIVINGAPARTMENTS_MODE': dtype('float64'),
 'NONLIVINGAREA_MODE': dtype('float64'),
 'APARTMENTS_MEDI': dtype('float64'),
 'BASEMENTAREA_MEDI': dtype('float64'),
 'YEARS_BEGINEXPLUATATION_MEDI': dtype('float64'),
 'YEARS_BUILD_MEDI': dtype('float64'),
 'COMMONAREA_MEDI': dtype('float64'),
 'ELEVATORS_MEDI': dtype('float64'),
 'ENTRANCES_MEDI': dtype('float64'),
 'FLOORSMAX_MEDI': dtype('float64'),
 'FLOORSMIN_MEDI': dtype('float64'),
 'LANDAREA_MEDI': dtype('float64'),
 'LIVINGAPARTMENTS_MEDI': dtype('float64'),
 'LIVINGAREA_MEDI': dtype('float64'),
 'NONLIVINGAPARTMENTS_MEDI': dtype('float64'),
 'NONLIVINGAREA_MEDI': dtype('float64'),
 'FONDKAPREMONT_MODE': dtype('O'),
 'HOUSETYPE_MODE': dtype('O'),
 'TOTALAREA_MODE': dtype('float64'),
 'WALLSMATERIAL_MODE': dtype('O'),
 'EMERGENCYSTATE_MODE': dtype('O'),
 'OBS_30_CNT_SOCIAL_CIRCLE': dtype('float64'),
 'DEF_30_CNT_SOCIAL_CIRCLE': dtype('float64'),
 'OBS_60_CNT_SOCIAL_CIRCLE': dtype('float64'),
 'DEF_60_CNT_SOCIAL_CIRCLE': dtype('float64'),
 'DAYS_LAST_PHONE_CHANGE': dtype('float64'),
 'FLAG_DOCUMENT_2': dtype('int64'),
 'FLAG_DOCUMENT_3': dtype('int64'),
 'FLAG_DOCUMENT_4': dtype('int64'),
 'FLAG_DOCUMENT_5': dtype('int64'),
 'FLAG_DOCUMENT_6': dtype('int64'),
 'FLAG_DOCUMENT_7': dtype('int64'),
 'FLAG_DOCUMENT_8': dtype('int64'),
 'FLAG_DOCUMENT_9': dtype('int64'),
 'FLAG_DOCUMENT_10': dtype('int64'),
 'FLAG_DOCUMENT_11': dtype('int64'),
 'FLAG_DOCUMENT_12': dtype('int64'),
 'FLAG_DOCUMENT_13': dtype('int64'),
 'FLAG_DOCUMENT_14': dtype('int64'),
 'FLAG_DOCUMENT_15': dtype('int64'),
 'FLAG_DOCUMENT_16': dtype('int64'),
 'FLAG_DOCUMENT_17': dtype('int64'),
 'FLAG_DOCUMENT_18': dtype('int64'),
 'FLAG_DOCUMENT_19': dtype('int64'),
 'FLAG_DOCUMENT_20': dtype('int64'),
 'FLAG_DOCUMENT_21': dtype('int64'),
 'AMT_REQ_CREDIT_BUREAU_HOUR': dtype('float64'),
 'AMT_REQ_CREDIT_BUREAU_DAY': dtype('float64'),
 'AMT_REQ_CREDIT_BUREAU_WEEK': dtype('float64'),
 'AMT_REQ_CREDIT_BUREAU_MON': dtype('float64'),
 'AMT_REQ_CREDIT_BUREAU_QRT': dtype('float64'),
 'AMT_REQ_CREDIT_BUREAU_YEAR': dtype('float64')} 

def pretreatment(client, imput, dic_enc_le, enc_ohe, imput2, scaler, best_clf):
    
    client_ = client.copy()

    client_ = client_.reset_index(drop=True)

    client_ = client_.replace('', np.nan)

    client_ = client_.astype(dtypes_dict)
    
    #imputation
    client_.iloc[:,:] = imput.transform(client_)
    
    #encodage
    ##le
    for col, enc_le in dic_enc_le.items():
        client_[col] = enc_le.transform(client_[col])
        
    ##ohe
    list_ohe = list(client_.select_dtypes(include=['object']).columns) 

    ohe = enc_ohe.transform(client_[list_ohe])
    ohe_df = pd.DataFrame(ohe.todense(), columns=enc_ohe.get_feature_names_out().tolist())
    client_ = pd.concat([client_, ohe_df], axis=1)
    
    client_.drop(client_.dtypes[client_.dtypes == 'object'].index.tolist(), inplace=True, axis=1)
    
    #anomalies
    ##create an anomalous flag column
    client_['DAYS_EMPLOYED_ANOM'] = client_["DAYS_EMPLOYED"] == 365243

    ##replace the anomalous values with nan
    client_['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    
    #imputation
    client_.iloc[:,:] = imput2.transform(client_)
                
    client_['DAYS_BIRTH'] = abs(client_['DAYS_BIRTH'])

    X = client_
    
    #feature engineering
    X['annuity_income_ratio'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['credit_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['credit_goods_price_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['credit_downpayment'] = X['AMT_GOODS_PRICE'] - X['AMT_CREDIT']
    X['AGE_INT'] = X['DAYS_BIRTH'] / 365

    Xcol = X.columns
    
    #feature scaling
    X = scaler.transform(X)
    
    #pred
    y_prob = best_clf.predict_proba(X)[:,1]
    y_pred = (y_prob > 0.55).astype(int)

    return y_pred, y_prob, X, Xcol
