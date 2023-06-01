# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

def pretreatment(client, imput, dic_enc_le, enc_ohe, imput2, scaler, best_clf):
    
    client_ = client.copy()
    
    #imputation
    client_.iloc[:,:] = imput.transform(client_)
    
    #encodage
    ##le
    
    for col, enc_le in dic_enc_le.items():
        client_[col] = enc_le.transform(client_[col])
        
    ##ohe
    list_ohe = list(client_.select_dtypes(include=['object']).columns) 

    print(list_ohe)

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
    
    X = client_.iloc[:,2:]
    
    #y = client.iloc[:,1]
    
    #feature engineering
    X['annuity_income_ratio'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['credit_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['credit_goods_price_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['credit_downpayment'] = X['AMT_GOODS_PRICE'] - X['AMT_CREDIT']
    X['AGE_INT'] = X['DAYS_BIRTH'] / 365
    
    #feature scaling
    X = scaler.transform(X)
    
    #pred
    #print(best_clf.predict_proba(X))
    y_prob = best_clf.predict_proba(X)[:,1]
    #print(y_prob)
    y_pred = (y_prob > 0.55).astype(int)
    #print(y_pred)

    return y_pred, y_prob