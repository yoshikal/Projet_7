import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL")

def test_api_prediction():

    #print('Reading test data')
    # Load the data sample from the CSV file
    data_sample = pd.read_csv("data_sample.csv", index_col='SK_ID_CURR')
    #print(data_sample)
    #print(data_sample.index) 

    data_sample = data_sample.reset_index(drop=True)
    data_sample = data_sample.iloc[0]
    #print(data_sample.to_dict())
    data_sample = data_sample.fillna('')

    # Send a POST request to the API endpoint with the data as the request body
    response = requests.post(API_URL, json=data_sample.to_dict())

    # Check that the response status code is 200 OK
    assert response.status_code == 200

    # Check that the response body contains valid JSON data
    assert response.headers["content-type"] == "application/json"

    # Parse the JSON data from the response body
    prediction = response.json().get("predict_proba")
    #print(prediction)

    # Check that the prediction value s a float
    assert isinstance(prediction, float)

    assert prediction == 0.15600481776483213

# test_api_prediction()  # to be able to test it with Python, not only Pytest

def test_api_prediction2():

    #print('Reading test data')
    # Load the data sample from the CSV file
    data_sample = pd.read_csv("data_sample2.csv", index_col='SK_ID_CURR')
    #print(data_sample)
    #print(data_sample.index) 

    data_sample = data_sample.reset_index(drop=True)
    data_sample = data_sample.iloc[0]
    #print(data_sample.to_dict())
    data_sample = data_sample.fillna('')

    # Send a POST request to the API endpoint with the data as the request body
    response = requests.post(API_URL, json=data_sample.to_dict())

    # Check that the response status code is 200 OK
    assert response.status_code == 200

    # Check that the response body contains valid JSON data
    assert response.headers["content-type"] == "application/json"

    # Parse the JSON data from the response body
    prediction = response.json().get("predict_proba")
    #print(prediction)

    # Check that the prediction value s a float
    assert isinstance(prediction, float)

    assert prediction == 0.14280952657454374
