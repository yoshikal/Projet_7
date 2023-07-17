import streamlit as st 
import json
import pandas as pd
import numpy as np
import joblib
import requests
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.graph_objects as go
import time
import shap
best_clf = joblib.load('pkl/LGBM.pkl')

@st.cache_data
def fetch_data():

    sample_nrows = 10000

    #load data
    sample_df = pd.read_csv('data/application_test.csv', nrows=50, index_col='SK_ID_CURR')
    float_cols = [c for c in sample_df if sample_df[c].dtype == "float16"]
    float16_cols = {c: np.float16 for c in float_cols}
    Xtest = pd.read_csv('data/application_test.csv', engine='c', dtype=float16_cols, nrows=sample_nrows, index_col='SK_ID_CURR')
    
    Xtest = Xtest.replace(np.nan, '')

    return Xtest

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df
    
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def gauge_animated_figure(score, score_threshold, frame_duration=0.1):
    # Define the initial sscore value
    sscore = 0

    # Create the initial Plotly figure with the sscore value
    gaugefig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sscore,
        title={'text': "Score crédit"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "yellow"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, score_threshold], 'color': 'red'},
                {'range': [score_threshold, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "yellow", 'width': 4},
                'thickness': 0.75,
                'value': sscore
            }
        }
    ))
    gaugefig.update_layout(height=350)

    # Create the Streamlit plotly_chart object with the initial Plotly figure
    plotly_chart = st.plotly_chart(gaugefig, use_container_width=True)

    # Update the sscore value in the loop and update the Plotly figure and Streamlit plotly_chart object
    for sscore in range(1, int(score), 1):
        gaugefig.update_traces(
            value=sscore,
            gauge={
                'threshold': {
                    'value': sscore
                }
            }
        )
        plotly_chart.plotly_chart(gaugefig, use_container_width=True)
        time.sleep(frame_duration)

    # Final: real score
    gaugefig.update_traces(
        value=score,
        gauge={
            'threshold': {
                'value': score
            }
        }
    )
    plotly_chart.plotly_chart(gaugefig, use_container_width=True)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def extractDigits(lst):
    l = [el for el in lst]
    return np.array([l])


def main():
    # Page config
    st.set_page_config(page_title="Scoring credit")

    #API_endpoint = "http://127.0.0.1:8000/clients_pretrait"
    API_endpoint = "https://scoringmodeloc.azurewebsites.net/clients_pretrait"

    # Fetch data
    Xtrain = fetch_data()



    # Dashboard structure
    st.title("Scoring crédit")

    tab1, tab3, tab4 = st.tabs(['Visualiser le dataframe',
                               'Visualiser le score',
                               'Comparer les données d\'un client à un autre'])
    
    ########## Sidebar select client ##########
    #print(Xtrain.index)
    clist = ['Client ' + str(x + 1) for x in Xtrain.index]
    clist = ['<Sélectionnez un client>'] + clist
    client_id = clist[0]  # default
    score = 0
    client_id = st.sidebar.selectbox('Sélectionnez le client (ou entrez son identifiant)', clist, key=1000)
    if client_id == clist[0]:
        pass
    else:
        # Client data
        data_sample = Xtrain.iloc[[clist.index(client_id) - 1]]
        data_sample = data_sample.reset_index(drop=True)
        gender = data_sample['CODE_GENDER'].apply(lambda x: 'M' if x == 1.0 else 'F')


        # Display basic client data
        st.sidebar.subheader('Données de base')
        st.sidebar.metric('Genre', '%s' % gender.values[0])
        st.sidebar.metric('Age', '%d ans' % np.ceil(-1 * data_sample['DAYS_BIRTH'] / 365.25))
        st.sidebar.metric('Revenus', '{:,}$'.format(int(data_sample['AMT_INCOME_TOTAL'].values[0])).replace(',', ' '))
        st.sidebar.metric("Nombre d'enfants", '%d' % data_sample['CNT_CHILDREN']) 


    with tab1:
        st.dataframe(filter_dataframe(Xtrain))

    with tab3:
            if client_id == clist[0]:
                st.write('_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
            else:
                col1, col2 = st.columns([0.5, 1])
            
                with col1:
                    predict_btn = st.button('Calcul du score', type='primary')

                    score_threshold = st.slider('Seuil de décision', min_value=0, max_value=100, value=int(80), step=1,
                                            format='%d')

                if predict_btn:

                    data_sample = data_sample.iloc[0]

                    #print(data_sample)
                    #print(data_sample.to_dict())

                    response = requests.post(API_endpoint, json=data_sample.to_dict())

                    #print(response.json())
                    #print(response.json().get("predict_proba"))
                    #pred = response.json()[0]

                    pred = response.json().get("predict_proba")

                    score = (1 - pred) * 100

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        gauge_animated_figure(score, 50, frame_duration=0.05)

                        with col2:

                            st.sidebar.metric('Score :', value=('%d/100' % score))
                            if score < score_threshold:
                                st.markdown(
                                    '<h1 style="text-align:center;color:red;font-weight:700;font-size:26px">Risque de défaut élevé.  \n\nRecommandation : refuser le crédit.</h1>',
                                    unsafe_allow_html=True)
                            elif score >= score_threshold:
                                st.markdown(
                                    '<h1 style="text-align:center;color:green;font-weight:700;font-size:26px">Risque de défaut faible.  \n\nRecommandation : accorder le crédit.</h1>',
                                    unsafe_allow_html=True)
                            # st.balloons()
                            st.write("<center>Probabilité de défaut de remboursement : %.01f%%</center>" % (pred * 100), unsafe_allow_html=True)

                    #forceplot
                    X = response.json().get("X")
                    #print(X)
                    X  = json.loads(X)
                    X = list(X["0"].values())
                    X = extractDigits(X)
                    print(X)

                    Xcol = response.json().get("Xcol")
                    Xcol  = json.loads(Xcol)
                    Xcol = list(Xcol["columns"])
                    #print(Xcol)
                    
                    #best_estimator = response.json().get("best_estimator")
                    #print(best_estimator)

                    best_estimator = best_clf.best_estimator_
                    explainer = shap.TreeExplainer(best_estimator)
                    shap_values = explainer.shap_values(X)
                    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], Xcol))



        


        






if __name__ == '__main__':
    main()