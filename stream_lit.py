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
import plotly.figure_factory as ff
import plotly.express as px
best_clf = joblib.load('pkl/LGBM.pkl')

@st.cache_data
def fetch_data():

    sample_nrows = 10000

    #load data
    sample_df = pd.read_csv('application_test.csv', nrows=50, index_col='SK_ID_CURR')
    float_cols = [c for c in sample_df if sample_df[c].dtype == "float16"]
    float16_cols = {c: np.float16 for c in float_cols}
    Xtest = pd.read_csv('application_test.csv', engine='c', dtype=float16_cols, nrows=sample_nrows, index_col='SK_ID_CURR')
    
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

def show_client_data(data_client, data_group, varlist):
    theme_plotly = None  # None or 'streamlit'

    cols = st.columns(1)
    for i in range(len(cols)):
        with cols[i]:
            options = ['<Sélectionnez une variable>'] + varlist
            var = st.selectbox(f'Sélectionnez la variable à visualiser', options, key=i)
            if var != options[0]:

                # === Distribution plot, for continuous variables
                if data_group[var].nunique() > 2:
                    
                    hist_data = [data_group.loc[:, var]]
                    group_labels = ['Tous statuts']
                    colors = ['darkorange']
                    fig = ff.create_distplot(hist_data, group_labels, colors=colors, show_hist=False, show_rug=False,
                                             curve_type="kde", histnorm="probability")
                    # Add client line
                    fig.add_vline(x=data_client[var][0], line_dash='solid', line_color='yellow', line_width=3,
                                  annotation=dict(text="Client", font_size=18, showarrow=True, arrowhead=1, ax=0,
                                                  ay=-20, arrowcolor='white'), annotation_position='top')
                    # Set layout
                    # fig.update_layout(yaxis_type="log")
                    fig.update_traces(line=dict(width=3))  # only if show_hist=False
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)))
                    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='left', x=0, font=dict(size=16)))
                    fig.update_yaxes(showticklabels=False, showgrid=False, title='Densité', title_font=dict(size=18),
                                     tickfont=dict(size=18))

                # === Bar plot, for binary variables
                elif data_group[var].nunique() <= 2:
                    client_val = data_client[var][0]

                    # Specific labels for one variable
                    if var == 'CODE_GENDER':
                        val0 = 'F '
                        val1 = 'M '
                    else:
                        val0 = 'Non '
                        val1 = 'Oui '

                    
                    counts = data_group[var].value_counts()
                    counts_df = pd.DataFrame({'value': counts.index, 'count': counts.values})
                    fig = px.bar(counts_df, x='count', y='value', orientation='h',
                                     color_discrete_sequence=['darkorange'])
                    # Add client arrow
                    fig.add_annotation(x=counts_df.loc[counts_df['value'] == client_val, 'count'].iloc[0],
                                           y=client_val, ax=40, ay=0, text="Client", font_size=18, showarrow=True,
                                           arrowhead=1, arrowcolor='white')
                    # Set layout
                    fig.update_layout(
                        legend=dict(orientation='h', yanchor='top', y=1.15, xanchor='left', x=0, title='', font=dict(size=16)))
                    fig.update_layout(yaxis=dict(tickmode='array', ticktext=[val0, val1], tickvals=[0, 1]))
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)), yaxis=dict(tickfont=dict(size=18)))
                    fig.update_xaxes(title='Nombre de clients')
                    fig.update_yaxes(showticklabels=True, showgrid=False, title='')

                # Show plot
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

                # Show info about var and group
                client_val = data_client[var][0]
                group_mean = data_group[var].mean()
                group_std = data_group[var].std()
                group_median = data_group[var].median()
                rows = [
                    f'**Client**: {client_val:.02f}',
                    f'**Moyenne groupe**: {group_mean:.02f} (+/- {group_std:.02f})',
                    f'**Médiane groupe**: {group_median:.02f}'
                ]
                st.write('  \n'.join(rows))


def main():
    # Page config
    st.set_page_config(page_title="Scoring credit")

    #API_endpoint = "http://127.0.0.1:8000/clients_pretrait"
    API_endpoint = "https://scoringmodeloc.azurewebsites.net/clients_pretrait"

    # Fetch data
    Xtest = fetch_data()
    varlist = sorted(Xtest.columns)



    # Dashboard structure
    st.title("Scoring crédit")

    tab1, tab3, tab4 = st.tabs(['Visualiser le dataframe',
                               'Visualiser le score',
                               'Comparer les données d\'un client à un autre'])
    
    ########## Sidebar select client ##########
    #print(Xtrain.index)
    clist = ['Client ' + str(x + 1) for x in Xtest.index]
    clist = ['<Sélectionnez un client>'] + clist
    client_id = clist[0]  # default
    score = 0
    client_id = st.sidebar.selectbox('Sélectionnez le client (ou entrez son identifiant)', clist, key=1000)
    if client_id == clist[0]:
        pass
    else:
        # Client data
        data_sample = Xtest.iloc[[clist.index(client_id) - 1]]
        data_sample = data_sample.reset_index(drop=True)
        gender = data_sample['CODE_GENDER'].apply(lambda x: 'M' if x == 1.0 else 'F')


        # Display basic client data
        st.sidebar.subheader('Données de base')
        st.sidebar.metric('Genre', '%s' % gender.values[0])
        st.sidebar.metric('Age', '%d ans' % np.ceil(-1 * data_sample['DAYS_BIRTH'] / 365.25))
        st.sidebar.metric('Revenus', '{:,}$'.format(int(data_sample['AMT_INCOME_TOTAL'].values[0])).replace(',', ' '))
        st.sidebar.metric("Nombre d'enfants", '%d' % data_sample['CNT_CHILDREN']) 


    with tab1:
        st.dataframe(filter_dataframe(Xtest))

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
                    #print(shap_values)
                    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], Xcol))

    with tab4:
        if client_id == clist[0]:
            st.write('_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            show_client_data(data_sample, Xtest, varlist)





        


        






if __name__ == '__main__':
    main()
