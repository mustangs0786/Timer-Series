import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import pmdarima as pm
from dtaidistance import dtw
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_test_split(dataframe,Date_column):
    # try:
    st.markdown("<h3 style='text-align: center; color: gray;'>Training and Test data Split </h3>", unsafe_allow_html=True)
    try:
        
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column]).dt.date
    except Exception:
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column])
    dataframe = dataframe.sort_values(by=Date_column)
    tab1, tab2 = st.columns(2)
    with tab1:
        Min_date = st.date_input('Enter Train Starting Date',dataframe[Date_column].min(),min_value=dataframe[Date_column].min(), max_value=dataframe[Date_column].max())
        Max_date = st.date_input('Enter Train End Date',dataframe[Date_column].max(),min_value=dataframe[Date_column].min(), max_value=dataframe[Date_column].max())
        train_df = dataframe[(dataframe[Date_column]>=Min_date) & (dataframe[Date_column]<=Max_date)]
        
        train_df = train_df.sort_values(by=Date_column)
    
        # return train_df,flag
    with tab2:
        Min_date = st.date_input('Enter Test Starting Date',train_df[Date_column].max(),min_value=train_df[Date_column].max(), max_value=dataframe[Date_column].max())
        Max_date = st.date_input('Enter Test End Date',dataframe[Date_column].max(),min_value=train_df[Date_column].min(), max_value=dataframe[Date_column].max())
        test_df = dataframe[(dataframe[Date_column]>Min_date) & (dataframe[Date_column]<=Max_date)]
        test_df = test_df.sort_values(by=Date_column)
    return train_df,test_df
    # except Exception as e :
    #     st.text(e)
    #     st.stop()

def train_test_eda(train_df,test_df,Target_column,Date_column):
    st.markdown("<h3 style='text-align: center; color: gray;'>Chart of Train and Test Data </h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df[Date_column], y=train_df[Target_column],
                    mode='lines+markers',
                    name='Train Data'))
    fig.add_trace(go.Scatter(x=test_df[Date_column], y=test_df[Target_column],
                    mode='lines+markers',
                    name='Test Data'))
    fig.update_layout(   
                            title=f'{Target_column} Data',
                            xaxis_title=Target_column,
                            yaxis_title=Date_column,  

                            yaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),

                            xaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),


                            plot_bgcolor='white',
                            showlegend=True,)

    st.plotly_chart(fig, use_container_width = True)

    st.warning(" ")

def fbprophet_model(train_df,test_df,Target_column,Date_column):
    train_df = train_df.rename(columns={Date_column: 'ds',
                        Target_column: 'y'})
    test_df = test_df.rename(columns={Date_column: 'ds',
                        Target_column: 'y'})

    my_model = Prophet()
    my_model.fit(train_df)
    future_dates = test_df[['ds']]
    forecast = my_model.predict(future_dates)
    return forecast,train_df,test_df


def auto_arima_model(forecast,train_df,test_df,Target_column,Date_column):
    modl = pm.auto_arima(train_df[['y']], start_p=0, start_q=0, start_P=0, start_Q=0,
                     max_p=15, max_q=15, max_P=15, max_Q=15, seasonal=True,
                     stepwise=False, suppress_warnings=True, D=10, max_D=10,
                     error_action='ignore')

    preds, conf_int = modl.predict(n_periods=test_df.shape[0], return_conf_int=True)
    forecast['arima_pred'] = list(preds)
    return forecast,train_df,test_df

def rf_modeling(forecast,train_df,test_df,Target_column,Date_column):
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    train_df['Month'] = train_df['ds'].dt.month
    train_df['quarter'] = train_df['ds'].dt.quarter
    X_train,y_train = train_df[['Month','quarter']],train_df['y']

    test_df['ds'] = pd.to_datetime(test_df['ds'])
    test_df['Month'] = test_df['ds'].dt.month
    test_df['quarter'] = test_df['ds'].dt.quarter


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [3,7,8,10,15,20,25]
    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5,6]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2,]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
            #    'min_samples_split': min_samples_split,
            #    'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf_Model = RandomForestRegressor()

    rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose=2, n_jobs = 4)
    rf_RandomGrid.fit(X_train, y_train)
    rf_Model = RandomForestRegressor(**rf_RandomGrid.best_params_)
    rf_Model.fit(X_train, y_train)
    rf_pred = rf_Model.predict(test_df[['Month','quarter']])
    forecast['Rf_pred'] = list(rf_pred)
    forecast['actual'] = list(test_df['y'])

    rf_mape = np.mean(abs(forecast['actual']-forecast['Rf_pred'])/forecast['actual'])
    fb_mape = np.mean(abs(forecast['actual']-forecast['yhat'])/forecast['actual'])
    arima_mape = np.mean(abs(forecast['actual']-forecast['arima_pred'])/forecast['actual'])

    Model_result = pd.DataFrame()
    Model_result['Models'] = ['Arima','Fbprophet','Random_forest']
    Model_result['Mape'] = [arima_mape,fb_mape,rf_mape]
    
    

    ar_distance = dtw.distance(list(forecast['arima_pred']),list(forecast['actual']))
    fb_distance = dtw.distance(list(forecast['yhat']),list(forecast['actual']))
    rf_distance = dtw.distance(list(forecast['Rf_pred']),list(forecast['actual']))
    Model_result['Cost'] = [ar_distance,fb_distance,rf_distance]
    Model_result['cost_name'] = ['arima_pred','yhat','Rf_pred']
    Model_result = Model_result.sort_values(by='Cost')
    Model_result = Model_result.head(2)
    st.dataframe(Model_result)

    first = list(Model_result['Cost'])[0]
    second = list(Model_result['Cost'])[1]

    first_pred = list(Model_result['cost_name'])[0]
    second_Pred = list(Model_result['cost_name'])[1]

    first_wgt = 1 - first/(first+second)
    second_wgt = 1 - second/(first+second)
    forecast['Ensemble_pred'] = forecast[first_pred]*first_wgt + forecast[second_Pred]*second_wgt
    st.markdown("<h3 style='text-align: center; color: gray;'>Performance of Various Models </h3>", unsafe_allow_html=True)
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df['y'],
    #                 mode='lines',
    #                 name='Train Data'))
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'],
                    mode='lines',
                    name='Test Data'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    name='Fb_prophet Pred Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['arima_pred'],
                    mode='lines',
                    name='Arima Pred Data'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Ensemble_pred'],
                mode='lines',
                name='Ensemble Pred Data'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Rf_pred'],
                mode='lines',
                name='RF Pred Data'))

    fig.update_layout(   
                            title=f'{Target_column} Data',
                            xaxis_title=Target_column,
                            yaxis_title=Date_column,  

                            yaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),

                            xaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),


                            plot_bgcolor='white',
                            showlegend=True,)

    st.plotly_chart(fig, use_container_width = True)

    st.warning(" ")

    st.markdown("<h3 style='text-align: center; color: gray;'>Final Prediction Chart </h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df['y'],
                    mode='lines',
                    name='Train Data'))
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'],
                    mode='lines',
                    name='Test Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Ensemble_pred'],
                mode='lines',
                name='Ensemble Pred Data'))


    fig.update_layout(   
                            title=f'{Target_column} Data',
                            xaxis_title=Target_column,
                            yaxis_title=Date_column,  

                            yaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),

                            xaxis=dict(                         
                            showticklabels=True,
                            linecolor='rgb(204, 204, 204)',
                            linewidth=2,
                            ticks='outside',  ),


                            plot_bgcolor='white',
                            showlegend=True,)

    st.plotly_chart(fig, use_container_width = True)

    st.warning(" ")
    Model_result = pd.DataFrame()
    Model_result['Models'] = ['Arima','Fbprophet','Random_forest','Ensemble_Model']
    
    
    en_mape = np.mean(abs(forecast['actual']-forecast['Ensemble_pred'])/forecast['actual'])
    ensemble_distance = dtw.distance(list(forecast['Ensemble_pred']),list(forecast['actual']))

    Model_result['Mape'] = [arima_mape,fb_mape,rf_mape,en_mape]
    Model_result['Cost'] = [ar_distance,fb_distance,rf_distance,ensemble_distance]
    Model_result = Model_result.sort_values(by='Mape')
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center; color: gray;'>Model Performace </h3>", unsafe_allow_html=True)
        st.dataframe(Model_result)
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
    forecast = forecast[['ds','actual','yhat','arima_pred','Rf_pred','Ensemble_pred']]
    with col2:
        st.markdown("<h3 style='text-align: center; color: gray;'>Final Prediction Dataframe </h3>", unsafe_allow_html=True)
        st.dataframe(forecast)

    @st.cache
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(forecast)

    st.download_button(
        label="Download forecast Result as CSV",
        data=csv,
        file_name='forecast_Result.csv',
        mime='text/csv',
        )
