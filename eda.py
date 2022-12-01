import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

def eda(Date_column,Target_column,dataframe):
    st.markdown("<h3 style='text-align: center; color: gray;'>Chart of Target Column Over Date column </h3>", unsafe_allow_html=True)
    dataframe[Date_column] = pd.to_datetime(dataframe[Date_column]).dt.date
    
    dataframe = dataframe.sort_values(by=Date_column)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[Date_column], y=dataframe[Target_column],
                    mode='lines+markers',
                    name='lines+markers'))

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
                            showlegend=False,)

    st.plotly_chart(fig, use_container_width = True)

    st.warning(" ")
    st.markdown("<h3 style='text-align: center; color: gray;'>Month-Yearly Chart of Target Column Over Date column </h3>", unsafe_allow_html=True)
def month_year_eda(Date_column,Target_column,dataframe):
    try:
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column])#.dt.date
        dataframe['Month'] = dataframe[Date_column].dt.month
        dataframe['Year'] = dataframe[Date_column].dt.year
        dataframe = dataframe.groupby(['Year','Month'],as_index=False)[Target_column].sum()
        dataframe['Month_Year'] = dataframe['Year'].astype('str')+"-"+dataframe['Month'].astype('str')
        dataframe['Month_Year'] = pd.to_datetime(dataframe['Month_Year'], format="%Y-%m")
        dataframe = dataframe.groupby(['Month_Year'],as_index=False)[Target_column].sum()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dataframe['Month_Year'], y=dataframe[Target_column],
                        mode='lines+markers',
                        name='lines+markers'))

        fig1.update_layout(   
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
                                showlegend=False,)
        fig1.update_xaxes(tickangle=30)
        st.plotly_chart(fig1, use_container_width = True)
    except Exception as e:
        st.text(e)
        pass
    st.warning(" ") 

def year_eda(Date_column,Target_column,dataframe):
    st.markdown("<h3 style='text-align: center; color: gray;'>Yearly Chart of Target Column Over Date column </h3>", unsafe_allow_html=True)
    try:
        # st.dataframe(dataframe)
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column])
        dataframe['Month'] = dataframe[Date_column].dt.month
        dataframe['Year'] = dataframe[Date_column].dt.year
        dataframe = dataframe.groupby(['Year'],as_index=False)[Target_column].sum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dataframe['Year'], y=dataframe[Target_column],
                        mode='lines+markers',
                        name='lines+markers'))

        fig2.update_layout(   
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
                                showlegend=False,)
        fig2.update_xaxes(tickangle=30)
        st.plotly_chart(fig2, use_container_width = True)
    except Exception as e:
        st.text(e)



def trend_seasonality_decompose(Date_column,Target_column,dataframe):
    try:
        st.markdown("<h3 style='text-align: center; color: gray;'>trend_seasonality_decompose of Target Column</h3>", unsafe_allow_html=True)
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column])#.dt.date
        dataframe.index = dataframe[Date_column]
        dataframe = dataframe[Target_column]
        # st.dataframe(dataframe)
        dataframe.sort_index(inplace=True)
        decompose_result = seasonal_decompose(dataframe)
        trend_estimate    = decompose_result.trend
        periodic_estimate = decompose_result.seasonal
        plt.rcParams["figure.figsize"] = (10,7)
        st.pyplot(fig=decompose_result.plot())
    
    except Exception:
        pass