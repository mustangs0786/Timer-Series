##################     IMPORTED LIBRARIES ################################
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from eda import *
from Modelling import *


################## Date and Target Column Selector ######################
def column_selector(dataframe):
    columns_list = list(dataframe.columns)
    columns_list.insert(0,None)
    Date_column = st.sidebar.selectbox("Select Date Column",columns_list)
    Target_column = st.sidebar.selectbox("Select Target Column",columns_list)
    return Date_column, Target_column

def type_of_date():
    data_type = st.sidebar.selectbox("Select Data Type",[None,'Daily','Weekly','Monthly','Yearly'])
    return data_type

def min_max_date_selecter(dataframe,Date_column):
    try:
        dataframe[Date_column] = pd.to_datetime(dataframe[Date_column]).dt.date
        Min_date = st.sidebar.date_input('Enter Starting Date',dataframe[Date_column].min(),min_value=dataframe[Date_column].min(), max_value=dataframe[Date_column].max())
        Max_date = st.sidebar.date_input('Enter End Date',dataframe[Date_column].max(),min_value=dataframe[Date_column].min(), max_value=dataframe[Date_column].max())
        return Min_date, Max_date
    except Exception as e :
        st.text(e)
        st.stop()


##################      MAIN CODE ########################################
if __name__ == "__main__":
    st.error("  ")
    st.markdown("<h1 style='text-align: center; color: black;'>Time Series Analysis</h1>", unsafe_allow_html=True)
    st.error("  ")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)        
        Date_column, Target_column = column_selector(dataframe)
        if Date_column is not None and Target_column is not None:
            data_type = type_of_date() 
            Min_date, Max_date = min_max_date_selecter(dataframe,Date_column)
            dataframe = dataframe[(dataframe[Date_column]>=Min_date) & (dataframe[Date_column]<=Max_date)]
            if data_type is not None :
                agree = st.sidebar.checkbox('Any Filtering')
                if agree:
                    filter_col = st.sidebar.selectbox(
                                'Select Column',
                                (dataframe.columns))
                    filter_value = st.sidebar.selectbox(
                                'Select Column',
                                (dataframe[filter_col].unique()))
                    dataframe = dataframe[dataframe[filter_col]==filter_value]

                option = st.sidebar.selectbox(
                            'EDA or Modelling',
                            ('None','Generate EDA Report', 'Ready for Modelling'))
                st.sidebar.error(" ")
                if option == 'Generate EDA Report':
                    eda(Date_column,Target_column,dataframe)
                    month_year_eda(Date_column,Target_column,dataframe)
                    year_eda(Date_column,Target_column,dataframe)
                    trend_seasonality_decompose(Date_column,Target_column,dataframe)
                elif option == 'Ready for Modelling':
                    train_df,test_df = train_test_split(dataframe,Date_column)
                    if st.button('Model_Training'):
                        train_test_eda(train_df,test_df,Target_column,Date_column)
                        forecast,train_df,test_df = fbprophet_model(train_df,test_df,Target_column,Date_column)
                        forecast,train_df,test_df = auto_arima_model(forecast,train_df,test_df,Target_column,Date_column)
                        rf_modeling(forecast,train_df,test_df,Target_column,Date_column)
                    # st.text('model')
            else:
                st.dataframe(dataframe.head(15))
            
    

            
            



