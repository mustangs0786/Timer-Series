##################     IMPORTED LIBRARIES ################################
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from eda import *



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
            if data_type is not None :
                if st.sidebar.button('Generate EDA Report'):
                    eda(Date_column,Target_column,dataframe)
                    month_year_eda(Date_column,Target_column,dataframe)
                    year_eda(Date_column,Target_column,dataframe)
                    trend_seasonality_decompose(Date_column,Target_column,dataframe)
                if st.sidebar.button('Generate Modelling Report'):
                    st.text('model')
            else:
                st.dataframe(dataframe.head(15))
            
    

            
            



