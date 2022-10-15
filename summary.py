from datetime import date
import datetime
from nsepy import get_history
import pandas as pd,numpy as np
from matplotlib import pyplot as plt
from dateutil.parser import parse
import streamlit as st

# Function to get data in the form of data frame for a single stock for particular no of days
def get_backdata(stock_symbol,back_days):
    back_days = back_days
    todays_date = date.today()  # This will be the date till which the data is to be fetched
    weekday = datetime.datetime.weekday(todays_date) # Find out the day 
    start_date = datetime.timedelta(days=-back_days)+todays_date # Date of back_days back
    stock_data = get_history(symbol=stock_symbol,start=start_date,end=todays_date)
    return stock_data
#
# Function to create list of columns based on the number of days for which data is required
#
def get_columns_list(back_days):
    columns_list = ["SYMBOL",]
    prefixes = ["P","PC",'%PC',"TV","CTV","%CTV","TQ","CTQ","%CTQ","DV","CDV","%CDV","%DV","C%DV"]
    # P is for close prices 
    # PC is change in price from previus day to next day
    # %PC is percentage change in price from previous day to next day
    # TV is Traded Value
    # TQ is Traded Qty
    for prefix in prefixes:
        for day in reversed(range(back_days)):
            column_name = (prefix + str(day+1))
            if "C" in column_name:   # Check if the column is for change then skip one column as change days are one less than total days
                if((day+1)==back_days):
                    continue
            columns_list.append(column_name)
    return columns_list
#
# Function to get data from a symbol and convert it into a row for the final dataframe according to the summary columns
#
def get_row_data(new_stocks_data):
    row_data = []
    row_data.append(new_stocks_data.iloc[0][0])
    for cols in range(new_stocks_data.shape[1]):
        if (cols != 0): 
            for i in reversed(range(new_stocks_data.shape[0])):
                row_data.append(new_stocks_data.iloc[i][cols])
            for i in reversed(range(new_stocks_data.shape[0])):  
                if(i != 0):
                    row_data.append(round((new_stocks_data.iloc[i][cols]-new_stocks_data.iloc[i-1][cols]),2))
            if(cols != (new_stocks_data.shape[1]-1)):
                for i in reversed(range(new_stocks_data.shape[0])):  
                    if(i != 0):
                        row_data.append(round((((new_stocks_data.iloc[i][cols]-new_stocks_data.iloc[i-1][cols])/new_stocks_data.iloc[i][cols])*100),2))
    return row_data
                        
# "P","PC",'%PC',"TV","CTV","%CTV","TQ","CTQ","%CTQ","DV","CDV","%CDV","%DV","C%DV 
# These are the columns that we want in our final dataframe
# Now we will fetch the data and clean the columns by dropping unwantend columns and renaming the rest
def get_new_stocks_data(stock_name,back_days):
    stocks_data = get_backdata(stock_name,back_days)
    new_stocks_data=stocks_data
    new_stocks_data.drop(["Series","Prev Close","Open","High","Low","Last","VWAP","Turnover"],axis=1, inplace= True)
    rename_dict = {"Symbol":"SYMBOL","Close":"P","Volume":"TV","Trades":"TQ","Deliverable Volume":"DV","%Deliverble":"%DV",}
    new_stocks_data.rename(columns = rename_dict, inplace=True)
    return new_stocks_data    
#
#
# This is the main function that will be used to create the summary dataframe and it calls all the functions above.
#
#
def return_summary_df(index,back_days=3):   
#     back_days = 3  # Days for which the data is required. Example for comparing last 3 days put it as 3.
    df = pd.read_csv(index)
    print(index)
    stocks_list=df['Symbol'].to_list()
    print(stocks_list,len(stocks_list))
    new_stocks_data = get_new_stocks_data(stocks_list[1],back_days)
    # print(new_stocks_data)
    columns_list = get_columns_list(len(new_stocks_data))
    # print("List for columns ",columns_list) ## This is for debugging
    data = []
    # stocks_df = pd.DataFrame(data,columns = columns_list)
    
    for i in range(len(stocks_list)):
        if (i == 0):
            continue
        data.append(i)
    # print(data)
    
    
    stocks_df = pd.DataFrame(columns = columns_list,index=data) #debugging 
    
    #
    # This function will loop through the stocks list and collect all data in a single dataframe for the given list of stocks
    #
    for i in range(len(stocks_list)):
        if (i == 0):
            continue
        try:
            new_stocks_data = get_new_stocks_data(stocks_list[i],back_days)
            stocks_df.loc[i] = get_row_data(new_stocks_data)
        except:
            pass
        # print(get_row_data(new_stocks_data))  ##### This is for debugging
        # print(i)
        
    return stocks_df



st.set_page_config(layout="wide")
st.title('Stock Summary Price Action')

if(st.button("Run Algo")):
    summary_df = return_summary_df("index_fno.csv",7)
    st.write(summary_df)
else:
    summary_df = pd.read_csv("summary_result.csv")
    st.write(summary_df)
