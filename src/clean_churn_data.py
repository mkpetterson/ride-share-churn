import numpy as np
import pandas as pd
from datetime import datetime

def clean_churn_data(data):
    
    churn = data.copy()
    
    # fix nan in numeric features
    avg_avg_rating_of_driver = np.mean(churn['avg_rating_of_driver'])
    avg_avg_rating_by_driver = np.mean(churn['avg_rating_by_driver'])
    churn['avg_rating_by_driver'] = churn['avg_rating_by_driver'].apply(
        lambda x: avg_avg_rating_by_driver if np.isnan(x) else x)
    churn['avg_rating_of_driver'] = churn['avg_rating_of_driver'].apply(
        lambda x: avg_avg_rating_of_driver if np.isnan(x) else x)
    
    # remove remaining nan
    churn = churn.dropna()
    
    # convert to datetime
    churn['signup_date'] = pd.to_datetime(churn['signup_date'], format='%Y-%m-%d')
    churn['last_trip_date'] = pd.to_datetime(churn['last_trip_date'], format='%Y-%m-%d')
    
    # convert boolean to int
    churn['luxury_car_user'] = churn['luxury_car_user'].astype(int)
    
    #encode phone to boolean (1 for iphone, 0 for android)
    churn['phone'] = churn['phone'].apply(
        lambda x: 1 if x == 'iPhone' else 0)
    
    #one hot encode city
    churn = pd.get_dummies(churn, columns=['city'], prefix='city', prefix_sep=': ', dtype='int')
    
    # add churn column
    churn_date = datetime.strptime('2014-06-01', '%Y-%m-%d')
    churn['churn?'] = (churn['last_trip_date'] < churn_date).astype(int)
    
    # drop date columns
    churn = churn.drop(columns=['signup_date','last_trip_date'])
    
    return churn