import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


"""
This code provides the illustrations found under chapter 4.3 facebook prophet benchmarks
"""

"""
Collecting dataset
"""


# The data set
data = pd.read_csv('ampacity_dataset.csv', header=0)
df = pd.DataFrame(data)

# Prophet package only wants to colums, datestamp and targetvalues
prophet_df = df.drop(["air_temperature", 'wind_speed', 'wind_direction'], axis='columns')
prophet_df.columns = ['ds', 'y']

# Scaling the targetvalues with Standardscaler
y_scaler = StandardScaler()
prophet_df['y'] = y_scaler.fit_transform(np.array(prophet_df['y']).reshape(-1,1))

# Remove timezone from ds column
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)  #removing timezone

"""
Now the data is in correct format for using fb prophet and the model will be set up and trained
before it is made to do predictions.

The facebookprophet starts to predict from the first datapoint, and you can specify how many 
datapoints further you want to predict. 

It is chosen to train the model on the whole dataset, on 16 months of dataset, 30 days of dataset
and 22 hours of dataset. 

With this the following training and predictions is made:
train: whole dataset --> predict 7 months (June 2023 - December 2023)
train: 16 months of dataset --> predict 1 month (May 2023)
train: 30 days of dataset --> predict 1 day (31st of January 2022)
train: 22 hours of dataset --> predict 2 hours (last two hours of Januray 1st 2022)
"""

# Making shorter df for each scenario
month_df = prophet_df.drop(df.tail(744).index) # predicting last month of data set (may 2023)
day_df = prophet_df.iloc[:720] # want to predict 1 day of january
hour_df = prophet_df.iloc[:23] # want to make df of 22 points, try to predict next 2 hours

# initiazing models with a 95% confidence interval
fb_whole = Prophet(interval_width=0.95)
fb_whole.fit(prophet_df) # training on the whole data set

fb_month = Prophet(interval_width=0.95)
fb_month.fit(month_df) # training on 16 months of dataset

fb_day = Prophet(interval_width=0.95)
fb_day.fit(day_df) # training on 30 days of data

fb_hour = Prophet(interval_width=0.95)
fb_hour.fit(hour_df) # training on 22 hours of data

# making a predictive df which contain a lot of statistics, including the predictive y-values
# the frequency is set to hourly, as the dataset have an hourly spacing
# Periods is how many datapoints into the future you want to predict
whole_future = fb_whole.make_future_dataframe(periods=5136, freq='H')
month_future = fb_month.make_future_dataframe(periods=744, freq='H')
day_future = fb_day.make_future_dataframe(periods=24, freq='H')
hour_future = fb_hour.make_future_dataframe(periods=2, freq='H')

# predict the future
whole_forecast = fb_whole.predict(whole_future) # predicting June-December 2023
month_forecast = fb_month.predict(month_future) # predicting a month
day_forecast = fb_day.predict(day_future) # predicting a day
hour_forecast = fb_hour.predict(hour_future) # predict two last hour of day

"""
Plotting and vizualising predictions
"""

# Visualize the predictions
fb_whole.plot(whole_forecast)
plt.xlabel('Date')
plt.ylabel('Ampacity [A]')
plt.title('Predicting June-December 2023')
plt.show()

fb_month.plot(month_forecast)
plt.xlabel('Date')
plt.ylabel('Ampacity [A]')
plt.title('Predicting May 2023')
plt.show()

fb_day.plot(day_forecast)
plt.xlabel('Date')
plt.ylabel('Ampacity [A]')
plt.title('Predicting last day of January 2022')
plt.show()

"""
Plotting trends
"""
fb_whole.plot_components(month_forecast)
plt.show()

fb_month.plot_components(month_forecast)
plt.show()

fb_day.plot_components(day_forecast)
plt.show()

"""
Check R2 and MSE score of facebook prophet models. yhat values are the facebook prophets
predictions of the target value.

Checking for month prediction, day prediction and 2 hour prediction.
"""

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# Monthly
R2_monthly = R2(prophet_df['y'].values[-744:], month_forecast['yhat'].values[-744:])
print(f'R2 for month prediction is: {R2_monthly}')
MSE_monthly = MSE(prophet_df['y'].values[-744:], month_forecast['yhat'].values[-744:])
print(f'MSE for month prediction is: {MSE_monthly}')

# daily scores
R2_daily = R2(prophet_df['y'].values[720:744], day_forecast['yhat'].values[-24:])
print(f'R2 for day prediction is: {R2_daily}')
MSE_daily = MSE(prophet_df['y'].values[720:744], day_forecast['yhat'].values[-24:])
print(f'MSE for day prediction is: {MSE_daily}')

# hourly scores
R2_hour = R2(prophet_df['y'][23:25].values, hour_forecast['yhat'].values[-2:])
print(f'R2 for hour prediction is: {R2_hour}')
MSE_hour = MSE(prophet_df['y'].values[23:25], hour_forecast['yhat'].values[-2:])
print(f'MSE for hour prediction is: {MSE_hour}')
