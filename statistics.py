# Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This code provides the statistical results shown in chapter 4.1 Introductory investigations

It produces one correlation graph and 4 scatterplots investigating the correlations.
"""

"""
Importing and handling data.
"""

data = pd.read_csv('ampacity_dataset.csv')
df = pd.DataFrame(data)

#Series with the five datainputs. There is no nan values or 0 values in the dataset
time = df['time']
air_temp = df['air_temperature']
wind_speed = df['wind_speed']
wind_direction = df['wind_direction']
ampacity = df['ampacity']

"""
Make a correlation graph and plot each feature against the ampacity
"""

corr_df = df.drop(['time'], axis='columns')

# getting the correlationmatrix
correlation_matrix = corr_df.corr().round(1)

# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


# Plot the correlations with scatterplots

plt.figure()
plt.title('Air temperature vs Ampacity')
plt.xlabel('Air temperature [°C]')
plt.ylabel('Ampacity [A]')
plt.scatter(air_temp, ampacity, s=0.2)
plt.show()

plt.figure()
plt.title('Windspeed vs Ampacity')
plt.xlabel('Windspeed [m/s]')
plt.ylabel('Ampacity [A]')
plt.scatter(wind_speed, ampacity, s=0.2)
plt.show()

plt.figure()
plt.title('Wind direction vs Ampacity')
plt.xlabel('Wind direction [rad(north=0)]')
plt.ylabel('Ampacity [A]')
plt.scatter(wind_direction, ampacity, s=0.2)
plt.show()

plt.figure()
plt.title('Windspeed vs Wind direction')
plt.xlabel('Windspeed [m/s]')
plt.ylabel('Wind direction [rad(north=0)]')
plt.scatter(wind_speed, wind_direction, s=0.2)
plt.show()
