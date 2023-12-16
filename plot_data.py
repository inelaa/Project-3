import pandas as pd
import matplotlib.pyplot as plt

"""
This code provides visualization of the dataset
The ampacity is plotted as a function of air temperature, wind speed and direction
Levels of ampacity is indicated with the heatmap
"""

# Import data
data = pd.read_csv('ampacity_dataset.csv')
df = pd.DataFrame(data)


# Series with the five datainputs. There is no nan values or 0 values in the dataset
time = df['time']
air_temp = df['air_temperature']
wind_speed = df['wind_speed']
wind_direction = df['wind_direction']
ampacity = df['ampacity']


"""
Create designmatrix
"""

x = air_temp
y = wind_speed
z = wind_direction
c = ampacity

#Vizualize the full dataset with a heat scatterplot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, z, y, c=c, s=2, cmap='GnBu')
ax.set_xlabel('Air temperature')
ax.set_ylabel('Wind direction')
ax.set_zlabel('Wind speed')
ax.set_title('Ampacity as a function of air temperature, wind speed and direction')
fig.colorbar(img, orientation="horizontal", shrink=0.55, pad=0.15).set_label('Ampacity')
ax.view_init(azim=-50)
plt.show()