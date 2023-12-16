## Importing packages

# Different methods from Keras needed to create an RNN
# This is not necessary but it shortened function calls
# that need to be used in the code.

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

# For timing the code
from timeit import default_timer as timer

# For plotting
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

"""
This code is written by Morten Hjorth-Jensen code and adjusted by Ine Lovise Aarnes to produce 
wanted results.

This code provides for the results under chapter 4.2 Results from RNN trials

It is an experimental code, and provides the base for the different predictions in results.
Different predictions is done by changing the nubmer of datapoints extracted from the scaled 
design matrix and changing the dim variable. The dim variable decides from where in your e
xtracted data you want to start predicting. 

The code is based on three functions: format_data, lstm and test_lstm.
"""

"""
Importing and preparing dataset 
"""

# Set the seed to 777 for reproducable results
random.seed(777)

# The data set
datatype='DLR'
data = pd.read_csv('ampacity_dataset.csv')
df = pd.DataFrame(data)

#Series with the five datainputs. There is no nan values or 0 values in the dataset
time = df['time']
air_temp = df['air_temperature']
wind_speed = df['wind_speed']
wind_direction = df['wind_direction']
ampacity = df['ampacity']

# Creating designmatrix and target
X = np.zeros((df.shape[0], df.shape[1] - 2)) #timestamp and ampacity is not a part of the design matrix
X[:,0] = air_temp
X[:,1] = wind_speed
X[:,2] = wind_direction

y = ampacity.tolist()

# Scaling X and y
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(np.array(y).reshape(-1,1))
y_scaled_list = y_scaled.reshape(len(y)).tolist()

"""
Formatting the Data

Data gets prepared to be used in training the RNN
"""

# FORMAT_DATA
def format_data(data, length_of_sequence = 2):
    """
        Inputs:
            data(a numpy array): the data that will be the inputs to the recurrent neural
                network
            length_of_sequence (an int): the number of elements in one iteration of the
                sequence patter.  For a function approximator use length_of_sequence = 2.
        Returns:
            rnn_input (a 3D numpy array): the input data for the recurrent neural network.  Its
                dimensions are length of data - length of sequence, length of sequence,
                dimnsion of data
            rnn_output (a numpy array): the training data for the neural network
        Formats data to be used in a recurrent neural network.
    """

    X, Y = [], []
    for i in range(len(data)-length_of_sequence):
        # Get the next length_of_sequence elements
        a = data[i:i+length_of_sequence]
        # Get the element that immediately follows that
        b = data[i+length_of_sequence]
        # Reshape so that each data point is contained in its own array
        a = np.reshape (a, (len(a), 1))
        X.append(a)
        Y.append(b)
    rnn_input = np.array(X)
    rnn_output = np.array(Y)

    return rnn_input, rnn_output



"""
Creating a function lstm that creates a keras model with wanted input size.

This function makes it easy to change types of layers and other parameters.
"""

def lstm(length_of_sequences, batch_size = None, stateful = False):
    """
        Inputs:
            length_of_sequences (an int): the number of y values in "x data".  This is determined
                when the data is formatted
            batch_size (an int): Default value is None.  See Keras documentation of SimpleRNN.
            stateful (a boolean): Default value is False.  See Keras documentation of SimpleRNN.
        Returns:
            model (a Keras model): The recurrent neural network that is built and compiled by this
                method
        Builds and compiles a recurrent neural network with two dense layers and a LSTM hidden
         layer and returns the model.
    """
    # Number of neurons on the input/output layer and the number of neurons in the
    # dense and hidden layer
    in_out_neurons = 1
    dens1_neurons = 26
    hidden_neurons = 20

    # Input Layer
    inp = Input(batch_shape=(batch_size,
                length_of_sequences,
                in_out_neurons))

    # feed forward network before LSTM
    dens1 = Dense(dens1_neurons, name="dens1", kernel_regularizer='l1_l2')(inp)

    # Hidden layers
    rnn= LSTM(hidden_neurons,
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN", use_bias=True, activation='tanh')(dens1)

    # Output layer
    dens2 = Dense(in_out_neurons,name="dens2", kernel_regularizer='l1_l2')(rnn)

    # Define the model
    model = Model(inputs=[inp],outputs=[dens2])
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam') # defininf loss and optimizer
    # Return the model
    return model

"""
Predicting New Points With A Trained Recurrent Neural Network

This performs predictions. The function visualize the predictions and calculates the MSE and R2
scores. 
"""

def test_lstm(x1, y_test):
    """
        Inputs:
            x1 (a list or numpy array): The complete x component of the data set
            y_test (a list or numpy array): The complete y component of the data set

        Returns:
            None.

        Uses a trained recurrent neural network model to predict future points in the
        series.  Computes the MSE of the predicted data set from the true data set, saves
        the predicted data set to a csv file, and plots the predicted and true data sets.
        MSE and R2 scores get calculated.
    """
    # Add the training data as the first dim points in the predicted data array as these
    # are known values.
    y_pred = y_test[:dim]

    # Generate the first input to the trained recurrent neural network using the last two
    # points of the training data.  Based on how the network was trained this means that it
    # will predict the first point in the data set after the training data.  All of the
    # brackets are necessary for Tensorflow.

    next_input = np.array([[ [y_test[dim-10]], [y_test[dim-9]],
        [y_test[dim-8]], [y_test[dim-7]], [y_test[dim-6]], [y_test[dim-5]],
                            [y_test[dim-4]], [y_test[dim-3]], [y_test[dim-2]], [y_test[dim-1]]]])
    # Save the very last point in the training data set.  This will be used later.
    last = [y_test[dim-1]]

    # Iterate until the complete data set is created.
    for i in range (dim, len(y_test)):
        # Predict the next point in the data set using the previous two points.
        next = model.predict(next_input)
        # Append just the number of the predicted data set
        y_pred.append(next[0][0])
        # Create the input that will be used to predict the next data point in the data set.
        next_input = np.array([[last, next[0]]], dtype=np.float64)
        last = next

    # Print the mean squared error between the known data set and the predicted data set.
    print('MSE: ', np.square(np.subtract(y_test, y_pred)).mean())

    def R2(y_data, y_model):
        RSS = np.sum([(x - y)**2 for x, y in zip(y_data, y_model)])
        mean = np.mean(y_data)
        SSR = np.sum([(x - mean)**2 for x in y_data])
        return 1 - RSS/SSR

    R2 = R2(y_test, y_pred)
    print(f'R2 score: {R2}')

    # Invert back the scaled elements
    x_inverted = X_scaler.inverse_transform(x1)

    air_temp = x_inverted[:,0]
    wind_speed = x_inverted[:, 1]
    wind_direction = x_inverted[:, 2]

    y_test_inverted = y_scaler.inverse_transform(np.array(y_test).reshape(len(y_test), 1))
    y_pred_inverted = y_scaler.inverse_transform(np.array(y_pred).reshape(len(y_pred), 1))

    # Save the predicted data set as a csv file for later use
    name = datatype + 'Predicted'+str(dim)+'.csv'
    np.savetxt(name, y_pred_inverted, delimiter=',')

    # finding differences in the predicted ampacity
    amp_diff = y_test_inverted-y_pred_inverted

    # Plot the known data set and the predicted data set.

    # plotting the true values as scatterplot with ampacity as heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img1 = ax.scatter(air_temp, wind_direction, wind_speed, c=y_test_inverted, s=2, cmap='GnBu')
    ax.set_xlabel('Air temperature')
    ax.set_ylabel('Wind direction')
    ax.set_zlabel('Wind speed')
    ax.set_title('Calculated ampacity')
    ax.view_init(azim=-45)
    fig.colorbar(img1, orientation="horizontal", shrink=0.55, pad=0.1).set_label('Ampacity')
    plt.show()

    # plotting the predicted values as scatterplot with ampacity as heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img2 = ax.scatter(air_temp, wind_direction, wind_speed, c=y_pred_inverted, s=2, cmap='GnBu')
    ax.set_xlabel('Air temperature')
    ax.set_ylabel('Wind direction')
    ax.set_zlabel('Wind speed')
    ax.set_title('Predicted ampacity')
    fig.colorbar(img2, orientation="horizontal", shrink=0.55, pad=0.1).set_label('Ampacity')
    ax.view_init(azim=-45)
    plt.show()

    # plotting the difference between true and predicted ampacity heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img2 = ax.scatter(air_temp, wind_direction, wind_speed, c=amp_diff, s=2, cmap='GnBu')
    ax.set_xlabel('Air temperature')
    ax.set_ylabel('Wind direction')
    ax.set_zlabel('Wind speed')
    ax.set_title('Ampacity difference for predictions and true data')
    fig.colorbar(img2, orientation="horizontal", shrink=0.55, pad=0.1).set_label('Ampacity')
    ax.view_init(azim=-45)
    plt.show()

    # Make a plot of true and predictive values for each datapoint to track how well predictive
    # values match true values

    fig, ax = plt.subplots()
    ax.set_title(f'Ampacity limits for {len(x1[:,0])} number of datapoints')
    ax.set_xlabel('Datapoint number')
    ax.set_ylabel('Ampacity [A]')
    ax.plot(range(len(x1[:,0]))[::20], y_test_inverted[::20], label="true", linewidth=1)
    ax.plot(range(len(x1[:,0]))[::20], y_pred_inverted[::20], 'g-.',label="predicted", linewidth=1)
    ax.legend()
    plt.show()

# Check to make sure the data set is complete
assert len(X[:,0]) and len(X[:,1]) and len(X[:,2])  == len(y)


"""
Performing lstm
"""
# This is the number of points that will be used in as the training data
dim=720

# Generate the training data for the RNN
rnn_input, rnn_training = format_data(y_scaled_list[:744], 10)

# Create a recurrent neural network in Keras and produce a summary of the
# machine learning model
model = lstm(length_of_sequences = rnn_input.shape[1], batch_size=120)
model.summary()

# Start the timer.  Want to time training+testing
start = timer()
# Fit the model using the training data genenerated above using 150 training iterations and a 5%
# validation split.  Setting verbose to True prints information about each training iteration.
hist = model.fit(rnn_input, rnn_training, batch_size=120, epochs=100,
                 verbose=True,validation_split=0.05)

#Plot the training loss against the validation loss
for label in ["loss","val_loss"]:
    plt.plot(hist.history[label],label=label)

plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
plt.legend()
plt.show()

# Use the trained neural network to predict more points of the data set
test_lstm(X_scaled[:744, :], y_scaled_list[:744])
# Stop the timer and calculate the total time needed.
end = timer()
print('Time: ', end-start)


# Getting information of the last udated weights and biases for the predictions
adjusted_weights = model.get_layer('dens2').get_weights()[0] # Get the adjusted weights of the dense layer
adjusted_biases = model.get_layer('dens2').get_weights()[1] # Get the adjusted biases of the dense layer
print(adjusted_weights, adjusted_biases) # Print the adjusted weights and biases