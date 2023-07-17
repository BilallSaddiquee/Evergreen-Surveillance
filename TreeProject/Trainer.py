import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Layer,Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from keras import backend as K

#Start and end date for test data
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

#uses DataReader from pandas module 
data=pd.read_csv('data/AQI.csv')
print(data)

#Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["AQI"].values.reshape(-1,1))
print(scaled_data)

#sets the number of days to be predicted and test data
#looks at past [prediction_days] to predict
prediction_days = 60
future_day = 1

#creates x_train and y_train
x_train, y_train = [], []


for x in range(prediction_days, len(scaled_data)-future_day):
    #past [prediction_days] and appends the real data to x_train
    x_train.append(scaled_data[x-prediction_days:x, 0])
    #the [future_day] is appended as the trained predected data to y_train
    y_train.append(scaled_data[x+future_day, 0])
    
#turns x and y train into numpy arrays and then reshapes them
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape[0],',',x_train.shape[1])
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="normal")
        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:

            return output
        return K.sum(output, axis=1)

#Create neural network

model = Sequential()

#LSTM layers for feeding data to neural network. Dropout layers to prevent overfitting
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(attention(return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#compiling and fitting the model 
model.compile(optimizer="adam", loss="mean_squared_error",metrics="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)
model.save('AQI_1_day')


#Testing the model
test_data = pd.read_csv('data/AQI_test_data.csv')
actual_aqi = test_data["AQI"].values

#combine the test data and actual data
total_dataset = pd.concat((data["AQI"], test_data["AQI"]), axis=0)

#places the actual data - test and prediction data into the model input 
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
print(model_inputs[0])
print(model_inputs.shape)
model_inputs = model_inputs.reshape(-1, 1)
print(model_inputs.shape)
model_inputs = scaler.fit_transform(model_inputs)
print(model_inputs.shape)
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Predicts the prices using the model
predicted_aqi = model.predict(x_test)
predicted_aqi = scaler.inverse_transform(predicted_aqi)

#Plots them on a graph
plt.plot(actual_aqi, color="red", label="Actual AQI")
plt.plot(predicted_aqi, color="green", label="Predicited AQI")
plt.title("AQI Prediction")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.legend()
plt.show()
