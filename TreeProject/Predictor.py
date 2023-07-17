from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model = keras.models.load_model('AQI_1_day')

prediction_days = 60
future_day = 1
#results = model.evaluate(test_features,test_label)
def predict():
    file_path="AQI_test_data.csv"
    data=pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["AQI"].values.reshape(-1,1))
    _test= np.array(scaled_data)
    print(_test)
    _test = _test.reshape(-1, 1)
    print(_test.shape)
    _test = np.reshape(_test, (_test.shape[1], _test.shape[0], 1))
    print(_test.shape)
    predicted_aqi=model.predict(_test)
    predicted_aqi = scaler.inverse_transform(predicted_aqi)
    print(predicted_aqi)
predict()