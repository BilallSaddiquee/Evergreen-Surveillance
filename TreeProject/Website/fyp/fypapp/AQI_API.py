import requests
import json
from datetime import datetime as dt
import time
#For Prediction View
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import numpy as np
import pandas as pd
#For Satellite View
import urllib.request as ur
import cv2
from PIL import Image

model = keras.models.load_model('AQI_1_day')
model_15 = keras.models.load_model('AQI_15_day')

prediction_days = 60
future_day = 1
api_key='33589068750ce8d9c394d0b845e451fa'
#AQI DATA GET API
def GET_API_DATA():
    start_date='1606223802'
    end_date='1606482999'
    end_date=int(time.time())
    sub_date=3*24*60*60
    start_date=end_date-sub_date
    #print('start date',start_date)
    #print('end date',end_date)
    DATA=[]
    data=requests.get('http://api.openweathermap.org/data/2.5/air_pollution/history?lat=31.558&lon=74.35071&start='+str(start_date)+'&end='+str(end_date)+'&appid='+api_key)
    #print(data.content)
    json_data=json.loads(data.content)
    #print(json_data)
    aqi_list=json_data['list']
    #print(len(aqi_list[-60:]))
    f = open("fypapp/api_data.csv", "w")
    f.write('Date (LT),AQI\n')
    for i in aqi_list[-60:]:
        date=dt.fromtimestamp(i['dt']).strftime('%#m/%#d/%Y %#H:%M')
        str_data_line=str(date)+','+str(i['main']['aqi'])+'\n'
        DATA.append(str_data_line)
        f.write(str_data_line)
        #print(date,i['main']['aqi'])
    f.close()




#results = model.evaluate(test_features,test_label)
def predict(DAY):
    GET_API_DATA()
    file_path="api_data.csv"
    data=pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["AQI"].values.reshape(-1,1))
    _test= np.array(scaled_data)
    print(_test)
    _test = _test.reshape(-1, 1)
    print(_test.shape)
    _test = np.reshape(_test, (_test.shape[1], _test.shape[0], 1))
    print(_test.shape)
    if DAY==1:
        predicted_aqi=model.predict(_test)
    if DAY==15:
        predicted_aqi=model_15.predict(_test)
    predicted_aqi = scaler.inverse_transform(predicted_aqi)
    print(predicted_aqi)
    return int(predicted_aqi)


#Satellite View API

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = ur.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def Satellite_View(location):
    location=location.replace(' ','%20')
    image=url_to_image('https://maps.googleapis.com/maps/api/staticmap?center='+location+'&zoom=18&size=400x400&key=AIzaSyD-dID-nQW8FtF5l45oX9hBgHvdQi0jrWM')
    sat_image=url_to_image('https://maps.googleapis.com/maps/api/staticmap?center='+location+'&zoom=18&size=400x400&maptype=satellite&key=AIzaSyD-dID-nQW8FtF5l45oX9hBgHvdQi0jrWM')
    cv2.imwrite('Satellite_View.png',sat_image)
    cv2.imwrite('Map_View.png',image)
    number_of_green_pix = np.sum(image[:,:,1] == 218)
    number_of_green_pix2 = np.sum(image[:,:,1] == 218, where=((image[:,:,0] == 181) & (image[:,:,2] == 168)))
    green_percent=number_of_green_pix/image.size
    print(number_of_green_pix ,number_of_green_pix2, green_percent, image.size)
