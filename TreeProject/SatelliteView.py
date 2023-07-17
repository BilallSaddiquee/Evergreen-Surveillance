#NASA API KEY  kNeAfyTLIDIl7hPnWeaMIxsCsGYsnxkXYQ4yzVUb
#GOOGLE API KEY AIzaSyD-dID-nQW8FtF5l45oX9hBgHvdQi0jrWM
import numpy as np
import urllib.request as ur
import cv2
from PIL import Image

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = ur.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

image=url_to_image('https://maps.googleapis.com/maps/api/staticmap?center=Forman%20Christian%20College,Lahore&zoom=14&size=400x400&key=AIzaSyD-dID-nQW8FtF5l45oX9hBgHvdQi0jrWM')
cv2.imwrite('FC_Satellite_View.png',image)
image=cv2.imread('Gaddafi.png')
number_of_green_pix = np.sum(image[:,:,1] == 218)
number_of_green_pix2 = np.sum(image[:,:,1] == 218, where=((image[:,:,0] == 181) & (image[:,:,2] == 168)))
green_percent=number_of_green_pix/image.size
print(number_of_green_pix ,number_of_green_pix2, green_percent, image.size)
img=Image.open('Gaddafi.png')
width,height=img.size
print("Size of Image ", width*height)