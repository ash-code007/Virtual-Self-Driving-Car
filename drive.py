import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
sio = socketio.Server()
# to setup bi-directional communication with the simulator 
 
app = Flask(__name__) #'__main__'

speed_limit = 10
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
 
 
 
@sio.on('connect')
# on connection to server, the connect() method will be invoked which will print 'Connected' on the terminal
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
    # initial steering angle and throttle of zero 
    
 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
    # event name 'steer' is emitted which is an simulator defined event
    # it takes in two input arguements : steering_angle and throttle(both converted to string)
 
 
if __name__ == '__main__':
    model = load_model('model_andersy005GIT2.h5')
    app = socketio.Middleware(sio, app)
    # middleware to carry traffic between Flask App and SocketIO Server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    # to open listening events that listen on any available IP '' and port 4567