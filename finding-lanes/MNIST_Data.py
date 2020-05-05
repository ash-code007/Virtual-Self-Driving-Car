import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 # 60000 train, 10000 test images

print(X_train.shape)
# (60000,28,28)
print(X_test.shape)
# (10000,28,28)
print(y_train.shape[0])
# 60000
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"

num_of_samples = []

cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 8))
fig.tight_layout()
# to prevent overlapping of text
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        # only jth digit will be stored
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        # selecting a random image of jth digit from 0 to length-1, 28x28 pixels(:,:), and setting grayscale
        axs[j][i].axis("off")
        # turning off the axes labels
        if i == 2:
            axs[j][i].set_title(str(j))
            # displaying title( string representation of the jth digit) on top of 3rd column (index 2) of each row
            num_of_samples.append(len(x_selected))
            # stores 10 count values, one for each digit

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

# one hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# normalization
X_train = X_train/255
X_test = X_test/255

# reshaping 28x28 into a single array of 784 pixels
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

# defining the NN model
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model()
print(model.summary())

history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)
# 10% of training data is validation set
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')

score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
# (score, accuracy)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# model is now trained and tested, notw testing it on an image from the web
import requests
from PIL import Image
# Python Imaging Library

url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

import cv2
img = np.asarray(img)
# converts img to a np array
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
# black background and white digits, coz thats what the model was trained on, and this is ulta
plt.imshow(img, cmap=plt.get_cmap('gray'))

img = img/255
img = img.reshape(1, 784)

prediction = model.predict_classes(img)
print("predicted digit:", str(prediction))
