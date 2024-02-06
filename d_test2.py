import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

model = tf.keras.models.load_model('dset.model')

img = cv2.imread('222.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = np.invert(img)
img = img.reshape(1, 28, 28)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print("The predicted class is:", predicted_class)