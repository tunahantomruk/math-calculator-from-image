import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

# mnist = tf.keras.datasets.mnist
# #image    class                     returns 2 tuples as train and test
# (x_train,y_train),(x_test,y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test =  tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model on the training data
# model.fit(x_train, y_train, epochs=3)

# model.save('yutub.model')

model = tf.keras.models.load_model('yutub.model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

# Load custom images and predict them
# image_number = 1
# while os.path.isfile('digits/digit{}.png'.format(image_number)):
#     try:
#         img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print("The number is probably a {}".format(np.argmax(prediction)))
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#         image_number += 1
#     except:
#         print("Error reading image! Proceeding with next image...")
#         image_number += 1

img = cv2.imread('6d.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = np.invert(img)
img = img.reshape(1, 28, 28)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print("The predicted class is:", predicted_class)
