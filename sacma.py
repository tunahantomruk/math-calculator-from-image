import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values of the images to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
# model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test data
# loss, accuracy = model.evaluate(x_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# Save the model
# model.save('mnist_model.h5')
# print("Model saved.")

# Load the saved model
model = tf.keras.models.load_model('mnist_model.h5')
print("Model loaded.")

# Load and preprocess a custom number image
custom_image_path = '2.png'
custom_image = Image.open(custom_image_path).convert('L')
custom_image = custom_image.resize((28, 28))
custom_image_array = np.array(custom_image) / 255.0

# Reshape the image array
custom_image_array = custom_image_array.reshape((1, 28, 28))

# Make prediction on the custom number using the loaded model
prediction = model.predict(custom_image_array)
predicted_label = np.argmax(prediction)

print("Predicted Label:", predicted_label)
