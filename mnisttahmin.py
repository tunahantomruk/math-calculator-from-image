import cv2
import pytesseract
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
def perform_ocr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    grayscale_img = get_grayscale(image)
    thresholded_img = thresholding(grayscale_img)
    denoised_img = remove_noise(thresholded_img)

    # Perform OCR
    text = ocr_core(denoised_img)
    return text



def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

def main():
    # Load the MNIST dataset and split it into training and testing sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Print the shape of the training set
    print("Training set:")
    print("Number of samples:", x_train.shape[0])
    print("Image shape:", x_train.shape[1:])

    # Print the shape of the test set
    print("\nTest set:")
    print("Number of samples:", x_test.shape[0])
    print("Image shape:", x_test.shape[1:])

    # Print the number of unique labels/classes
    num_classes = len(set(y_train))
    print("\nNumber of classes:", num_classes)
    # Normalize the pixel values of the images to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model if not already trained
    if not tf.keras.backend.learning_phase():
        # Check if saved weights exist
        try:
            model.load_weights('model_weights.h5')
        except OSError:
            # Train the model on the training data
            model.fit(x_train, y_train, epochs=5)
            # Save the model weights
            model.save_weights('model_weights.h5')

    # Evaluate the model on the test data
    model.evaluate(x_test, y_test, verbose=2)

    # Load and preprocess the custom number image
    custom_image_path = 'h3.png'
    custom_image = Image.open(custom_image_path).convert('L')
    custom_image = custom_image.resize((28, 28))
    custom_image_array = np.array(custom_image) / 255.0

    # Invert the colors of the image array
    custom_image_array = 1 - custom_image_array

    # Reshape the image array
    custom_image_array = custom_image_array.reshape((1, 28, 28))

    # Make prediction on the custom number using the MNIST model
    prediction = model.predict(custom_image_array)
    predicted_label = np.argmax(prediction)

    print("Predicted Label:", predicted_label)

    # Perform OCR on the custom number image
    ocr_result = perform_ocr(custom_image_path)
    print("OCR Result:", ocr_result)



main()
