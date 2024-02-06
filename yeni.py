import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('in.model')

# Function to preprocess a single image
def preprocess(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    grayscale_img = get_grayscale(image)
    thresholded_img = thresholding(grayscale_img)
    denoised_img = remove_noise(thresholded_img)

    # Resize the denoised image to (28, 28)
    resized_img = cv2.resize(denoised_img, (28, 28))

    # Reshape the image for CNN input
    image = resized_img.reshape(1, 28, 28, 1)

    # Normalize the image
    image = image.astype('float32') / 255.0

    return image


# Function to convert the image to grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to remove noise from the image
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# Function to apply thresholding to the image
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Function to predict the label of a single image
def predict_image(image_path):
    # Preprocess the image
    preprocessed_image = preprocess(image_path)

    # Show the preprocessed image
    cv2.imshow("Preprocessed Image", preprocessed_image.reshape(28, 28))
    cv2.waitKey(0)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Get the predicted label
    predicted_label = np.argmax(predictions[0])
    return predicted_label

# Load and preprocess the custom test image
image_path = "222.png"
predicted_label = predict_image(image_path)
print("Predicted label:", predicted_label)
