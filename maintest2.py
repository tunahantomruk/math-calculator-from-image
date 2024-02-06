import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model('in.model')

# Function to preprocess a single image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Reshape the image for CNN input
    image = image.reshape(1, 28, 28, 1)

    # Normalize the image
    image = image.astype('float32') / 255.0
    
    return image

# Function to predict the label of a single image
def predict_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Show the preprocessed image
    cv2.imshow("Preprocessed Image", preprocessed_image.reshape(28, 28))
    cv2.waitKey(0)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Get the predicted label
    predicted_label = np.argmax(predictions[0])
    return predicted_label

# Load and preprocess the custom test image
image_path = "4wb.png"
image = cv2.imread(image_path)
preprocessed_image = preprocess_image(image)

# Make predictions on the custom test image
predicted_label = predict_image(image)
print("Predicted label:", predicted_label)

