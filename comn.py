import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def segment_characters(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert to binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform morphological operations to improve character connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    erode = cv2.erode(dilate, kernel, iterations=1)

    # Find contours of connected components
    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours based on area and aspect ratio
    min_area = 120
    max_area = 3000
    min_aspect_ratio = 0.2
    max_aspect_ratio = 1.5
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h if h != 0 else 0
        if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            filtered_contours.append(contour)

    # Sort contours from left to right
    filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

    # Extract and save each character
    output_images = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        character_img = erode[y:y + h, x:x + w]
        output_images.append(character_img)

    return output_images

# Load the trained model     'yutub.model'  -->  'in.model'
model = tf.keras.models.load_model('yutub.model')

# Load custom image and segment characters
image_path = '4+3.png'
characters = segment_characters(image_path)

# Show each segmented character
for i, character in enumerate(characters):
    # Preprocess the character image
    character = cv2.resize(character, (28, 28))
    character = np.invert(character)
    character = character.reshape(1, 28, 28)

    # Perform prediction
    prediction = model.predict(character)
    predicted_class = np.argmax(prediction)

    # Display the character
    cv2.imshow("Character {}".format(i), character)
    cv2.waitKey(0)

    print("Character {}: Predicted class is {}".format(i, predicted_class))

# Close all windows
cv2.destroyAllWindows()
