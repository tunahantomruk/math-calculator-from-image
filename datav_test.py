import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('datav.model')

# Load and preprocess the custom image
img = cv2.imread('slash4.png', cv2.IMREAD_GRAYSCALE)
original_img = img.copy()

# Resize the image to 28x28
img = cv2.resize(img, (28, 28))

# Invert the pixel values
img = np.invert(img)

# Reshape the image to match the input shape expected by the model
img = img.reshape(1, 28, 28, 1)

# Normalize the pixel values
img = img / 255.0

# Perform prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# Display the original image and preprocessed image
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

plt.show()

print("The predicted class is:", predicted_class)
