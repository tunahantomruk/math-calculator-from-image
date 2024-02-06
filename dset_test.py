import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

# Set the path to the main dataset folder
main_folder = 'dataset\\dataset'

# Load the images and labels
images = []
labels = []

# Recursive function to read images from nested folders
def read_images_from_folder(folder_path, label):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            try:
                image = cv2.imread(file_path, 0)  # Read the image in grayscale
                if image is None:
                    print(f"Error reading image: {file_path}")
                    continue
                image = cv2.resize(image, (28, 28))  # Resize the image to a consistent size
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image: {file_path}")
                print(e)
        elif os.path.isdir(file_path):
            read_images_from_folder(file_path, label)

# Iterate through each subfolder in the main dataset folder
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)
    if os.path.isdir(folder_path):
        read_images_from_folder(folder_path, folder)  # Call the recursive function

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Preprocess the images
images = images.astype('float32') / 255.0

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Reshape the images for CNN input
images = images.reshape(-1, 28, 28, 1)

# Load the saved model
saved_model_path = 'dset.model'
if os.path.isfile(saved_model_path):
    model = load_model(saved_model_path)
else:
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    print("Saved model not found. Please train the model first.")

# Function to predict custom images
def predict_custom_image(image_path):
    # Read and preprocess the custom image
    custom_image = cv2.imread(image_path, 0)
    if custom_image is None:
        print(f"Error reading image: {image_path}")
        return
    custom_image = cv2.resize(custom_image, (28, 28))
    custom_image = custom_image.astype('float32') / 255.0
    custom_image = custom_image.reshape(1, 28, 28, 1)

    # Make predictions using the loaded model
    predictions = model.predict(custom_image)
    predicted_label = np.argmax(predictions)
    predicted_class_name = label_encoder.classes_[predicted_label]

    return predicted_class_name

# Test with a custom image
custom_image_path = 'min1.png'  # Replace with the path to your custom image
prediction = predict_custom_image(custom_image_path)
print(f"Prediction for custom image: {prediction}")
