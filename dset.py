import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
# Set the path to the main dataset folder
main_folder = 'input\\CompleteImages\\All data (Compressed)'

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

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the images for CNN input
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

model.save('in.model')
