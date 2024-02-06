import tkinter as tk
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog
from PIL import ImageTk, Image
import re

# Load the saved model
model = tf.keras.models.load_model('datav.model')

# Create the Tkinter window
window = Tk()
window.title("Expression Calculator")

# Load the image
image = Image.open("kidspic.png")
image = image.resize((400, 300), Image.ANTIALIAS)  # Resize the image to fit the window

# Create a PhotoImage object from the image
photo = ImageTk.PhotoImage(image)

# Create a label widget with the PhotoImage as the background
background_label = tk.Label(window, image=photo)
background_label.pack(fill=tk.BOTH, expand=tk.YES)


# Create a label to display the result
result_label = Label(window, text="Result: ")
result_label.pack()

# Function to handle image processing and calculation
def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path, 0)  # Load the image in grayscale

    # Threshold the image to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Margin value for expanding the bounding boxes
    margin = 5

    # Create an empty string to store the predicted characters
    expression = ""

    # Iterate over the contours and extract the segmented characters
    for contour in contours:
        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Expand the bounding box coordinates with a margin
        x -= margin
        y -= margin
        w += 2 * margin
        h += 2 * margin

        # Ensure that the expanded bounding box is within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Crop the character region from the image
        character_image = image[y:y+h, x:x+w]

        # Preprocess the character image
        character_image = cv2.resize(character_image, (28, 28))
        character_image = np.invert(character_image)
        character_image = character_image.reshape(1, 28, 28, 1)
        character_image = character_image / 255.0

        # Perform prediction using the model
        prediction = model.predict(character_image)
        predicted_class = np.argmax(prediction)

        # Add the recognized character to the expression string
        if predicted_class < 10:
            expression += str(predicted_class)
        else:
            symbol_mapping = {10: '+', 11: '-', 12: '*', 13: '/'}
            expression += symbol_mapping[predicted_class]

        # Display or save the segmented character image
        plt.imshow(character_image.reshape(28, 28), cmap='gray')
        plt.title("Segmented Character")
        plt.axis('off')
        plt.show()

    # Print the final expression string
    print("Expression:", expression)

    # Tokenize the extracted text
    tokens = tokenize(expression)

    # Perform the calculations if tokens are present
    if tokens:
        if tokens[-3] == '=' and tokens[-2].isdigit():
            inputResult = int(tokens[-2])
            print("input result:", inputResult)
            tokens = tokens[:-3]  # Remove the '= number' part
        else:
            inputResult = None

        result = parse_expression(tokens)
        result_label.config(text="Result: " + str(result))

        if inputResult is not None:
            if result == inputResult:
                print("The result matches the expected result.")
            else:
                print("The result does not match the expected result.")
    else:
        print("Invalid input. Please enter a valid arithmetic expression.")

# Tokenize the input string using regular expressions
def tokenize(expression):
    return re.findall(r'\d+|\D', expression)

# Convert the symbol to the corresponding arithmetic operator
operator = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x / y
}

# Define parsing functions for recursive descent parsing
def parse_number(tokens):
    return int(tokens.pop(0))

def parse_factor(tokens):
    if tokens[0].isdigit():
        return parse_number(tokens)
    elif tokens[0] == '(':
        tokens.pop(0)
        result = parse_expression(tokens)
        if tokens[0] == ')':
            tokens.pop(0)
        return result

    raise ValueError("Invalid expression")

def parse_term(tokens):
    left = parse_factor(tokens)

    while tokens and tokens[0] in ('*', '/'):
        operator_token = tokens.pop(0)
        right = parse_factor(tokens)
        left = operator[operator_token](left, right)

    return left

def parse_expression(tokens):
    left = parse_term(tokens)

    while tokens and tokens[0] in ('+', '-'):
        operator_token = tokens.pop(0)
        right = parse_term(tokens)
        left = operator[operator_token](left, right)

    return left

# Function to handle file drop event
def on_drop(event):
    # Get the dropped file path
    file_path = event.data

    # Process the dropped image
    process_image(file_path)

# Function to open file dialog on button click
def open_file_dialog():
    # Show the file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    # Process the selected image
    process_image(file_path)

# Create a label for the drop area
drop_label = Label(window, text="Drop an image here")
drop_label.pack(pady=20)

# Bind the drop event to the drop label
drop_label.bind("<<Drop>>", on_drop)


# Create a button to open the file dialog
open_button = Button(window, text="Open", command=open_file_dialog)
open_button.pack(pady=10)

# Start the Tkinter main loop
window.mainloop()
