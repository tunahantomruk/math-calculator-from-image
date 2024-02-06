import cv2
import pytesseract

# Load the image
image = cv2.imread('4+3.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to extract the characters and symbols
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours of the characters and symbols
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store the recognized values and symbols
values = []
symbols = []

# Loop through the contours and recognize each character/symbol
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi, lang='eng', config='--psm 10')
    if text.isdigit():
        values.append(int(text))
    elif text in ['+', '-', '*', '/']:
        symbols.append(text)

# Perform arithmetic operations based on the recognized values and symbols
#result = values[0]
if len(values) > 0:
    result = values[0]
else:
    result = 0

for i in range(len(symbols)):
    if symbols[i] == '+':
        result += values[i+1]
    elif symbols[i] == '-':
        result -= values[i+1]
    elif symbols[i] == '*':
        result *= values[i+1]
    elif symbols[i] == '/':
        result /= values[i+1]

print('The recognized values are:', values)
print('The recognized symbols are:', symbols)
print('The calculated result is:', result)
