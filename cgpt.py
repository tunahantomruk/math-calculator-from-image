import cv2
import pytesseract
import sympy

# Load the image
image = cv2.imread('handwritten_number.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to remove noise
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform OCR to extract the text from the image
text = pytesseract.image_to_string(thresh, config='--psm 11')

# Parse the mathematical expression using sympy
expr = sympy.parse_expr(text)

# Evaluate the expression and print the result
result = expr.evalf()
print(result)
