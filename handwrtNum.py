import pytesseract
from PIL import Image

# Load the image
image = Image.open('sdf.png')

# Set the path to the Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Recognize the handwritten number
number = pytesseract.image_to_string(image, lang='eng', config='--psm 10')

#result = eval(number)

print('The recognized number is:', number)
#print("result:",result)
