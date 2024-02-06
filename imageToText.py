import pytesseract as tess
from PIL import Image
import re

image = Image.open('4+7_h.png')

text = tess.image_to_string(image)

op="4+5+"

parse1 = op.split("+")[0]

parse2 = op.split("+")[1]

# Define regular expression to match symbols
symbol_regex = r"[-+/*]"

# Find matches for symbols
symbol_matches = re.findall(symbol_regex, op)

#loop içinde ilk sembole göre(symbol_matches[0]) 1. ve 2. sayılara işlem uygulayacak
#bunun sonucu ile [1] sembolü ve 3.sayı aksiyona girecek
parts = text.split("+")
#second_number = parts[1]
created_val = None

if symbol_matches[0]=='+':
    created_val = int(parse1) + int(parse2);

size = len(symbol_matches)
i=0;
while(size>i):
    i+=1;
    

print("parse1:",parse1)
print("parse2:",parse2) 
print(symbol_matches)
print('')
print(op,"=",created_val)
