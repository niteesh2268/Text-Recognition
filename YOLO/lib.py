
import pytesseract
from pytesseract import Output
import  matplotlib.pyplot as plt
import cv2
img = cv2.imread('./images/100.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop_img = img[y:y+h, x:x+w]
    plt.imshow(crop_img)
    plt.savefig(str(i))
cv2.imshow('img', img)
#cv2.waitKey(0)
'''
import cv2
import pytesseract
def to_text(path):
  img = cv2.imread(path)
  print(type(img))
  from PIL import Image
  # Adding custom options
  text = pytesseract.image_to_string(img)
  return text

print(to_text("./images/100.jpg"))
'''