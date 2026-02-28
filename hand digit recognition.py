import cv2
import easyocr
import numpy as np

# Image path
image_path = r"D:\aiml ass project\image.jpg"

# Load image
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15, 3
)

# OCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Detect digits only
results = reader.readtext(thresh, allowlist='0123456789')

# Sort by y-position (top to bottom)
results = sorted(results, key=lambda x: x[0][0][1])

print("\nRecognized Digits (Line by Line):\n")

lines = []
current_line = []
previous_y = None

for (bbox, text, prob) in results:
    y = bbox[0][1]

    if previous_y is None or abs(y - previous_y) < 30:
        current_line.append(text)
    else:
        lines.append(current_line)
        current_line = [text]

    previous_y = y

if current_line:
    lines.append(current_line)

for line in lines:
    print("".join(line))