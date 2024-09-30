import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
import matplotlib.pyplot as plt

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read and resize the image
image = cv2.imread("image.jpg")
if image is None:
    print("Error loading image.")
    sys.exit()

image = imutils.resize(image, width=500)


# Function to display images using matplotlib
def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# Display the original image
display_image(image, "Original Image")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image(gray, "Grayscale Conversion")

# Blur the image to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
display_image(gray, "Bilateral Filter")

# Perform edge detection
edged = cv2.Canny(gray, 170, 200)
display_image(edged, "Canny Edges")

# Find contours in the edged image
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

NumberPlateCnt = None
count = 0

# Loop over the contours
for c in cnts:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If the approximated contour has four points, assume it is the license plate
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

# Mask the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
if NumberPlateCnt is not None:
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    display_image(new_image, "Final Image")
else:
    print("Number plate contour not found.")
    sys.exit()

# Configuration for Tesseract OCR
config = "-l eng --oem 1 --psm 3"

# Run Tesseract OCR on the image
text = pytesseract.image_to_string(new_image, config=config)

# Data is stored in CSV file
raw_data = {"date": [time.asctime(time.localtime(time.time()))], "text": [text]}
df = pd.DataFrame(raw_data)
df.to_csv("data.csv", mode="a", header=False, index=False)

# Print the recognized text
print("Recognized Text:", text)

