
import cv2
import numpy as np

image = cv2.imread("data/sampleimage5.png")
if image is None:
    raise FileNotFoundError("Image not found at data/sampleimage3.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. Get HSV at (0, 0) - top-left corner
x, y = 0, 0
hsv_pixel = hsv[y, x]  # OpenCV uses (y, x) indexing

# 4. Print the HSV values
print(f"HSV at ({x}, {y}): H={hsv_pixel[0]}, S={hsv_pixel[1]}, V={hsv_pixel[2]}")