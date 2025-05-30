import cv2
import numpy as np

image = cv2.imread("data/sampleimage5.png")
if image is None:
    raise FileNotFoundError("Image not found at data/sampleimage5.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([105, 120, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Use a larger kernel and apply closing then dilation
kernel = np.ones((25, 25), np.uint8)
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_dilated = cv2.dilate(mask_closed, kernel, iterations=1)

contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ski_contours = []
#min_ski_area = 0.01 * image_area
min_ski_area = 500
#print(f"Minimum ski area: {min_ski_area} pixels^2")

for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_ski_area:
        continue
    width, height = cv2.minAreaRect(contour)[1]
    if max(width, height) / min(width, height) > 3:
        ski_contours.append(contour)

output = image.copy()
for i, contour in enumerate(ski_contours):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
    length_px = max(rect[1])
    area = cv2.contourArea(contour)
    print(f"Ski {i + 1}: {length_px:.2f} pixels, Area: {area:.2f} pixels^2")


cv2.imshow("Blue Mask", mask_dilated)
cv2.imshow("Detected Skis", output)
cv2.waitKey(0)
cv2.destroyAllWindows()