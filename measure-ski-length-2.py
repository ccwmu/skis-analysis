import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Loads an image from the specified path.
    Raises an error if the image is not found.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return image

def to_hsv_mask(image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
    """
    Converts an image to HSV color space and creates a binary mask
    where pixels within the specified HSV range are white (255),
    and others are black (0).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask

def preprocess_mask(mask: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """
    Applies morphological operations to clean up the binary mask:
    - Closing (dilation followed by erosion) to close small holes inside objects
    - Dilation to slightly enlarge the detected regions
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_dilated = cv2.dilate(mask_closed, kernel, iterations=1)
    return mask_dilated

def filter_ski_contours(contours, min_area=500, aspect_ratio_threshold=3.0):
    """
    Filters contours to find those that likely correspond to skis:
    - Rejects small contours under the min_area threshold
    - Keeps only elongated shapes with a high aspect ratio
    """
    ski_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        width, height = cv2.minAreaRect(contour)[1]
        if min(width, height) == 0:
            continue  # skip invalid shapes

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > aspect_ratio_threshold:
            ski_contours.append(contour)

    return ski_contours

def draw_detected_skis(image: np.ndarray, contours) -> np.ndarray:
    """
    Draws green boxes around detected skis on the image.
    Also logs the length and area of each ski contour to the console.
    """
    output = image.copy()
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

        length_px = max(rect[1])
        area = cv2.contourArea(contour)
        print(f"Ski {i + 1}: Length = {length_px:.2f} px, Area = {area:.2f} px²")

    return output

def main():
    # Filepath to input image
    image_path = "data/sampleimage.jpg"

    # HSV range for detecting green-colored skis
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Step 1: Load image from file
    image = load_image(image_path)

    # Step 2: Convert image to HSV and create binary mask for green regions
    mask = to_hsv_mask(image, lower_green, upper_green)

    # Step 3: Apply morphological operations to clean the mask
    processed_mask = preprocess_mask(mask, kernel_size=25)

    # Step 4: Find contours from the processed mask
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter contours based on area and shape to isolate skis
    ski_contours = filter_ski_contours(contours, 100000)

    # Step 6: Draw results and print stats
    output = draw_detected_skis(image, ski_contours)

    # Step 7: Show results
    cv2.namedWindow("Processed Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Mask", processed_mask)
    cv2.namedWindow("Detected Skis", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Skis", output)
    cv2.imshow("Processed Mask", processed_mask)
    cv2.imshow("Detected Skis", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
