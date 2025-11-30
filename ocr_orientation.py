import easyocr
import numpy as np
import cv2

def detect_orientation(image_array: np.ndarray) -> str:
    """
    Detects the orientation of a chess board image using OCR.

    Args:
        image_array: A NumPy array representing the image.

    Returns:
        A string indicating the orientation (e.g., "a1_bottom_left", "h1_bottom_left",
        "a8_bottom_left", "h8_bottom_left"), or "unknown" if not detected.
    """
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_array)

    # Store detected numbers and letters with their bounding box centers
    numbers = []
    letters = []

    for (bbox, text, prob) in results:
        # Clean up text: remove non-alphanumeric characters and convert to lowercase
        cleaned_text = ''.join(filter(str.isalnum, text)).lower()
        
        # Get center of bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        if cleaned_text == '1' or cleaned_text == '8':
            numbers.append({'text': cleaned_text, 'center_x': center_x, 'center_y': center_y})
        elif cleaned_text == 'a' or cleaned_text == 'h':
            letters.append({'text': cleaned_text, 'center_x': center_x, 'center_y': center_y})
    if numbers and letters:
        return f"{letters[0]["text"]}{numbers[0]["text"]}"
    return ""

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy image (e.g., a black image with white text)
    dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Simulate different orientations by drawing text at different locations
    # For a1_bottom_left: 'a' and '1' at bottom-left
    # cv2.putText(dummy_image, 'a', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(dummy_image, '1', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # For h8_bottom_left: 'h' and '8' at bottom-left
    cv2.putText(dummy_image, 'h', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dummy_image, '8', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # For a8_bottom_left: 'a' and '8' at bottom-left
    # cv2.putText(dummy_image, 'a', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(dummy_image, '8', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # For h1_bottom_left: 'h' and '1' at bottom-left
    # cv2.putText(dummy_image, 'h', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.putText(dummy_image, '1', (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    print("Detecting orientation...")
    orientation = detect_orientation(dummy_image)
    print(f"Detected orientation: {orientation}")

    # You can display the dummy image to verify
    # cv2.imshow("Dummy Image", dummy_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
