import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import io
import cairosvg

def is_valid_svg(filename):
    """Check if the file is a valid SVG by trying to parse it with ElementTree."""
    try:
        ET.parse(filename)
        return True
    except ET.ParseError:
        return False

def get_image_from_svg(filename, target_size=(750, 750)):
    """Convert SVG to a CV2 image with the target size."""
    with open(filename, 'r') as f:
        svg_content = f.read()

    png_io = io.BytesIO()
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=png_io, output_width=target_size[0], output_height=target_size[1])
    png_bytes = png_io.getvalue()
    png_array = np.frombuffer(png_bytes, dtype=np.uint8)
    
    img = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)
    return img

def save_image_with_label(img, label):
    """Save an image with a label."""
    filename = f"{label}.png"
    cv2.imwrite(filename, img)

def compute_skeleton(binary_mask):
    """Compute the skeleton (center lines) of the binary mask."""
    skeleton = np.zeros(binary_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(binary_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_mask, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_mask = eroded.copy()
        
        done = (cv2.countNonZero(binary_mask) == 0)
    return skeleton

def find_problem_areas(img, min_stroke_width=5, filtering_kernel_size=13):
    """Find and mark problem areas on the original image."""
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    # Threshold the image
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Compute the skeleton
    skeleton = compute_skeleton(binary_mask.copy())
    
    # Erode the binary mask
    kernel = np.ones((min_stroke_width, min_stroke_width), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel)
    
    # Compute the difference map
    diff = skeleton - eroded_mask
    
    diff_filtered = diff
    
    # Apply blurring to get rid of small dots
    blurred = cv2.blur(diff_filtered, (filtering_kernel_size, filtering_kernel_size))
    
    # Threshold the blurred image
    _, thresholded = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around the identified problem areas on the original image
    drawn = False

    marked_img = img.copy()
    for contour in contours:
        if cv2.contourArea(contour) > min_stroke_width:  # Filter out small areas
            x, y, w, h = cv2.boundingRect(contour)
            if (w * h) < 100:
                continue
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(marked_img, center, radius, (0, 0, 255, 255), 2)  # Draw red circles
            drawn = True
    
    if drawn:
        save_image_with_label(marked_img, "marked_image")
        return True

    return False

def has_transparent_background(img):
    """Check if the image has a transparent background."""
    alpha_channel = img[:, :, 3]  # Alpha channel is the 4th channel
    return np.min(alpha_channel) == 0  # If the minimum value in the alpha channel is 0, there's some transparency

def all_solid_parts_are_white(img):
    """Ensure solid parts of the image are all white."""
    solid_regions = img[img[:, :, 3] == 255]  # Where alpha is 255 (fully opaque)
    return np.all(solid_regions[:, :3] == [255, 255, 255])  # Check if all RGB values in solid regions are white

def sufficient_solid_area(img, min_solid_percentage=1):
    """Ensure at least a certain percentage of the image is solid."""
    total_pixels = img.shape[0] * img.shape[1]
    solid_pixels = np.sum(img[:, :, 3] == 255)
    solid_percentage = (solid_pixels / total_pixels) * 100
    return solid_percentage >= min_solid_percentage

def validate_logo(filename):
    """Validate the logo based on the given criteria."""
    if not is_valid_svg(filename):
        sys.stderr.write("Invalid SVG file.\n")
        return 1
    
    img = get_image_from_svg(filename)
    
    if not has_transparent_background(img):
        sys.stderr.write("Background is not transparent.\n")
        return 2
    
    if not all_solid_parts_are_white(img):
        sys.stderr.write("Not all solid parts of the image are white.\n")
        return 3
    
    if not sufficient_solid_area(img):
        sys.stderr.write("Insufficient solid area in the image.\n")
        return 4
    
    problem_areas_exist = find_problem_areas(img)
    
    if problem_areas_exist:
        sys.stderr.write("Problem areas detected. See image for more details.\n")
        return 5
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python validate_logo.py <filename.svg>\n")
        sys.exit(1)
    
    filename = sys.argv[1]
    result = validate_logo(filename)
    sys.exit(result)
