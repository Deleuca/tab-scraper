import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import shutil
import pytesseract
from PIL import Image

video = input("Enter the video path: ")
cap = cv.VideoCapture(video)


def is_tab_frame(img, debug=False):
    edges = cv.Canny(img, 50, 150, apertureSize=3)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    horizontal_lines = 0

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if abs(angle - 90) < 5:  # near-horizontal
                horizontal_lines += 1

    if debug:
        print("Horizontal lines detected:", horizontal_lines)

    return bool(int(horizontal_lines >= 5))  # tune threshold

# 1. Filtering frames

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

frame_number = 0
tab_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_number % 30 == 0:
        if is_tab_frame(frame):
            tab_frames.append(frame)
        if cv.waitKey(20) and 0xFF == ord("q"):
            break

    if frame_number % 500 == 0:
        print(frame_number, "/", total_frames)
    frame_number += 1

cap.release()
cv.destroyAllWindows()

# 2. Defining dimension

plt.imshow(cv.cvtColor(tab_frames[0], cv.COLOR_BGR2RGB))
plt.title("Sample")
plt.axis("on")
plt.show()

def prompt_for_rectangle():
    print("Please enter the coordinates of the rectangle containing the tab area.")
    print("You will be asked for two points: top-left and bottom-right.")

    def get_point(name):
        while True:
            try:
                coords = input(f"Enter the {name} point as 'x y' (e.g., 100 200): ").strip().split()
                if len(coords) != 2:
                    raise ValueError("Please enter exactly two numbers.")
                x, y = map(int, coords)
                return (x, y)
            except ValueError as e:
                print(f"Invalid input: {e}. Try again.")

    top_left = get_point("top-left")
    bottom_right = get_point("bottom-right")

    print(f"Tab region defined: top-left={top_left}, bottom-right={bottom_right}")
    return top_left, bottom_right

top_left, bottom_right = prompt_for_rectangle()

# 3. Process key frames

burn_in_done = False
initial_frame = None
last_saved_frame = None

filtered_tab_frames = []

for frame in tab_frames:
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped = frame[y1:y2, x1:x2]

    if initial_frame is None:
        initial_frame = cropped
        continue

    if not burn_in_done:
        similarity = ssim(
            cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY),
            cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        )

        if similarity < 0.85:  # diagram likely gone
            burn_in_done = True
            last_saved_frame = cropped
            filtered_tab_frames.append(cropped)
        continue

    similarity = ssim(
        cv.cvtColor(last_saved_frame, cv.COLOR_BGR2GRAY),
        cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    )

    if similarity < 0.95:  # new tab section
        filtered_tab_frames.append(cropped)
        last_saved_frame = cropped

def browse_frames(frames):
    index = 0
    total = len(frames)

    while True:
        current_frame = frames[index]
        cv.imshow(f"Frame {index+1}/{total}", current_frame)

        key = cv.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n') and index < total - 1:
            index += 1
        elif key == ord('p') and index > 0:
            index -= 1

        cv.destroyAllWindows()

    cv.destroyAllWindows()

# 4. Write key frames

def save_tab_frames(frames, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.glob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    saved_paths = []

    for idx, curr_frame in enumerate(frames):
        path = output_dir / f"frame_{idx:03d}.png"
        cv.imwrite(str(path), curr_frame)
        saved_paths.append(path)

    return saved_paths


def stitch_frames(frames, output_path):
    if not frames:
        raise ValueError("No frames to stitch")

    # Resize all to same width (smallest width)
    min_width = min(f.shape[1] for f in frames)
    resized_frames = [cv.resize(f, (min_width, int(f.shape[0] * min_width / f.shape[1]))) for f in frames]

    stitched = cv.vconcat(resized_frames)
    cv.imwrite(str(output_path), stitched)
    print(f"Stitched image saved to {output_path}")
    return output_path

stitched_image_path = Path("stitched_input.png")
stitch_frames(filtered_tab_frames, stitched_image_path)

# 5. OCR

# Optional: Whitelist only characters that appear in tabs
tesseract_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789-ehABCDEFGabcdefg|/\\-'

def ocr_image(image_path):
    """Run OCR on a given image file path."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, config=tesseract_config)

def ocr_images_in_dir(image_paths):
    """Run OCR on multiple image paths and return concatenated text."""
    all_text = []
    for path in image_paths:
        text = ocr_image(path)
        all_text.append(text.strip())
    return "\n\n".join(all_text)

output_dir = Path("tab_frames")
image_paths = save_tab_frames(filtered_tab_frames, output_dir)

stitched_text = ocr_image(stitched_image_path)
per_frame_text = ocr_images_in_dir(image_paths)

# Pick the better one (based on length for now)
if len(per_frame_text) > len(stitched_text):
    print("Using OCR from individual frames (better character yield).")
    tab_text = per_frame_text
else:
    print("Using OCR from stitched image (cleaner result).")
    tab_text = stitched_text

# Save to file
with open("extracted_tab.txt", "w") as f:
    f.write(tab_text)

print("Tab OCR extraction complete. Saved to 'extracted_tab.txt'.")
