import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import shutil
import sys

if len(sys.argv) != 2:
    print("Usage: python scraper.py /path/to/vid")
    sys.exit(1)

video = sys.argv[1]
cap = cv.VideoCapture(video)


# 1. Filtering frames

def is_tab_frame(img, debug=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    horizontal_lines = 0

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle < 5 or angle > 175:  # horizontal lines
                horizontal_lines += 1

    if debug:
        print("Horizontal lines detected:", horizontal_lines)

    return horizontal_lines >= 5  # or adjust threshold


def print_progress(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

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
    if frame_number % 240 == 0:
        print_progress(frame_number, total_frames)
    frame_number += 1

print_progress(frame_number, total_frames)
cap.release()
cv.destroyAllWindows()
print()

# 2. Defining dimension

plt.imshow(cv.cvtColor(tab_frames[0], cv.COLOR_BGR2RGB))
plt.title("Sample")
plt.axis("on")
plt.show()

def prompt_for_rectangle():
    print("Please enter the coordinates of the rectangle containing the tab area.")
    print("You will be asked for two points: top-left and bottom-right.")
    print()

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
    print()
    return top_left, bottom_right

top_left, bottom_right = prompt_for_rectangle()

# 3. Process key frames

burn_in_done = False
initial_frame = None
last_saved_frame = None

filtered_tab_frames = []

skip_first_frame = input("Skip first key frame? [y/N]: ").strip().lower()
skip_first_frame = skip_first_frame == 'y'

for frame in tab_frames:
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped = frame[y1:y2, x1:x2]

    if initial_frame is None:
        initial_frame = cropped
        if not skip_first_frame:
            filtered_tab_frames.append(cropped)
            last_saved_frame = cropped
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

    if similarity < 0.91:  # new tab section
        filtered_tab_frames.append(cropped)
        last_saved_frame = cropped

def browse_frames(frames):
    # Add a call to this function if you'd like a sanity check
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


def clean_bw_conversion(img, blur_kernel=3, block_size=11, C=2, invert=False):
    """
    Convert colored image to clean black-and-white with minimal noise.

    Parameters:
    - img: Input BGR image
    - blur_kernel: Gaussian blur kernel size (should be odd, 0 to disable)
    - block_size: Adaptive threshold neighborhood size (must be odd)
    - C: Constant subtracted from the mean
    - invert: If True, flips black/white (useful for dark backgrounds)
    """
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Optional: Mild blur to reduce high-frequency noise
    if blur_kernel > 0:
        gray = cv.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Adaptive thresholding (better than global thresholding)
    thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, block_size, C
    )

    # Invert if needed (for dark backgrounds)
    if invert:
        thresh = 255 - thresh

    return thresh


def stitch_frames(frames):
    if not frames:
        raise ValueError("No frames to stitch")

    min_width = min(f.shape[1] for f in frames)
    resized_frames = [cv.resize(f, (min_width, int(f.shape[0] * min_width / f.shape[1]))) for f in frames]

    img = cv.vconcat(resized_frames)
    return img


output_path = Path("output.png")
stitch = stitch_frames(filtered_tab_frames)

gray = input("Apply b/w conversion? [Y/n]: ")
if gray == "n":
    cv.imwrite(str(output_path), stitch)
else:
    cv.imwrite(str(output_path), clean_bw_conversion(stitch))

print()
print(f"Stitched image saved to {output_path}")

