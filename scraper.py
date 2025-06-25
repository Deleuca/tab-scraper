import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import shutil
import subprocess
from music21 import converter, stream

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

    if similarity < 0.85:  # new tab section
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

save_tab_frames(filtered_tab_frames, output_dir="frames")

# 5. Run Audiveris

def run_audiveris(image_path: Path, output_dir: Path):
    result = subprocess.run([
        "audiveris",
        "-batch",
        str(image_path),
        "-export",
        "-output", str(output_dir)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error processing {image_path.name}:\n{result.stderr}")
    else:
        print(f"Processed {image_path.name}")

input_folder = Path("frames/")
output_folder = Path("audiveris_output/")
output_folder.mkdir(exist_ok=True)

for image in sorted(input_folder.glob("*.png")):
    run_audiveris(image, output_folder)

# 6. Merge output

def merge_musicxml(folder: Path) -> stream.Score:
    score = stream.Score()
    for xml_file in sorted(folder.glob("*.xml")):
        try:
            part = converter.parse(xml_file)
            score.append(part)
        except Exception as e:
            print(f"Skipping {xml_file.name}: {e}")
    return score

final_score = merge_musicxml(output_folder)
final_score.write("musicxml", fp="final_score.xml")