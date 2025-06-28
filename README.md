# tab-scraper

This script accepts a video as an argument and parses the video for tabs or sheet music. It then stitches the tabs together to make them more amenable for practice.

Note that the onus is on the user to define the frame in the video designated for notation. Also note that HD video is not supported by the program. 

Required libraries: 
- opencv-python
- numpy
- matplotlib
- scikit-image

To clone the repository, run `git clone https://github.com/minigon/tab-scraper.git && cd tab-scraper`

To install the relevant packages, run: `pip install -r requirements.txt`

Usage: `python scraper.py /path/to/vid`
