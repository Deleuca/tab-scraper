# tab-scraper

This script accepts a YouTube URL as an argument and parses the video for tabs or sheet music. It then stitches the tabs together to make them more amenable for further practice.

Note that the onus is on the user to define the frame in the video designated for notation. Also note that HD video is not supported by the program. 

Required libraries: 
- opencv-python
- numpy
- matplotlib
- scikit-image
- slugify

The script also requires yt-dlp for video download purposes. To install on Arch: `sudo pacman -S yt-dlp`

To clone the repository, run `git clone https://github.com/deleuca/tab-scraper.git && cd tab-scraper`

To install the relevant packages, run: `pip install -r requirements.txt`

Usage: `python scraper.py <youtube-url>`
