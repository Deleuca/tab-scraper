FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget nano openjdk-11-jdk python3 python3-pip python3-dev \
    build-essential unzip tesseract-ocr libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Audiveris
RUN git clone https://github.com/Audiveris/audiveris.git /opt/audiveris \
 && cd /opt/audiveris \
 && ./gradlew build -x test --no-daemon
ENV PATH="/opt/audiveris/bin:${PATH}"

# Clone OCR-tabber for guitar tab OCR
RUN git clone https://github.com/sn0v/OCR-tabber.git /opt/ocr-tabber

RUN pip3 install opencv-python numpy scikit-image music21 pytesseract

WORKDIR /opt/ocr-tabber
RUN pip3 install -r requirements.txt || true

# Clone your project into /app
RUN git clone https://github.com/your-username/TabScraper.git /app
WORKDIR /app
RUN chmod +x scraper.py

# Default command drops you into a shell
CMD ["/bin/bash"]
