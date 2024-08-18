# Face Anonymizer with Mediapipe
This project provides a tool to detect and blur faces in images, videos, or in real-time using a webcam. The script uses OpenCV and Mediapipe's face detection model to process the media and anonymize detected faces.

## Features
### Image Processing: Detects and blurs faces in an image.
### Video Processing: Detects and blurs faces frame-by-frame in a video.
### Real-Time Webcam Processing: Detects and blurs faces in real-time from the webcam feed.

## Requirements
Python 3.6+
OpenCV
Mediapipe
argparse

## Installation
#### Clone the repository:

```bash
 git clone https://github.com/yourusername/FaceAnonymizer.git
 cd FaceAnonymizer
```

#### Create and activate a virtual environment:
```bash
python3 -m venv open-cv-venv
source open-cv-venv/bin/activate
Install the required dependencies:
```

```bash
pip install opencv-python mediapipe argparse

``` 
## Usage
You can run the script in three different modes: image processing, video processing, and real-time webcam processing.

### 1. Image Processing
To process an image, use the following command:
```bash
python realTimeFaceAnonmizer.py --mode image --filePath /path/to/your/image.jpg
```
The processed image with blurred faces will be saved to the output directory as output.jpg.

### 2. Video Processing
To process a video, use the following command:
```bash 
python realTimeFaceAnonmizer.py --mode video --filePath /path/to/your/video.mp4
```
The processed video with blurred faces will be saved to the output directory as output.mp4.

### 3. Real-Time Webcam Processing
To process the webcam feed in real-time, use the following command:

```bash
python realTimeFaceAnonmizer.py --mode webcam
``` 
This will display the webcam feed with detected faces blurred in real-time. Press q to exit the webcam feed.

### Output
All processed media files (images and videos) are saved in the output directory.

### Troubleshooting
Webcam Not Detected: Ensure your webcam is connected and not being used by another application.

File Not Found: Double-check the file path you provide for image and video modes.

Dependencies Not Installed: Make sure all required packages are installed in your environment.