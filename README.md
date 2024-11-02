# Automatic Number Plate Recognition (ANPR) Application

This project is an Automatic Number Plate Recognition (ANPR) application that uses the YOLOv7 model to detect license plates in images and videos. The application is designed with a graphical user interface (GUI) using PyQt5, allowing users to load videos or use a webcam for real-time processing.

## Features

- **License Plate Detection**: Utilizes the YOLOv7 model for detecting license plates.
- **Text Recognition on Plates**: Uses EasyOCR to extract text from detected plates.
- **Data Storage**: Saves detection results in an SQLite database.
- **User-Friendly Interface**: Designed with PyQt5 for easier interaction with the application.
- **Input from Video and Webcam**: Allows loading of videos or using a webcam for live processing.



### Requirements

Before running the application, make sure to install the following libraries:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install PyQt5
pip install easyocr
pip install deep-sort-realtime
