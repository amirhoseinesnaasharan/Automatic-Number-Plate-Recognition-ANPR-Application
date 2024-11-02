import os
import cv2 as cv
import numpy as np
import sqlite3
import datetime
import torch
from PyQt5 import QtWidgets, QtGui
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort
from easyocr import Reader

# Paths for data and model weights
weights =r'C:\Users\Amir\Desktop\python\code python\number_car\yolov7\yolov7.pt'
device_id = 'cpu'
image_size = 640
data_yaml_path = r"C:\Users\Amir\Desktop\python\code python\number_car\yolov7\ANPR_ir-1\data.yaml"

# Function to train the YOLOv7 model
def train_yolov7():
    os.system(f"python train.py --img {image_size} --batch 16 --epochs 50 --data \"{data_yaml_path}\" --weights yolov7.pt")

# Device and model setup
device = select_device(device_id)
half = device.type != 'cpu'  # Half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # Load FP32 model
stride = int(model.stride.max())  # Model stride
imgsz = check_img_size(image_size, s=stride)  # Check img_size
if half:
    model.half()

# Set up Persian OCR
reader = Reader(['fa'])

# Create SQLite database
def create_database():
    conn = sqlite3.connect('anpr_detection_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY, plate_text TEXT, 
                  timestamp TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

create_database()

# Save detection information to the database
def save_detection(plate_text, confidence):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('anpr_detection_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections (plate_text, timestamp, confidence) VALUES (?, ?, ?)",
              (plate_text, timestamp, confidence))
    conn.commit()
    conn.close()

# Save the plate image
def save_plate_image(plate_region, plate_text, save_path="detections"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, f"{plate_text}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv.imwrite(filename, plate_region)

# Detect plates in the image
def detect_plate(source_image):
    img = letterbox(source_image, image_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    det_confidences = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())
    return plate_detections, det_confidences

# Read the plate text using OCR
def ocr_plate(plate_region):
    rescaled = cv.resize(plate_region, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    grayscale = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    plate_text_easyocr = reader.readtext(grayscale)
    if plate_text_easyocr:
        (bbox, text_easyocr, ocr_confidence) = plate_text_easyocr[0]
        return text_easyocr, ocr_confidence
    else:
        return "_", 0

# PyQt5 User Interface for ANPR application
class ANPRApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Optionally start training the model when the app starts
        # train_yolov7()

    def initUI(self):
        self.setWindowTitle('ANPR Application')
        self.setGeometry(100, 100, 800, 600)
        
        # Buttons
        self.videoBtn = QtWidgets.QPushButton('Select Video', self)
        self.videoBtn.clicked.connect(self.loadVideo)
        
        self.webcamBtn = QtWidgets.QPushButton('Use Webcam', self)
        self.webcamBtn.clicked.connect(self.startWebcam)
        
        # Image display field
        self.imageLabel = QtWidgets.QLabel(self)
        self.imageLabel.resize(640, 480)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.videoBtn)
        layout.addWidget(self.webcamBtn)
        layout.addWidget(self.imageLabel)
        self.setLayout(layout)

    def loadVideo(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.processVideo(video_path)

    def processVideo(self, video_path):
        video = cv.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            plates, confidences = detect_plate(frame)
            for plate, conf in zip(plates, confidences):
                plate_region = frame[plate[1]:plate[3], plate[0]:plate[2]]
                text, conf_ocr = ocr_plate(plate_region)
                save_detection(text, conf_ocr)
                save_plate_image(plate_region, text)
                cv.rectangle(frame, (plate[0], plate[1]), (plate[2], plate[3]), (0, 255, 0), 2)
                cv.putText(frame, text, (plate[0], plate[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame in the UI
            self.displayFrame(frame)

    def displayFrame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qImg = QtGui.QImage(frame.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qImg))

    def startWebcam(self):
        video = cv.VideoCapture(0)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            plates, confidences = detect_plate(frame)
            for plate, conf in zip(plates, confidences):
                plate_region = frame[plate[1]:plate[3], plate[0]:plate[2]]
                text, conf_ocr = ocr_plate(plate_region)
                save_detection(text, conf_ocr)
                save_plate_image(plate_region, text)
                cv.rectangle(frame, (plate[0], plate[1]), (plate[2], plate[3]), (0, 255, 0), 2)
                cv.putText(frame, text, (plate[0], plate[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame in the UI
            self.displayFrame(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()

# Run the application
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWin = ANPRApp()
    mainWin.show()
    # Uncomment the line below if you want to train the model when starting the app
    # train_yolov7()
    sys.exit(app.exec_())
