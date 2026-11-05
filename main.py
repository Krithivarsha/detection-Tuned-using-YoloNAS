import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

video_path = 'pp4.mp4'
image_path = ''  


cap = cv2.VideoCapture(video_path) if video_path else cv2.imread(image_path)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()

# Define the classes to detect
classes_to_detect = ['person', 'car', 'boat', 'dog', 'cat', 'cow']

while True:
    if isinstance(cap, cv2.VideoCapture):
        ret, frame = cap.read()
    else:
        frame = cap
        ret = frame is not None

    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in classes_to_detect:  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("RGB", frame)
    key = cv2.waitKey(25)  
    if key & 0xFF == ord('q'):
        break

if isinstance(cap, cv2.VideoCapture):
    cap.release()

cv2.destroyAllWindows()
