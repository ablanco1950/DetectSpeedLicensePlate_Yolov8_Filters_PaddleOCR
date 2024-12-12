# del boletin de roboflow de 29/11/2024
#https://docs.ultralytics.com/es/guides/speed-estimation/#how-do-i-estimate-object-speed-using-ultralytics-yolo11

import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Traffic IP Camera video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speed_obj = solutions.SpeedEstimator(
    region=[(0, 360), (1280, 360)],
    model="yolo11n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    im0 = speed_obj.estimate_speed(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
