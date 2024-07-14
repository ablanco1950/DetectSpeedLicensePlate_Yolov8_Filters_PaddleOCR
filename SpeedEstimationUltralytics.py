# https://docs.ultralytics.com/guides/speed-estimation/#how-do-i-estimate-object-speed-using-ultralytics-yolov8
# https://www.ultralytics.com/es/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects

import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("Traffic IP Camera video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speed_obj = solutions.SpeedEstimator(
    #reg_pts=[(0, 360), (1280, 360)],
    reg_pts=[(0, 485), (1280, 485)],
    names=names,
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
