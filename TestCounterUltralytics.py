"""
Author: https://github.com/pderrenger/pderrenger

https://github.com/orgs/ultralytics/discussions/8112
specifically pderrenger's explanation

and you also have to install a specific version of lap:

inside conda in the scripts directory of the user environment

python pip-script.py install --no-cache-dir "lapx>=0.5.2"

upgrade ultralytics

python pip-script.py install --upgrade ultralytics

If this instruction is executed first it gives an error that "lapx>=0.5.2" had to be installed with the parameters indicated
"""



import cv2
from ultralytics import YOLO, solutions

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture("Traffic IP Camera video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
