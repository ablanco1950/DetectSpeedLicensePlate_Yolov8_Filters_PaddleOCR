# https://medium.com/@harunkurtdev/real-time-object-detection-and-tracking-with-yolo-and-roboflow-trackers-a-complete-python-8a9cd8d16ee3
# adapted and modified by Alfonso Blanco García
import cv2
import supervision as sv
from ultralytics import YOLO
from trackers import SORTTracker  # Roboflow's tracking implementation
import numpy as np

from ReadLicensePlateFromImage import ReadLicencePlateImage

##fps #frames per second of video, see its properties
#
lengthRegion=4.5 #the depth of the considered region corresponds
                 # to the length of a parking space which is usually 4.5m

# Formula
# Snapshots detected in the video region
# Speed (Km/hour)=lenthRegion * fps * 3.6 / Snapshots
#  Where 3.6 = (3600 sec./ 1 hour) * (1Km/ 1000m)
#  This formula depends on the number of snapshots detected, which depends on the quality and speed of the plate detector,
# so in any case it has to be adjusted with practical tests in the field.



cameraPath="traffic_-_27260 (540p).mp4"
cameraPath="Traffic IP Camera video.mp4"

# Zone to be considered

Poligono=[[200,500],[200,700],[1250,700],[1250,500]]
#Poligono=[[200,485],[200,655],[1250,655],[1250,485]]
#Poligono=[[500,485],[500,655],[1250,655],[1250,485]]
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
polygon1 = Polygon(Poligono)
pts1=np.array(Poligono,np.int32)
pts1=pts1.reshape((-1,1,2))

Tab_ID = []
Tab_ID_Snapshots=[]
Tab_ID_Plate=[]

def run_camera():
    # Initialize camera capture (change to 0 for default camera)
    cap = cv2.VideoCapture(cameraPath)
    
    if not cap.isOpened():
        print("Camera could not be opened.")
        return
    # https://medium.com/ai-qa-nexus/seeing-is-coding-unlocking-video-processing-with-python-and-opencv-part-1-introduction-and-bbd5b436ec02
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Load the YOLO model
    model = YOLO("yolo11n.pt")
    
    # Initialize Roboflow's SORT tracker and annotator
    tracker = SORTTracker()
    box_annotator = sv.BoxAnnotator()
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        # draw the line that limits the zone to calaculate de speed        
        cv2.polylines(frame,[pts1],1,(0,255,255))

        
        # Perform object detection
        results = model(frame,verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # https://roboflow.com/how-to-filter-detections/yolov8
        selected_classes = [2, 7]
        detections = detections[np.isin(detections.class_id, selected_classes)]
        
        #detections = detections[detections.confidence > 0.5]
        
        # Update Roboflow tracker with new detections
        tracked = tracker.update(detections)
        
        # Annotate frame with bounding boxes
        annotated_frame = box_annotator.annotate(frame, tracked)
        
        
        # Draw tracker IDs on the frame
        for box, tid in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            
            if(polygon1.contains(Point(int(x2),int(y2)))) or (polygon1.contains(Point(int(x1),int(y1)))):
                    pp=0
            else:
                    continue
                
            #Plate=ReadLicencePlateImage(annotated_frame)    
            label = f"ID {tid}"
            #
            annotated_frame_cropped=annotated_frame[y1:x1,y2:x2]
            if annotated_frame_cropped.shape[1] == 0 or annotated_frame_cropped.shape[0] == 0: continue
            Plate=ReadLicencePlateImage(annotated_frame_cropped) 
            
            if len(Tab_ID) == 0:
                Tab_ID.append(label)
                Tab_ID_Snapshots.append(1)
                Tab_ID_Plate.append(Plate)
                
                #cv2.imwrite(label+".jpg",annotated_frame)
            else:
                # buscar y actualizar los snapshots del ID
                SwEncontrado=0
                for k in range (len(Tab_ID)):
                    if Tab_ID[k]==label:
                        SwEncontrado=1
                        
                        #Tab_ID_Snapshots[k]=Tab_ID_Snapshots[k]+1 
                        if Plate != " " and Plate != "": # Mientras se lean matriculas el punto de referencia
                                                        # del coche está dentro del rectangulo y es valido el snapshot
                                                        # el punto de referencia es la matricula y se considera
                                                        # valido mientras se sigan leyendo matriculas 
                          Tab_ID_Snapshots[k]=Tab_ID_Snapshots[k]+1   
                          Tab_ID_Plate[k]=Plate
                if SwEncontrado == 0:
                    Tab_ID.append(label)
                    Tab_ID_Snapshots.append(1)           
                    Tab_ID_Plate.append(Plate)
                    #cv2.imwrite(label+".jpg",annotated_frame)
            cv2.putText(annotated_frame, label + " " + Plate, (x1, y1 - 10),           
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)       
                    
        
        # Display the annotated frame
        cv2.imshow("Live Video - Object Detection and Tracking", annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Presentar resultados

    for j in range(len(Tab_ID)):
        
        #fps=25 #frames per second of video, see its properties
        #lengthRegion=4.5 #the depth of the considered region corresponds
                          # to the length of a parking space which is usually 4.5m

        # Formula:
        # Snapshots detected in the video region
        # Speed (Km/hour)=lenthRegion * fps * 3.6 / Snapshots
        #  Where 3.6 = (3600 sec./ 1 hour) * (1Km/ 1000m)
        #  This formula depends on the number of snapshots detected, which depends on the quality and speed of the plate detector,
        # so in any case it has to be adjusted with practical tests in the field.
        speed=lengthRegion * fps * 3.6 / Tab_ID_Snapshots[j]
        print ( str(Tab_ID[j]) + " Snapshots = " + str( Tab_ID_Snapshots[j]) + "   " + str(int(speed)) + " Km/h  Plate:" + str(Tab_ID_Plate[j]))
        
if __name__ == "__main__":
    run_camera()
