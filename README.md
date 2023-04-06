# DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR
This work is an extension of the project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR adding the possibility to detect the speed

The requirements are exactly the same as those indicated in the aforementioned project.

Downloaded the project, execute the pythom program

VIDEODetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR.py


The test program VIDEODetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR.py is only prepared to work with the attached Traffic IP Camera video.mp4 test video,
dowloaded from 
https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources, 
since speed detection is performed over a region of the video, marked with a green rectangle, whose depth coincides with the length of a parking space that appears in the video and according to the following formula and parameters:

"""
          
         
           fps=25 #frames per second of video, see its properties

           fpsReal= fps/SpeedUpFrames # To speed up the process only one of SpeedUpFrames
                                      # is considered, SpeedUpFrames=5
           lengthRegion=4.5 #the depth of the considered region corresponds
                           # to the length of a parking space which is usually 4.5m

           Snapshots detected in the video region

           Speed (Km/hour)=lenthRegion * fpsReal * 3.6 / Snapshots

          Where 3.6 = (3600 sec./ 1 minute) / (1Km/ 1000m)

As a result, the console gets the following output:

AR606L Speed: 27.0Kmh snapshots: 3

EAR6061 Speed: 81.0Kmh snapshots: 1

AE670S Speed: 81.0Kmh snapshots: 1

APHI88 Speed: 81.0Kmh snapshots: 1

A3K96 Speed: 40.5Kmh snapshots: 2

A3K961 Speed: 81.0Kmh snapshots: 1

A968B6 Speed: 40.5Kmh snapshots: 2

AV6190 Speed: 40.5Kmh snapshots: 2


In which it is verified that the speed is determined by the number of snapshots in the delimited region of interest and errors coming from false registration detections such as EAR6061 that is a false registration and A3k961 that only detects a snapshot because there are snapshots in which is detected as A3K96

A camera with more frames per second is needed, a computer  with better features and better license plate detection.

You also get a logging file VIDEOLicenseResults.txt with the detected license plates

and a summary file: VIDEOLicenseSummary.txt with the following fields:

- License detected
- number of snapshots in the region of interest
- time of first snapshot
- last snapshot time
- estimated speed

References:

https://github.com/amanraja/vehicle-speed-detection-

https://www.ijcseonline.org/pub_paper/124-IJCSE-06271.pdf

https://medium.com/@raja_8462/an-efficient-approach-for-vehicle-speed-detection-1fce82aacaf2

https://pypi.org/project/paddleocr/

https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

https://public.roboflow.com/object-detection/license-plates-us-eu/3

https://docs.ultralytics.com/python/

https://medium.com/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c

https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6

https://medium.com/adevinta-tech-blog/text-in-image-2-0-improving-ocr-service-with-paddleocr-61614c886f93

https://machinelearningprojects.net/number-plate-detection-using-yolov7/

https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters

Filters:

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

https://gist.github.com/endolith/255291#file-parabolic-py

https://learnopencv.com/otsu-thresholding-with-opencv/

https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05

https://programmerclick.com/article/89421544914/

https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438

https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/

https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e

https://github.com/victorgzv/Lighting-correction-with-OpenCV

https://medium.com/@yyuanli19/using-mnist-to-visualize-basic-conv-filtering-95d24679643e

