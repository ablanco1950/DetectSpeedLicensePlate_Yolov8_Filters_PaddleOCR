# DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR
This work is an extension of the project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR adding the possibility to detect the speed, tracking and counting cars

The requirements are exactly the same as those indicated in the aforementioned project.

A requirements.txt file is attatched.( paddleocr gives some warnings but works)

Of all the tests performed, the one that seems most suitable is based on establishing the speed based on the distances between the current x2 y2 points of the ID assigned by the roboflow tracker and the x2y2 assigned by the tracker in the previous frame, and dividing by the time between frames.

Execute the program 

python DetectSpeed_By_PixelsDistance_And_RoboflowTracker.py (As of 07/29/2025, an incompatibility with the last roboflow tracker has been detected, so this program does not work. While waiting to resolve it, you can continue testing with the rest of the programs listed below.)

the results and comparison with the method based on the number of snapshots in which the car has been detected are:

ID 2 AR606L speed by Snapshots = 18 22 Km/h, by pixel distance = 49. Km/h

ID 3 AR606L speed by Snapshots = 17.23 Km/h, by pixel distance = 37. Km/h

ID 5 AE670S speed by Snapshots = 18 22 Km/h, by pixel distance = 40. Km/h

ID 8 APHI88 speed by Snapshots = 9 45 Km/h, by pixel distance = 122 Km/h

ID -1 APHI88 speed by Snapshots = 1 405 Km/h, by pixel distance = 0 Km/h

ID 10 A3K96 speed by Snapshots = 14 28 Km/h, by pixel distance = 67. Km/h

ID 11 A968B6 speed by Snapshots = 5 81 Km/h, by pixel distance = 41. Km/h

ID 12 AV6190 speed by Snapshots = 12.33 Km/h, by pixel distance = 70. Km/h


prepared to work with the attached Traffic IP Camera video.mp4 test video

ID 2 and ID 3 are the same car, once identified as car and another as truck.

ID -1 is a false detection of tracker


======================================================================================================================

Previously performed tests, order by test date:

Tests based on number of snapshots:

VIDEODetectSpeed_and_ Counter_LicensePlate_Yolov8_Filters_PaddleOCR.py

This presentation participated in a Ready Tensor tournament and has since been updated to feature and leverage improvements brought by Roboflow and Ultralytics.

The test program VIDEODetectSpeed_and_ Counter_LicensePlate_Yolov8_Filters_PaddleOCR.py is only prepared to work with the attached Traffic IP Camera video.mp4 test video,
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

          Where 3.6 = (3600 sec./ 1 hour) * (1Km/ 1000m)
          
          This formula depends on the number of snapshots detected, which depends on the quality and speed of the plate detector, so in any case it has to be adjusted with practical tests in the field.

As a result, the console gets the following output:

AR606L Speed: 27.0Km/h  snapshots: 3

AE670S Speed: 40.5Km/h  snapshots: 2

APHI88 Speed: 81.0Km/h  snapshots: 1

A3K96 Speed: 40.5Km/h  snapshots: 2

A968B6 Speed: 40.5Km/h  snapshots: 2

AV6190 Speed: 27.0Km/h  snapshots: 3


In which it is verified that the speed is determined by the number of snapshots in the delimited region of interest. There is one error from false  detections of plate A3K961 that is detected as A3k96

Adjustments would be necessary with real and verifiable cases.

A camera with more frames per second is needed, a computer  with better features and better license plate detection. 

You also get a logging file VIDEOLicenseResults.txt with the detected license plates

and a summary file: VIDEOLicenseSummary.txt with the following fields:

- License detected
- number of snapshots in the region of interest
- time of first snapshot
- last snapshot time
- estimated speed

And the car`s counter

The main advantage of this method is that it seems to be insensitive to the load that the computer that executes it is supporting, which allows the observed values ​​to be adjusted.

The same thing doesn't seem to happen when running:

SpeedEstimationUltralytics.py

copied from https://docs.ultralytics.com/guides/speed-estimation/

in which the speeds vary at different execution times of the program.

At https://www.ultralytics.com/es/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects
It is indicated that the results depend on the speed of the GPU

With respect the  tracking method,  depends directly on the detection of the license plate. It is more precise but produces errors if the license plate is detected incorrectly.
You can compare the results with those obtained with the tracking functions incorporated in the new ultralytics versions by running:
TestCounterUltralytics.py

For that you must have an upgraded version of ultralytics and the proper version of lap

inside conda in the scripts directory of the user environment

python pip-script.py install --no-cache-dir "lapx>=0.5.2"

upgrade ultralytics

python pip-script.py install --upgrade ultralytics

To see the results it is better to watch the output video

object_counting_output.avi

A test has also been performed using sahi prediction integrated with yolov8

(https://docs.ultralytics.com/es/guides/sahi-tiled-inference/)

Which can be verified by running:

(You need to have installed sahi: pip install -U ultralytics sahi )

VIDEODetectSpeed_and_ Counter_LicensePlate_SahiYolov8_Filters_PaddleOCR.py

The results, barring error or omission, are worse

It is expected to improve the results in subsequent versions by applying the specifications that appear in https://blog.roboflow.com/estimate-speed-computer-vision/ and others

12/12/2024
==========

Another way to estimate speeds in cars is by running this simple program:

python SpeedUltralyticsSolutions.py

Copied from the article:

https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-tracking?utm_source=newsletter&utm_medium=email&utm_term=2024-11-29&utm_campaign=+Ultralytics+Unplugged+-+Snapshot+2

Although it requires creating a separate environment to avoid incompatibilities between yolov8 and yolov11.

In this new environment yolov11 is installed by downloading the latest version of ultralytics

pip install ultralytics

There may be incompatibilities with the installed version of numpy, so you should downgrade it;

pip install numpy==1.23

You should also install:

pip install --no-cache-dir "shapely>=2.0.0"

pip install --no-cache-dir "lapx>=0.5.2"

The results are similar to those of the other tests, although the estimated speeds seem lower.

03/07/2025
Taking advantage of the features of the roboflow tracker that appeared in the article https://medium.com/@harunkurtdev/real-time-object-detection-and-tracking-with-yolo-and-roboflow-trackers-a-complete-python-8a9cd8d16ee3, this version is implemented that seems to detect speeds with greater precision and independently of the computer load.

python DetectSpeed_RoboflowTracker.py

The results appear in the console in this format:

ID 2 Snapshots = 18  22 Km/h

ID 3 Snapshots = 17  23 Km/h

ID 5 Snapshots = 19  21 Km/h

ID 8 Snapshots = 11  36 Km/h

ID -1 Snapshots = 2  202 Km/h

ID 10 Snapshots = 25  16 Km/h

ID 11 Snapshots = 9  45 Km/h

ID 12 Snapshots = 19  21 Km/h

05/07/2025

We are trying to improve the previous version by obtaining the license plate number:

python DetectSpeed_Plate_RoboflowTracker.py

ID 3 Snapshots = 9   45 Km/h  Plate:AR606L

ID 4 Snapshots = 15   27 Km/h  Plate:AR606L

ID 6 Snapshots = 17   23 Km/h  Plate:AE670S

ID 8 Snapshots = 9   45 Km/h  Plate:APHI88

ID -1 Snapshots = 1   405 Km/h  Plate:

ID 9 Snapshots = 8   50 Km/h  Plate:A3K961

ID 10 Snapshots = 6   67 Km/h  Plate:A968B6

ID 11 Snapshots = 11   36 Km/h  Plate:AV6190


ID 3 and ID4 are the same car, once identified as car and another as truck the calculate speed is not real.

ID -1 is a False ID produced by tracker

08/07/2025

New test based on establishing the speed based on the distances between the current x2 y2 points of the ID assigned by the roboflow tracker and the x2 y2 assigned by the tracker (Pixel distance) in the previous frame, and dividing by the time between frames:

python DetectSpeed_By_PixelsDistance_And_RoboflowTracker.py

the results are:

ID 2 AR606L speed by Snapshots = 18 22 Km/h, by pixel distance = 49. Km/h

ID 3 AR606L speed by Snapshots = 17.23 Km/h, by pixel distance = 37. Km/h

ID 5 AE670S speed by Snapshots = 18 22 Km/h, by pixel distance = 40. Km/h

ID 8 APHI88 speed by Snapshots = 9 45 Km/h, by pixel distance = 122 Km/h

ID -1 APHI88 speed by Snapshots = 1 405 Km/h, by pixel distance = 0 Km/h

ID 10 A3K96 speed by Snapshots = 14 28 Km/h, by pixel distance = 67. Km/h

ID 11 A968B6 speed by Snapshots = 5 81 Km/h, by pixel distance = 41. Km/h

ID 12 AV6190 speed by Snapshots = 12.33 Km/h, by pixel distance = 70. Km/h

ID3 and ID4 are the same car, once identified as car and another as truck .

ID -1 is a False ID produced by tracker


The results should be tested with real cases to verify their accuracy or approximation.

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

https://docs.ultralytics.com/guides/speed-estimation/

https://www.ultralytics.com/es/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects

https://blog.roboflow.com/estimate-speed-computer-vision/

In this article on vehicle speed detection in videos from snapshots of detected license plates, a formula is established to determine the speed based on the number of snapshots:
https://medium.com/towards-artificial-intelligence/image-processing-based-vehicle-number-plate-detection-and-speeding-radar-aa375952d0f6

https://medium.com/@harunkurtdev/real-time-object-detection-and-tracking-with-yolo-and-roboflow-trackers-a-complete-python-8a9cd8d16ee3


citations and thanks:

@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}


Tracking and Counting:

https://github.com/pderrenger/pderrenger

https://github.com/pderrenger?tab=repositories

https://github.com/orgs/ultralytics/discussions/8112  (example from pderrenger)

https://towardsdatascience.com/mastering-object-counting-in-videos-3d49a9230bd2

https://docs.ultralytics.com/es/guides/sahi-tiled-inference/


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

https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-tracking?utm_source=newsletter&utm_medium=email&utm_term=2024-11-29&utm_campaign=+Ultralytics+Unplugged+-+Snapshot+2




