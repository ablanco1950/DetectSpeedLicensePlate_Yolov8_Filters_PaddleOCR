# DetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR
This work is an extension of the project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR adding the possibility to detect the speed

The requirements are exactly the same as those indicated in the aforementioned project.

The test program VIDEODetectSpeedLicensePlate_Yolov8_Filters_PaddleOCR.py is only prepared to work with the attached Traffic IP Camera video.mp4 test video, since speed detection is performed over a region of the video, marked with a green rectangle, whose depth coincides with the length of a parking space that appears in the video and according to the following formula and parameters:

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

A camera with more frames per second is needed, a team with better features and better license plate detection.

You also get a logging file VIDEOLicenseResults.txt with the detected license plates

and a summary file: VIDEOLicenseSummary.txt with the following fields:

-License detected
- number of snapshots in the region of interest
- time of first snapshot
- last snapshot time
- estimated speed
