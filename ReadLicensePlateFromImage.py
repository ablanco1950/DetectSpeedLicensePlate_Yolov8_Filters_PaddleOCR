# -*- coding: utf-8 -*-
"""

Gets de License Plate from a car´s image

Created on 05/07/2025

@author: Alfonso Blanco adpating https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR
"""

dirnameYolo="best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo, verbose=False)
class_list = model.model.names
#print(class_list)


######################################################################
from paddleocr import PaddleOCR
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
# https://pypi.org/project/paddleocr/
#
# supress anoysing logging messages parameter show_log = False
# https://github.com/PaddlePaddle/PaddleOCR/issues/2348
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False) # need to run only once to download and load model into memory

import numpy as np

import cv2

X_resize=220
Y_resize=70


from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe


def GetPaddleOcr(img):

    """
    Created on Tue Mar  7 10:31:09 2023
    
    @author: https://pypi.org/project/paddleocr/ (adapted from)
    """
    
    cv2.imwrite("gray.jpg",img)
    img_path = 'gray.jpg'
    
    
    result = ocr.ocr(img_path,  cls=True)
    # draw result
    #from PIL import Image
    result = result[0]
    #image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    licensePlate= ""
    accuracy=0.0
    #print("RESULTADO  "+ str(txts))
    #print("confiabilidad  "+ str(scores))
    if len(txts) > 0:
        licensePlate= txts[0]
        accuracy=float(scores[0])
    #print(licensePlate)
    #print(accuracy)
      
    return licensePlate, accuracy
    
   
#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0  
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
        print(License + " rotate "+ str(rotation))
        gray=imutils.rotate(gray,angle=rotation)
   
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    
    X_resize=x_resize
    Y_resize=y_resize
    #print("gray.shape " + str(gray.shape)) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    AccuraccyMin=0.7
    TotHits=0
  
    # https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d
    #kernel = np.array([[0, -1, 0],
    #               [-1, 5, -1],
    #               [0, -1, 0]])
   
    
    kernel = np.array([[0, -1, 0],
                   [-1,10, -1],
                   [0, -1, 0]])
    dst = cv2.filter2D(gray, -1, kernel)
    img_concat = cv2.hconcat([gray, dst])
    text, Accuraccy = GetPaddleOcr(img_concat)
    if Accuraccy < AccuraccyMin:
        text=""
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              #print(text + "  Hit with image concat  ")
              TotHits=TotHits+1
           #else:
               #print(License + " detected with Filter image concat "+ text) 
    
    
    for z in range(5,6):
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
       text, Accuraccy = GetPaddleOcr(gray1)
       if Accuraccy < AccuraccyMin:
           text=""
       
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              #print(text +  "  Hit with Sharpen filter z= "  +str(z))
              TotHits=TotHits+1
           #else:
           #    pp=0
               #print(License + " detected with Sharpen filter z= "  +str(z) + " as "+ text) 
      
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text, Accuraccy = GetPaddleOcr(gray1)
    if Accuraccy < AccuraccyMin:
        text=""
   
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
        if text==License:
            #print(text + "  Hit with Stable and THRESH_TRUNC" )
            TotHits=TotHits+1
        #else:
            #print(License + " detected with Stable and THRESH_TRUNC as "+ text)
            
     
    
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
     
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text, Accuraccy = GetPaddleOcr(gray1)
    if Accuraccy < AccuraccyMin:
        text=""
   
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           #print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        #else:
            #print(License + " detected with Brightness and THRESH_TOZERO as "+ text)
    
    
    ################################################################
    return TabLicensesFounded, ContLicensesFounded


def Detect_International_LicensePlate(Text):
    if len(Text) < 3 : return -1
    for i in range(len(Text)):
        if (Text[i] >= "0" and Text[i] <= "9" )   or (Text[i] >= "A" and Text[i] <= "Z" ):
            continue
        else: 
          return -1 
       
    return 1

def ProcessText(text):
  
    if len(text)  > 10:
        text=text[len(text)-10]
        if len(text)  > 9:
          text=text[len(text)-9]
        else:
            if len(text)  > 8:
              text=text[len(text)-8]
            else:
        
                if len(text)  > 7:
                   text=text[len(text)-7:] 
    if Detect_International_LicensePlate(text)== -1: 
       return ""
    else:
       return text

def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectLicenseWithYolov8 (img):
  
    TabcropLicense=[]
    y=[]
    yMax=[]
    x=[]
    xMax=[]
    results = model.predict(img, verbose=False)
    for i in range(len(results)):
        # may be several plates in a frame
        result=results[i]
        
        xyxy= result.boxes.xyxy.numpy()
        confidence= result.boxes.conf.numpy()
        class_id= result.boxes.cls.numpy().astype(int)
        # Get Class name
        class_name = [class_list[z] for z in class_id]
        # Pack together for easy use
        sum_output = list(zip(class_name, confidence,xyxy))
        # Copy image, in case that we need original image for something
        out_image = img.copy()
        for run_output in sum_output :
            # Unpack
            #print(class_name)
            label, con, box = run_output
            if label == "vehicle":continue
            cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            #cv2.imshow("Crop", cropLicense)
            #cv2.waitKey(0)
            TabcropLicense.append(cropLicense)
            y.append(int(box[1]))
            yMax.append(int(box[3]))
            x.append(int(box[0]))
            xMax.append(int(box[2]))
        
    return TabcropLicense, y,yMax,x,xMax


###########################################################
# MAIN
##########################################################
def ReadLicencePlateImage(image):
            TabLicensesmax=[]
            ContLicensesmax=[]
            TimeIniLicensesmax=[]
            TimeEndLicensesmax=[]


    
            gray=image
                       
            License="License"
            TabImgSelect, y, yMax, x, xMax =DetectLicenseWithYolov8(gray)
            
            if TabImgSelect==[]:
                #print(License + " NON DETECTED")
                
                return ""
            
                
            for i in range(len(TabImgSelect)):
                gray=TabImgSelect[i]  
                
           
                #cv2.waitKey(0)
                x_off=3
                y_off=2
                
                x_resize=220
                y_resize=70
                
                Resize_xfactor=1.78
                Resize_yfactor=1.78
                
                ContLoop=0
                
                SwFounded=0
                
                BilateralOption=0
                TabLicensesFounded=[]
                ContLicensesFounded=[]
                
                TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                       Resize_xfactor, Resize_yfactor, BilateralOption)
                  
                
                                
                ymax=-1
                contmax=0
                licensemax=""
              
                for z in range(len(TabLicensesFounded)):
                    if ContLicensesFounded[z] > contmax:
                        contmax=ContLicensesFounded[z]
                        licensemax=TabLicensesFounded[z]
                return licensemax
 
# MAIN
#image=cv2.imread("Test1\\6662GKS.jpg")
#print(ReadLicencePlateImage(image))
          
