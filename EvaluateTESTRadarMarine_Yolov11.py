# -*- coding: utf-8 -*-
"""
Created on Jun 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
# dataset https://universe.roboflow.com/ron-zhyan/solar-panel-anomalies-hikbk-0joqn/dataset/1
dir=""

dirname= "test\\images"
dirnameLabels="test\\labels"


#dirnameYolo="runs\\train\\exp5\\weights\\last.pt"

dirnameYolo="last3epoch-LR0005-216Detections.pt"



import cv2
import time
Ini=time.time()

#from ultralytics import YOLOv10
from ultralytics import YOLO

#model = YOLOv10(dirnameYolo)
model = YOLO(dirnameYolo)

class_list = model.model.names
print(class_list)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

########################################################################
def loadimages(dirname):
 
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 
     imgpath = dirnameLabels + "\\"
     
     TabLabels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 TabLinxyxy=[]
                 Labels=[]
                 for linea in f:
                      
                      indexFracture=int(linea[0])
                      Label=class_list[indexFracture]
                      xyxy=linea[2:]
                      TabLinxyxy.append(xyxy)
                      Labels.append(Label)                      
                 
                 
                 if Label=="":
                      ContNoLabels+=1
                 else:
                     ContLabels+=1
                     
                 TabLabels.append(Labels)
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append( TabLinxyxy)
     return TabLabels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

def unconvert (width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, ymin, xmax, ymax

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def RadarMarineYolov10 (img):
  
   TabcropPanelSolarAnomaly=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   Tabclass_cod=[]
   Tabconfidence=[]
   
   # https://blog.roboflow.com/yolov10-how-to-train/
   results = model(source=img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       #print(class_id)
       out_image = img.copy()
       for j in range(len(class_id)):
           con=confidence[j]
           Tabconfidence.append(con)
           label=class_list[class_id[j]] + " " + str(con)
           box=xyxy[j]
           
           cropPanelSolarAnomaly=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           TabcropPanelSolarAnomaly.append(cropPanelSolarAnomaly)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))

           # 
           Tabclass_name.append(label)
           Tabclass_cod.append(class_id[j])
            
      
   return Tabconfidence, TabcropPanelSolarAnomaly, y,yMax,x,xMax, Tabclass_name, Tabclass_cod

def plot_image(image, boxes, boxesTrue, imageCV, NameImage):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    #class_labels = PASCAL_CLASSES
    class_labels=class_list
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    fig.suptitle(NameImage)
    ax.imshow(im)   

    # Create a Rectangle patch
    Cont=0
    print(boxes)
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf=box[1]
        conf=str(conf)
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)] + " conf: " + str(conf[:3]),
            color="red",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
      
        

        
        Cont+=1
        #if Cont > 1: break # only the most predicted box
        #break

    for i in range(len(boxesTrue)):
         boxTrue=boxesTrue[i].split(" ")
         upper_left_x_True = float(boxTrue[0]) - float( boxTrue[2] )/ 2.0
         upper_left_y_True = float(boxTrue[1]) - float( boxTrue[3]) / 2.0
         rect = patches.Rectangle(
                 (upper_left_x_True * width, upper_left_y_True * height),
                 float(boxTrue[2]) * width,
                 float(boxTrue[3]) * height,
                 linewidth=2,
                 edgecolor="green",
                 facecolor="none",
        )
        # Add the patch to the Axes
       
         ax.add_patch(rect)
      # rect with true fracture
    
    plt.show()

###########################################################
# MAIN
##########################################################

TabLabels, TabFileLabelsName, TabxyxyTrue, ContLabels, ContNoLabels= loadlabels(dirnameLabels)

print("Number of images to test : " + str(len(TabLabels)))

print("Number of files without labels : " + str(ContNoLabels))
print("Number of files with labels : " + str(ContLabels))


imagesComplete, TabFileName=loadimages(dirname)

print("Number of images to test: " + str(len(imagesComplete)))

#ContError=0
#ContHit=0
ContDetected=0
ContNoDetected=0
  
for i in range (len(imagesComplete)):
 
            if TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] != TabFileName[i][:len(TabFileName[i])-4]:
                 print("ERROR SEQUENCING IMAGES AN LABELS " + TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] +" --" + TabFileName[i][:len(TabFileName[i])-4])
                 break
            # no se consideran las que no vienen labeladas
            if len(TabLabels[i]) == 0: continue
            gray=imagesComplete[i]
           
            imgTrue=imagesComplete[i]

            TabLinxyxyTrue=TabxyxyTrue[i]
            
            for z in range (len(TabLinxyxyTrue)):
                 
                 XcenterYcenterWH=TabLinxyxyTrue[z].split(" ")
                 width=float(imgTrue.shape[0])
                 height=float(imgTrue.shape[1])
                 x=float(XcenterYcenterWH[0])
                 y=float(XcenterYcenterWH[1])
                 w=float(XcenterYcenterWH[2])
                 h=float(XcenterYcenterWH[3])
                 xTrue,yTrue,xMaxTrue,yMaxTrue=unconvert(width, height, x, y, w, h)
           
                 start_pointTrue=(int(xTrue),int(yTrue)) 
                 end_pointTrue=(int(xMaxTrue),int( yMaxTrue))
           
                 colorTrue=(0,0,255)
            
                 # Using cv2.rectangle() method
                 # Draw a rectangle with green line borders of thickness of 2 px
                 imgTrue = cv2.rectangle(imgTrue, start_pointTrue, end_pointTrue,(0,255,0), 2)
           
                 # Put text
                 text_locationTrue = (int(xMaxTrue),int(yMaxTrue))
                 text_colorTrue = (255,255,255)
                 #cv2.putText(imgTrue, Labels[i] ,text_locationTrue
                 #            , cv2.FONT_HERSHEY_SIMPLEX , 1
                 #            , text_colorTrue, 2 ,cv2.LINE_AA)
                 cv2.putText(imgTrue, "" ,text_locationTrue
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_colorTrue, 2 ,cv2.LINE_AA)

            #cv2.imshow('True', imgTrue)
            #cv2.waitKey(0)

            
            Tabconfidence, TabImgSelect, y, yMax, x, xMax, Tabclass_name, Tabclass_cod =RadarMarineYolov10(gray)
            Tabnms_boxes=[]
            #print(gray.shape)
            if TabImgSelect==[]:
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1
                img=imgTrue
                #continue
            else:
                #ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
                
            Labels = TabLabels[i]  
            for z in range(len(TabImgSelect)):
                #if TabImgSelect[z] == []: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                color=(255,0,0)
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                img = cv2.rectangle(gray, start_point, end_point,(255,0,0), 2)
                # Put text
                text_location = (x[z], y[z])
                print("Target detected in coordinates x = " +str(x[z]) + " y = " + str(y[z]))
                ContDetected=ContDetected+1
                text_color = (255,255,255)
                """
                if Tabclass_name[z][:len(Labels[z])] !=Labels[z]:
                   
                     #print(len(Tabclass_name[z]))
                     #print(len(Labels[i]))
                     print("ERROR " + TabFileName[i] + "Predicted "+ Tabclass_name[z] + " true is " + Labels[z])
                     ContError+=1
                else:
                     #print("HIT " + TabFileName[i] + "Predicted "+ Tabclass_name[z] )
                     ContHit+=1
                     
                cv2.putText(img, str(Tabclass_name[z][len(Labels[z]):]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 1 ,cv2.LINE_AA)
                cv2.putText(gray1, str(Tabclass_name[z][len(Labels[z]):]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 1 ,cv2.LINE_AA)

                """
                cv2.putText(img, str(Tabclass_name[z]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 1 ,cv2.LINE_AA)
                cv2.putText(gray1, str(Tabclass_name[z]) ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , 1
                        , text_color, 1 ,cv2.LINE_AA)
                        
                        
                #cv2.imshow('Bone Fracture', gray1)
                #cv2.waitKey(0)
                #break
                #      
                #show_image=cv2.resize(img,(1000,700))
                #cv2.imshow('Frame', show_image)
                #img=cv2.resize(img,(240,240))
                #cv2.imshow('Frame', img)
                #cv2.waitKey(0)
                nms_boxes=[]
                nms_boxes.append(Tabconfidence[z])
                #nms_boxes.append(Tabclass_name[z])
                nms_boxes.append(Tabclass_cod[z])

                
                X=x[z] + (xMax[z] - x[z])/ 2
                Y = y[z] + (yMax[z] -y[z] ) / 2
                W=xMax[z] - x[z]
                H=yMax[z] -y[z]
                nms_boxes.append(X)
                nms_boxes.append(Y)
                nms_boxes.append(W)
                nms_boxes.append(H)
                Tabnms_boxes.append(nms_boxes)
                #plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, boxesTrue, imagesCV[i])
            plot_image(img, Tabnms_boxes, TabxyxyTrue[i], img, TabFileName[i])
                
             
              
print("")           
print("Images no detected=" + str(ContNoDetected))
print("Total targets detected=" + str(ContDetected))

print("")      
print( " Time in seconds "+ str(time.time()-Ini))
