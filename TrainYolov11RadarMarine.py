
# https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85
# modified by Alfonso Blanco García
import os

import ultralytics
# at 02/11/2024 there is an incompatibility with numpy is neccesary down numpy to 1.23 version
# https://stackoverflow.com/questions/34051737/numpy-core-multiarray-failed-to-import
# pip install numpy==1.23
#########################################################
# for mi from conda environment yolov11
#(yolov111) C:\Users\Alfonso Blanco\.conda\envs\yolov11\Scripts>python pip-script.py install numpy==1.23

from ultralytics import YOLO
import torch

class ObjectDetection:
    def __init__(self):
        
        self.dir="C:/Radar_Marine-Yolov11"
    def train(self, runName):
      
        model = YOLO("yolo11n.pt")
        
        #Dataset downloaded from https://universe.roboflow.com/ron-zhyan/solar-panel-anomalies-hikbk-0joqn/dataset/1
        yaml_path = "data.yaml"
        results = model.train(
            data= yaml_path,         # Path to your dataset config file
            batch = 16,               # Training batch size
            #imgsz= 640,                   # Input image size
            #epochs= 2000,                  # Number of training epochs
            epochs= 20,                  # Number of training epochs
            optimizer= 'SGD',             # Optimizer, can be 'Adam', 'SGD', etc.
            lr0= 0.0005,                    # Initial learning rate
            
            lrf= 0.1,                     # Final learning rate factor
            
            weight_decay= 0.0005,         # Weight decay for regularization
            momentum= 0.937,              # Momentum (SGD-specific)
            verbose= True,                # Verbose output
            #device= '0',                  # GPU device index or 'cpu'
            device= 'cpu',                  # GPU device index or 'cpu'
            workers= 8,                   # Number of workers for data loading
            project= 'runs/train',        # Output directory for results
            name= 'exp',                  # Experiment name
            exist_ok= False,              # Overwrite existing project/name directory
            rect= False,                  # Use rectangular training (speed optimization)
            resume= False,                # Resume training from the last checkpoint
            #multi_scale= False,           # Use multi-scale training
            multi_scale= True,
            #single_cls= False,             # Treat data as single-class
            single_cls= False, 
            #freeze = 20, #default değer : none
            #resume=True, #Başka bilgisatarda eğitim devamı yapılamıyor.
            #name=runName,
        )

    @staticmethod
    def train_custom_dataset(runName):
        od = ObjectDetection()
        od.train(runName)


# Example usage:
ObjectDetection.train_custom_dataset('trained_model')
