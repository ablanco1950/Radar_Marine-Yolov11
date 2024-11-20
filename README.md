# Radar_Marine-Yolov11
Essay to detect targets in images obtained by radar. Use the images provided by https://github.com/kz258852/dataset_M_Radar and use yolov11 to create a model for detecting true targets in the images

Installation.

To avoid incompatibilities with other versions of yolo, install ultralytics in a new environment (you can easily create a new environment if you have the Anaconda development environment). Once the new environment is created, run in it:

pip install ultralytics

There may be incompatibilities with the installed version of numpy, so you should downgrade it;

pip install numpy==1.23

(a requirements.txt is attached)

Download all the project files to disk, they will be unzipped into a folder named Radar_Marine-Yolov11

Unzip the folder containing the test images test.zip

Perform an evaluation of the model by running:

EvaluateTESTRAdarMarine_Yolov11.py

which acts on a test of 50 images.

The images need to be closed in order to continue with the next one appear in succession.

In the images, small green squares appear that indicate the labeled targets, above them a sign appears with “ Detect” and a number indicating the probability with which they have been detected. If a red square appears with the label “detect”, it indicates a false positive, a target has been detected that has not been labeled as such and the number does not appear. corresponding green box that overlaps the red box.

https://github.com/ablanco1950/Radar_Marine-Yolov11/blob/main/Figure_1.png

![Fig1](https://github.com/ablanco1950/Radar_Marine-Yolov11/blob/main/Figure_1.png)

The coordinates of the detected targets appear on the console (in the image).

The evaluation was carried out with an optimized model last3epoch-LR0005-216Detections.pt

The following steps were followed to obtain the model:

1- Download the project https:/ /github.com/kz258852/dataset_M_Radar to disk and unzip the zip.

You get a folder named dataset_M_Radar-main in which the Annotations and Images folders are placed.

2- In the Annotations folder the labels come in xml format instead of in .txt format

To convert .xml files to .txt, copy the attached convert program to the Annotations subfolder and run it in that subfolder.

python convert.py

For each .xml file, the corresponding .txt appears

the convert.py program was obtained from https://github.com/tanjil072/xmlTotxt-converter-for-YOLO/blob/main/convert.py

modified due to that the .xml files to be processed did not include information about the height and width of the image to be processed

3- Next, you must create a folder structure in the project directory in the form:

train
  images
  labels

Copy from dataset_M_Radar-main\Images to the images subfolder the first 2500 images
In dataset_M_Radar-main\Annotations select all the .txt files and copy the first 2500 .txt files to the labels subfolder

4- Next you have to create a folder structure in the project directory in which form:

valid
  images
  labels

Copy from the dataset_M_Radar-main\Images the images from 002501.jpg to 002950.jpg to the images subfolder

In the file dataset_M_Radar-main\Annotations select all the .txt files and copy to the labels subfolder from file 2501. txt to 2950.txt

5- In the data.yaml file, in the first 3 lines, there are the absolute addresses of the train, valid and test folders, assuming that the project folder is in the c: directory, if not, these absolute addresses need to be changed.

6- Perform the training

python TrainYolov11RadarMarine.py

The models obtained with each epoch are stored in the address:
Radar_Marine-Yolov11\runs\train\exp\weights

The best results are achieved in the first epoch, there being no relationship between the mAP and the results real.

References:

https://github.com/kz258852/dataset_M_Radar

https://github.com/tanjil072/xmlTotxt-converter-for-YOLO/blob/main/convert.py


