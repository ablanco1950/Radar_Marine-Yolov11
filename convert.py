# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob
import cv2

lut={}
lut["D00"] =0
lut["D10"] =1
lut["D20"] =2
lut["D40"] =3


def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo( lut ):

    for fname in glob.glob("*.xml"):
        
        xmldoc = minidom.parse(fname)
        
        fname_out = (fname[:-4]+'.txt')
        fname_jpg=("..//Images//" + fname[:-4]+'.jpg') # MOD

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)
           

            ###Modified as width and height comes as 0
            img= cv2.imread(fname_jpg) # MOD
            if  None:  # MOD
                print("ERROR Reading Image " + fname_jpg) #MOD
            width=img.shape[0] # MOD
            height=img.shape[1] # MOD
           

            for item in itemlist:
                # get class label
                #classid =  (item.getElementsByTagName('name')[0]).firstChild.data # MOD
                classid=0 # MOD
                #if classid in lut: # MOD
                #    label_str = str(lut[classid]) # MOD
                #else: # MOD
                #    label_str = "-1" # MOD
                #    print ("warning: label '%s' not in look-up table" % classid) # MOD
                label_str="0"

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print ("wrote %s" % fname_out)



def main():
    convert_xml2yolo( lut )


if __name__ == '__main__':
    main()
