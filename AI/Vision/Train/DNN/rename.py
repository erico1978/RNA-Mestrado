# coding: utf-8

import time
import sys
import os
os.chdir('../../')
sys.path.append('./include')
sys.path.append('./src')

image = 1
names = [i.rsplit(".", 1)[0].replace(" ", "\ ") for i in os.listdir("./Train/DNN/imagensTrain")]

digits = 0 
size = len(names) 
while size > 0: 
    size /= 10 
    digits += 1

for name in names:
#     print name
    sub = "sed -i 's/"+ name.replace("\ ", " ") +"/a"+ str(image).zfill(digits) +"/g' ./Train/DNN/annotations/"+ name +".xml"
#     print sub
    os.system(sub)
    jpg = "mv ./Train/DNN/imagensTrain/"+ name +".jpg ./Train/DNN/imagensTrain/a"+ str(image).zfill(digits) +".jpg"
#     print jpg
    os.system(jpg)
    xml = "mv ./Train/DNN/annotations/"+ name +".xml ./Train/DNN/annotations/a"+ str(image).zfill(digits) +".xml"
#     print xml
    os.system(xml)
#     raw_input()
    image +=1

image = 1
names = [i.rsplit(".", 1)[0].replace(" ", "\ ") for i in os.listdir("./Train/DNN/imagensTrain")]

digits = 0 
size = len(names) 
while size > 0: 
    size /= 10 
    digits += 1

for name in names:
    sub = "sed -i 's/"+ name.replace("\ ", " ") +"/image"+ str(image).zfill(digits) +"/g' ./Train/DNN/annotations/"+ name +".xml"
    os.system(sub)
    jpg = "mv ./Train/DNN/imagensTrain/"+ name +".jpg ./Train/DNN/imagensTrain/image"+ str(image).zfill(digits) +".jpg"
    os.system(jpg)
    xml = "mv ./Train/DNN/annotations/"+ name +".xml ./Train/DNN/annotations/image"+ str(image).zfill(digits) +".xml"
    os.system(xml)
    image +=1
