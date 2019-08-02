# coding: utf-8

# ****************************************************************************
# * @file: extratc.py
# * @project: ROBOFEI-HT - FEI 😛
# * @author: Vinicius Nicassio Ferreira
# * @version: V0.0.1
# * @created: 23/10/2017
# * @e-mail: vinicius.nicassio@gmail.com
# * @brief: Class extratc
# ****************************************************************************

# ---- Imports ----

# Libraries to be used
import time
import sys
import os
os.chdir('../../')
sys.path.append('./include')
sys.path.append('./src')

# Used class developed by RoboFEI-HT.
from DNN import * # Class that implements object detection using a deep neural network (DNN).

class Argumentos:
    dnn = True
    train = True

a = Argumentos()

dnn = DNN(a)

dicionario = {}
os.system("mkdir ./Train/DNN/imagens\ to\ classify")

try:
    imagens = [i.rsplit(".", 1)[0] for i in os.listdir("./Train/DNN/imagens to classify")]
    xmls = [i.rsplit(".", 1)[0] for i in os.listdir("./Train/DNN/annotations DNN")]
    imagens = [i for i in imagens if i not in xmls]
    for image in imagens:
        dicionario["frame"] = cv2.imread("./Train/DNN/imagens to classify/"+image+".jpg")
        dicionario["time"] = time.time()
        dnn.detect(dicionario)
        os.system("rm ./Train/DNN/imagens\ to\ classify/"+image.replace(":","\\:").replace(" ","\\ ")+".jpg")
    cv2.destroyAllWindows()
except:
    pass

for video in os.listdir("./Train/DNN/Videos"):
    cap = cv2.VideoCapture("./Train/DNN/Videos/"+video)
    while True:
        for __ in xrange(0, 4):
            __, dicionario["frame"] = cap.read()
        if dicionario["frame"] is None:
            break
        dicionario["time"] = time.time()
        dnn.detect(dicionario)
    os.system("rm ./Train/DNN/Videos/"+video)
