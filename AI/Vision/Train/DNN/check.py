# coding: utf-8

# ****************************************************************************
# * @file: check.py
# * @project: ROBOFEI-HT - FEI 😛
# * @author: Vinicius Nicassio Ferreira
# * @version: V0.0.1
# * @created: 23/10/2017
# * @e-mail: vinicius.nicassio@gmail.com
# * @brief: Class check
# ****************************************************************************

# ---- Imports ----

# Libraries to be used
import random
import time
import sys
import os
sys.path.append('./include')
sys.path.append('./src')

# Abrindo programa de marcação
os.system("clear")
os.system("screen -d -m -S test python ~/labelImg/labelImg.py $(pwd)/imagens to classify/ $(pwd)/annotations DNN/")
key = "S"
while key.upper() == "S":
    # Selecionando arquivos
    all = os.listdir("./annotations DNN")
    try:
        random.shuffle(all)
    except:
        pass
    xmls = [i.rsplit(".", 1)[0] for i in all][:200]
    if xmls == []:
        print "Nenhuma imagens pré-classificada pela DNN"
        time.sleep(1)
        all = os.listdir("./imagens to classify")
        try:
            random.shuffle(all)
        except:
            pass
        xmls = [i.rsplit(".", 1)[0] for i in all][:200]
        if xmls == []:
            print "Não tem mais imagens para serem marcadas"
            time.sleep(1)
            break

    # Organizando arquivos nas pastas
    os.system("mkdir ./imagens\ to\ check")
    for files in xmls:
        os.system("mv ./imagens\ to\ classify/"+files.replace(":", "\\:").replace(" ", "\\ ")+".jpg ./imagens\ to\ check")

    # Mensagem
    os.system("clear")
    print "Imagens separadas"
    time.sleep(5)

    os.system("clear")
    key = raw_input("As marcações foram finalizadas ? [S/N]: ")
    if key.upper() == "N":
        os.system("mv ./imagens\ to\ check/* ./imagens\ to\ classify")
        break

    # Finalizando marcações e separando classificações
    os.system("mkdir ./annotations")
    os.system("mkdir ./imagensTrain")
    os.system("mv ./imagens\ to\ check/* ./imagensTrain")
    for files in xmls:
        os.system("mv ./annotations\ DNN/"+files.replace(":", "\\:").replace(" ", "\\ ")+".xml ./annotations")
        os.system('sed -i "s/imagens to check/imagensTrain/g" ./annotations/'+files.replace(":", "\\:").replace(" ", "\\ ")+".xml")

    os.system("clear")
    key = raw_input("Deseja checar/marcar mais 10 imagens ? [S/N]: ")

os.system("rm -R ./imagens\ to\ check")
