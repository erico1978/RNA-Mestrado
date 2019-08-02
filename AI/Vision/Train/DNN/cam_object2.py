import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util


# from imutils.video import WebcamVideoStream
from camvideostream import WebcamVideoStream
# from imutils.video import FPS
import argparse
import imutils
import cv2

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/fei/RoboFEI-HT-VisionDebug/AI/Vision/Data/rede/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/fei/RoboFEI-HT-VisionDebug/AI/Vision/Data/rede/object-detection.pbtxt')

NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#def load_image_into_numpy_array(image):
#  (im_width, im_height) = image.size
#  return np.array(image.getdata()).reshape(
#      (im_height, im_width, 3)).astype(np.uint8)


vs = WebcamVideoStream(src=0).start()
#fps = FPS().start()

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    while 1:

      frame = vs.read()
#      frame = imutils.resize(frame, width=400)


      image_np = np.asarray(frame)


#      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
#      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)

      df = pd.DataFrame()
      df['classes'] = classes[0]
      df['scores'] = scores[0]
      df['boxes'] = boxes[0].tolist()

      height, width = frame.shape[:2]
      print df['boxes'][0][0]
#      print df.head()

#      box_coords[ymin, xmin, ymax, xmax]
      y1 = int(df['boxes'][0][0]*height)
      x1 = int(df['boxes'][0][1]*width)
      y2 = int(df['boxes'][0][2]*height)
      x2 = int(df['boxes'][0][3]*width)

      # check to see if the frame should be displayed to our screen
      cv2.imshow("Frame", image_np)
      key = cv2.waitKey(1) & 0xFF

      if(df['scores'][0]>0.95):
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
      cv2.imshow("Frame2", frame)
      key = cv2.waitKey(1) & 0xFF

      # update the FPS counter
#      fps.update()


#stop the timer and display FPS information
#fps.stop()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
