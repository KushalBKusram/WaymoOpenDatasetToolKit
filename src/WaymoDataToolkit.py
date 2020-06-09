import os
import cv2
import math
import subprocess
import numpy as np
import itertools
from urllib.parse import urlparse
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoDataToolkit:

  # Variables required
  url = ""
  count = 0
  d = dict()

  def __init__(self, url):
    self.url = url
  
  def process_frames(self, frame, camera_labels):
      # Iterate over the individual labels.
      for camera_labels in frame.camera_labels:
        for label in camera_labels.labels:
          if(label.type==0):
              self.d["type_unknown"] += 1
          if(label.type==1):
              self.d["type_vehicle"] += 1
          if(label.type==2):
            self.d["type_pedestrian"] += 1
          if(label.type==3):
            self.d["type_sign"] += 1
          if(label.type==4):
            self.d["type_cyclist"] += 1
          if(label.detection_difficulty_level==0):
            self.d["label_unknown"] += 1
          if(label.detection_difficulty_level==1):
            self.d["label_level_1"] += 1
          if(label.detection_difficulty_level==2):
            self.d["label_level_2"] += 1

  def extract_image(self, camera_image):
    self.decodedImage = tf.io.decode_jpeg(camera_image.image, channels = 3, dct_method='INTEGER_ACCURATE')
    self.decodedImage = cv2.cvtColor(self.decodedImage.numpy(), cv2.COLOR_RGB2BGR)
    self.height = self.decodedImage.shape[0]
    self.width = self.decodedImage.shape[1]

  def extract_labels(self, camera_image, frame):
    self.labels = []
    for camera_labels in frame.camera_labels:
      if camera_labels.name != camera_image.name:
        continue
      for label in camera_labels.labels:
        self.labels.append((label.type, (label.box.center_x)/self.width, (label.box.center_y)/self.height, label.box.length/self.width, label.box.width/self.height))  

  def assign_camera_labels(self, camera_code):
    if camera_code == 1:
      self.front = self.decodedImage
      self.front_labels = self.labels
    if camera_code == 2:
      self.front_left = self.decodedImage
      self.front_left_labels = self.labels
    if camera_code == 3:
      self.front_right = self.decodedImage
      self.front_right_labels = self.labels
    if camera_code == 4:
      self.side_left = self.decodedImage
      self.side_left_labels = self.labels
    if camera_code == 5:
      self.side_right = self.decodedImage
      self.side_right_labels = self.labels

  def write_data(self, labels, image, camera_name):
    f = open('../data/train/labels/{}_{}.txt'.format(self.count, camera_name), 'w')
    for tuple in labels:
      line = ' '.join(str(x) for x in tuple)
      f.write(line + '\n')
    f.close()
    cv2.imwrite('../data/train/images/{}_{}.png'.format(self.count, camera_name), image)
  
  def dataExtractor(self):

    frame = open_dataset.Frame()

    # Convert tfrecord to a list
    datasetAsList = list(self.dataset.as_numpy_iterator())
    self.totalFrames = len(datasetAsList)

    # Convert to bytearray
    for frameIdx in range(self.totalFrames):
      frame.ParseFromString(datasetAsList[frameIdx])
      self.d = {"type_unknown": 0,
                "type_vehicle": 0,
                "type_pedestrian": 0,
                "type_sign": 0,
                "type_cyclist": 0,
                "label_unknown": 0,
                "label_level_1": 0,
                "label_level_2": 0}
      self.process_frames(frame, frame.camera_labels)
      for index, image in enumerate(frame.images):
        self.extract_image(image)
        self.extract_labels(image, frame)
        self.assign_camera_labels(image.name)
      
      # Write data to folder
      self.write_data(self.front_labels, self.front, 'FRONT')
      self.write_data(self.front_left_labels, self.front_left, 'FRONT_LEFT')
      self.write_data(self.front_right_labels, self.front_right, 'FRONT_RIGHT')
      self.write_data(self.side_left_labels, self.side_left, 'SIDE_LEFT')
      self.write_data(self.side_right_labels, self.side_right, 'SIDE_RIGHT')

      self.count += 1

  def dataRetriever(self):
      cmd = 'gsutil'
      filename = urlparse(self.url)
      filename = os.path.basename(filename.path)
      tempFile = '../data/{}'.format(filename) 
      FNULL = open(os.devnull, 'w')
      print('Retrieving {} and assigning as a dataset..'.format(filename))
      x = subprocess.call([cmd, "cp", self.url, tempFile], stdout=FNULL, stderr=subprocess.STDOUT)
      self.dataset = tf.data.TFRecordDataset(tempFile, compression_type='')
      print('Assigned as a dataset')
  



