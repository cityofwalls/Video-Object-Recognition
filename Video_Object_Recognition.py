# -*- coding: utf-8 -*-
"""
Object Recognition in Videos using Tensorflow and OpenCV.
"""

import cv2 as cv
import tensorflow as tf
from random import choice

VIDEO_SCRUB = 1
CONFIDENCE_THRESHOLD = 0.2
VIDEO = 'Swan.mp4'
OUTPUT = 'Swan_w_boxes.mp4'

with open('./labels.txt', 'r') as f:
    labels = f.readlines()
labels = list(map(lambda x: x[x.index(' '):], labels))
labels = list(map(lambda x: x.strip(), labels))

def load_trained_model():
  with tf.gfile.FastGFile('./frozen_inference_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  return graph_def

def draw_bounding_box(img, label_name, box_top, box_left, box_bottom, box_right):
  #r, g, b = choice([0, 127, 255]), choice([0, 127, 255]), choice([0, 127, 255])
  r, g, b = 0, 255, 0
  scale = 1.0
  thickness = 2
  cv.rectangle(img, (box_left, box_top), (box_right, box_bottom), (r, g, b), thickness=thickness)
  cv.putText(img, label_name, (box_left, box_top), cv.FONT_HERSHEY_SIMPLEX, scale, [r, g, b], thickness)
  return img

def predict_on_image(img, output, sess):
  height, width, _ = img.shape

  # The model expects a 300x300 image as input
  image_scaled = cv.resize(img, (300, 300))

  detections = sess.run([
    sess.graph.get_tensor_by_name('num_detections:0'),    # [0] in detections
    sess.graph.get_tensor_by_name('detection_scores:0'),  # [1] in detections
    sess.graph.get_tensor_by_name('detection_boxes:0'),   # [2] in detections
    sess.graph.get_tensor_by_name('detection_classes:0'), # [3] in detections
  ], feed_dict={
    'image_tensor:0': image_scaled.reshape(
        1, image_scaled.shape[0], image_scaled.shape[1], 3)
  })


  # num_detections:0
  detection_count = int(detections[0][0])
  print("Found", detection_count, "objects")

  for i in range(detection_count):
    # detection_scores:0
    confidence_score = detections[1][0][i]

    # detection_boxes:0
    box = detections[2][0][i]
    box_top = box[0]
    box_left = box[1]
    box_bottom = box[2]
    box_right = box[3]

    # detection_classes:0
    label_id = int(detections[3][0][i])
    label_name = labels[label_id]

    print("Found label {} (id {}) with a confidence of {}. Bounding box [top: {}, left: {}, bottom: {}, right: {}]".format(
        label_name, label_id, confidence_score, box_top, box_left, box_bottom, box_right))

    if confidence_score > CONFIDENCE_THRESHOLD:
      img = draw_bounding_box(img, label_name, int(box_top*height), int(box_left*width),
                                int(box_bottom*height), int(box_right*width))
  output.write(img)

cap = cv.VideoCapture(VIDEO)

height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv.CAP_PROP_FPS)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

fourcc = cv.VideoWriter_fourcc(*'MP4V')
output = cv.VideoWriter(OUTPUT, fourcc, fps, (width, height))

trained_model = load_trained_model()

with tf.Session() as sess:
  sess.graph.as_default()
  tf.import_graph_def(trained_model, name='')

  for i in range(0, total_frames, VIDEO_SCRUB):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
      raise Exception("Problem reading frame", i, " from video")

    predict_on_image(frame, output, sess)

cap.release()
output.release()
print("\nDONE")
