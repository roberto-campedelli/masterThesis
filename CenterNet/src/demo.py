from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

#initialize list for storing TP, FP and FN for each category
num_cat = 10
true_positive = [0] * (num_cat +1) 
false_negative = [0] * (num_cat +1)
false_positive = [0] * (num_cat +1)
#initialize to 0 every counter, i don't use the 0 index

def getDetAnnotation(res):
  cat = -1
  for i in res:
    if res[i].size != 0:
      cat = i
      break
  if cat < 0 : 
    print("category not found")
    return

  info = res[cat]
  
  alpha = info[0,0]
  topLeft_x = info[0,1]
  topLeft_y = info[0,2]
  topRight_x = info[0,3]
  bottomLeft_y = info[0,4]
  height_m = info[0,5]
  width_m = info[0,6]
  lentgh_m = info[0,7]
  center_x = info[0,8]
  center_y = info[0,9]
  center_z = info[0,10]
  rot_y = info[0,11]
  
  box2d = [topLeft_x, topLeft_y, topRight_x - topLeft_x, bottomLeft_y - topLeft_y]
  center = [center_x, center_y, center_z]
  dim = [height_m, width_m, lentgh_m]
  det_annotations = [cat, dim, box2d, alpha, center, rot_y]
  return det_annotations

def findBetween(line, start_word, end_word):
  index_start = line.find(start_word) + len(start_word)
  index_end = line.find(end_word)
  return line[index_start:index_end].strip()[:-1]

def getGtAnnotation(filePath, image_id):
  line_to_find = "\"image_id\": " + str(image_id)
  with open(filePath) as f:
    lines = f.readlines()
  for line in lines:
    if line_to_find in line:
      annotation = line

  cat_id = int(findBetween(annotation, "\"category_id\":", "\"dim\":"))
  dim = list(map(float, findBetween(annotation, "\"dim\":", "\"bbox\":")[1:-1].split(", ")))
  bbox = list(map(float, findBetween(annotation, "\"bbox\":", "\"depth\":")[1:-1].split(", ")))
  alpha = float(findBetween(annotation, "\"alpha\":", "\"truncated\":"))
  location = list(map(float,findBetween(annotation, "\"location\":", "\"rotation_y\":")[1:-1].split(", ")))
  rotation_y = float(findBetween(annotation, "\"rotation_y\":", "},"))

  gt_annotations = [cat_id, dim, bbox, alpha, location, rotation_y]
  return gt_annotations

#calculate statistics between ground truth and detection
#annotation order = [cat_id, dim, bbox, alpha, location, rotation_y]
def getStatistics(gt_annotation, det_annotation):
  gt_cat_id = gt_annotation[0]
  gt_dim = gt_annotation[1]
  gt_bbox = gt_annotation[2]
  gt_location = gt_annotation[4]
  gt_rot_y = gt_annotation[5]

  det_cat_id = det_annotation[0]
  det_dim = det_annotation[1]
  det_bbox = det_annotation[2]
  det_location = det_annotation[4]
  det_rot_y = det_annotation[5]

  #check if the classification is correct
  if gt_cat_id == det_cat_id:
    true_positive[gt_cat_id] = true_positive[gt_cat_id] + 1
  if gt_cat_id != det_cat_id:
    false_negative[gt_cat_id] = false_negative[gt_cat_id] + 1
    false_positive[det_cat_id] = false_positive[det_cat_id] + 1
  
  #check the difference between the volumes
  det_vol = det_dim[0]*det_dim[1]*det_dim[2]
  gt_vol = gt_dim[0]*gt_dim[1]*gt_dim[2]
  vol_err = (abs(det_vol - gt_vol))/ float(gt_vol)

  #check IoU of the front face of the 3dbbox
  #TODO

  #check the location of the center of the 3dbbox error using the euclidean distance
  location_err = np.linalg.norm(np.array(gt_location) - np.array(det_location))

  #check the error of the rotation 
  rot_err = abs(gt_rot_y - det_rot_y)/float(3.1415926)

  return vol_err, location_err, rot_err
 

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]
      ret = detector.run(image_name)
      res = ret['results']
      gt_annotation = getGtAnnotation("../data/my_dataset/annotations/annotations_3dop_train.json", image_id)
      det_annotation = getDetAnnotation(res)
      print("gt_annotations = " + str(gt_annotation))
      print("det_annotations = " +str(det_annotation))
      #print(res)
      vol_err, loc_err, rot_err = getStatistics(gt_annotation, det_annotation)
      print("Volume error = " + str(vol_err))
      print("Location error = " + str(loc_err))
      print("Rotation error = "+ str(rot_err))

      print("TP = " + str(true_positive))
      print("FP = " + str(false_positive))
      print("FN = " + str(false_negative))


      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
