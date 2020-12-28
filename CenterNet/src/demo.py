from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from my_demo_utils import *

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

#annotation file path
ann_file_path = "../data/my_dataset/annotations/annotations_3dop_train.json"
#ann_file_path = "../data/new_kitti.json"


#initialize list for storing TP, FP and FN for each category
num_cat = 10
true_positive = [0] * (num_cat +1) 
false_negative = [0] * (num_cat +1)
false_positive = [0] * (num_cat +1)
not_detected = 0

#define the lists for storing the errors 
vol_errors = []
iou_errors = []
rot_errors = []
loc_errors = []

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

      #image id for my dataset
      image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]

      #image id for kitti
      #image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]
      #i cut the zero from the beginning of the image name   
      #while(image_id.startswith("0") and len(image_id) > 1):
      #  image_id = image_id[1:]
          
      print("image id =  " + str(image_id))

      ret = detector.run(image_name)
      res = ret['results']
      #for multiple objet per image
      print(res)
      det_ann = getMultipleDetAnnotation(res)
      print("detection annotation list = " + str(det_ann))
      print("det elements = " + str(len(det_ann)))
      gt_ann = getMultipleGtAnnotation(ann_file_path, image_id)
      print("gt elements = " + str(len(gt_ann)))
      print("groung truth annotation list = " + str(gt_ann))
      vol_err, loc_err, rot_err, iou = getMultipleStatistics(gt_ann, det_ann, true_positive, false_positive, false_negative, not_detected)


      #for single object per image
      #gt_annotation = getGtAnnotation(ann_file_path, image_id)
      #det_annotation = getDetAnnotation(res)

      #print("gt_annotations = " + str(gt_annotation))
      #print("det_annotations = " +str(det_annotation))
            
      #vol_err, loc_err, rot_err, iou = getStatistics(gt_annotation, det_annotation, true_positive, false_positive, false_negative)
      
      #vol_errors.append(vol_err)
      #loc_errors.append(loc_err)
      #rot_errors.append(rot_err)
      #iou_errors.append(iou)
      
      print("Volume error = " + str(vol_err))
      print("Location error = " + str(loc_err))
      print("Rotation error = "+ str(rot_err))
      print("IoU = "+ str(iou))

      #print("TP = " + str(true_positive))
      #print("FP = " + str(false_positive))
      #print("FN = " + str(false_negative))


      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
    
  #print("Volume error = " + str(vol_errors))
  #print("Location error = " + str(loc_errors))
  #print("Rotation error = " + str(rot_errors))
  #print("IoU = " + str(iou_errors))

  for i in range(1, num_cat + 1):
    print("Precision cat " + str(i) + " = " + str(getPrecisionPerCat(i, true_positive, false_positive)))
  

  #get overall precision and recall
  precision = 0
  recall = 0
  for i in range(1, num_cat + 1):
    precision = precision + getPrecisionPerCat(i, true_positive, false_positive)
    recall = recall + getRecallPerCat(i, true_positive, false_negative)
  
  #precision = precision/float(num_cat)
  #recall = recall/float(num_cat)

  #print("precision = " + str(precision))
  #print("recall = " + str(recall))

    

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
