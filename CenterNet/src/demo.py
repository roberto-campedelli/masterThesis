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
#ann_file_path = "../data/my_dataset/annotations/annotations_3dop_train.json"
ann_file_path = "../data/new_kitti.json"


#initialize list for storing TP, FP and FN for each category
num_cat = 3
true_positive = [0] * (num_cat +1) 
false_negative = [0] * (num_cat +1)
false_positive = [0] * (num_cat +1)

#same thing for the detection
tp_fp_fn_det = [0, 0, 0]

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
      #image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]

      #image id for KITTI
      image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]
      #i cut the zero from the beginning of the image name   
      while(image_id.startswith("0") and len(image_id) > 1):
        image_id = image_id[1:]
          
      print("image id =  " + str(image_id))

      ret = detector.run(image_name)
      res = ret['results']
      #for multiple objet per image
      #print(res)
      det_ann = getMultipleDetAnnotation(res)
      gt_ann = getMultipleGtAnnotation(ann_file_path, image_id)

      if(len(gt_ann) <= 0):
        print("ground truth of the image not found")
        continue
        
      #i print the number of object detected and the number of object in the ground truth
      print("det elements = " + str(len(det_ann)))
      print("gt elements = " + str(len(gt_ann)))

      '''
      #i print the detected and the ground truth elements
      print("detection annotation list = ")
      for _ in det_ann:
        print(str(_) + "\n")
      print("groung truth annotation list = ")
      for _ in gt_ann:
        print(str(_) + "\n")
      '''
      
      #i get the statistics of the image. The mean of the statistics of each object in the image
      vol_err, loc_err, rot_err, iou = getMultipleStatistics(gt_ann, det_ann, true_positive, false_positive, false_negative, tp_fp_fn_det)

      ############################## OLD CODE - for detecting just one element at time
      #for single object per image
      #gt_annotation = getGtAnnotation(ann_file_path, image_id)
      #det_annotation = getDetAnnotation(res)

      #print("gt_annotations = " + str(gt_annotation))
      #print("det_annotations = " +str(det_annotation))
            
      #vol_err, loc_err, rot_err, iou = getStatistics(gt_annotation, det_annotation, true_positive, false_positive, false_negative)
      ################################

      #i create an array with all the parameters for each image
      vol_errors.append(vol_err)
      loc_errors.append(loc_err)
      rot_errors.append(rot_err)
      iou_errors.append(iou)
      
      #print("Volume error = " + str(vol_err))
      #print("Location error = " + str(loc_err))
      #print("Rotation error = "+ str(rot_err))
      #print("IoU = "+ str(iou))
     

      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

  #TODO
  #calculate the mean or some others stats of the array cumulatives of all the error value
  vol_errors_np = (np.array(vol_errors) - min(vol_errors))/float(max(vol_errors) - min(vol_errors))
  total_vol_error = np.sum(vol_errors_np) / vol_errors_np.size

  rot_errors_np = (np.array(rot_errors) - min(rot_errors))/float(max(rot_errors) - min(rot_errors))
  total_rot_error = np.sum(rot_errors_np) / vol_errors_np.size

  loc_errors_np = (np.array(loc_errors) - min(loc_errors))/float(max(loc_errors) - min(loc_errors))
  total_loc_error = np.sum(loc_errors_np) / loc_errors_np.size

  total_iou = np.sum(np.array(iou_errors)) / len(iou_errors)

  print(vol_errors_np.size)
  print(rot_errors_np.size)
  print(loc_errors_np.size)
  print(len(iou_errors))

  print("Total Volume error = " + str(total_vol_error))
  print("Location error = " + str(total_loc_error))
  print("Rotation error = " + str(total_rot_error))
  print("IoU = " + str(total_iou))

  print("TP = " + str(true_positive))
  print("FP = " + str(false_positive))
  print("FN = " + str(false_negative))

  print("TP - FP - FN detection = " + str(tp_fp_fn_det))


  for i in range(1, num_cat + 1):
    print("Precision cat " + str(i) + " = " + str(getPrecisionPerCat(i, true_positive, false_positive)))
  for i in range(1, num_cat + 1):
    print("Recall cat " + str(i) + " = " + str(getRecallPerCat(i, true_positive, false_negative)))

  #get overall precision and recall
  precision = 0
  recall = 0
  for i in range(1, num_cat + 1):
    precision = precision + getPrecisionPerCat(i, true_positive, false_positive)
    recall = recall + getRecallPerCat(i, true_positive, false_negative)
  
  precision = precision/float(num_cat)
  recall = recall/float(num_cat)

  print("precision classification = " + str(precision))
  print("recall classification= " + str(recall))

  precision_detection = tp_fp_fn_det[0]/float(tp_fp_fn_det[0] + tp_fp_fn_det[1])
  recall_detection = tp_fp_fn_det[0]/float(tp_fp_fn_det[0] + tp_fp_fn_det[2])

  print("precision detection = " + str(precision_detection))
  print("recall detection = " + str(recall_detection))


    

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
