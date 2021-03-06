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
ann_file_path = "../data/kitti_testing/annotations/new_kitti.json"


#initialize list for storing TP, FP and FN for each category
num_cat = 3
true_positive = [0] * (num_cat +1) 
false_negative = [0] * (num_cat +1)
false_positive = [0] * (num_cat +1)

#same thing for the detection
tp_fp_fn_det = [0, 0, 0]

#array for saving TP and FP for each 3dbbox parameter
#dim_loc_rot_tpfp = [tp_dim, fp_dim, tp_loc, fp_loc, tp_rot, fp_rot]
dim_loc_rot_tpfp = [0, 0, 0, 0, 0, 0]

#define the lists for storing the errors 
vol_errors = []
iou_errors = []
rot_errors = []
loc_errors = []
dim_errors = []
hwl_errors = []

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
      #print("det elements = " + str(len(det_ann)))
      #print("gt elements = " + str(len(gt_ann)))
   
        
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
      vol_err, loc_err, rot_err, iou, dim_err, hwl_err = getMultipleStatistics(gt_ann, det_ann, true_positive, false_positive, false_negative, tp_fp_fn_det, dim_loc_rot_tpfp)

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
      dim_errors.append(dim_err)
      iou_errors.append(iou)
      hwl_errors.append(hwl_err)
      
      #print("Volume error = " + str(vol_err))
      #print("Location error = " + str(loc_err))
      #print("Rotation error = "+ str(rot_err))
      #print("IoU = "+ str(iou))
     

      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
       #print(time_str)

  

  #calculate the average mean of the array cumulatives of all the error value and normalize it in [0,1]
  #vol_errors_normalized = (np.array(vol_errors) - min(vol_errors))/float(max(vol_errors) - min(vol_errors))
  #total_vol_error_normalized = np.sum(vol_errors_normalized) / vol_errors_normalized.size

  rot_errors_normalized = (np.array(rot_errors) - min(rot_errors))/float(max(rot_errors) - min(rot_errors))
  total_rot_error_normalized = np.sum(rot_errors_normalized) / rot_errors_normalized.size

  loc_errors_normalized = (np.array(loc_errors) - min(loc_errors))/float(max(loc_errors) - min(loc_errors))
  total_loc_error_normalized = np.sum(loc_errors_normalized) / loc_errors_normalized.size

  dim_errors_normalized = (np.array(dim_errors) - min(dim_errors))/float(max(dim_errors) - min(dim_errors))
  total_dim_error_normalized = np.sum(dim_errors_normalized) / dim_errors_normalized.size

  total_iou = np.sum(np.array(iou_errors)) / len(iou_errors)

  #vol_rmse = getRMSE(vol_errors)
  rot_rmse = getRMSE(rot_errors)
  dim_rmse = getRMSE(dim_errors)
  loc_rmse = getRMSE(loc_errors)

  #print("Vol RMSE = "  + str(vol_rmse))
  #print("Dim RMSE = "  + str(dim_rmse))
  #print("Loc RMSE = " + str(loc_rmse))
  #print("Rot RMSE = "  + str(rot_rmse))

  print("\n")
  np_hwl_errors = np.array(hwl_errors)
  mean_hwl_errors = np.sum(np_hwl_errors, axis=0) / len(hwl_errors)
  print("Height error = " + str(mean_hwl_errors[0]))
  print("Width error = " + str(mean_hwl_errors[1]))
  print("Length error = " + str(mean_hwl_errors[2])) 
  print("\n")
 

  #print("Vol error not normalized = " + str(np.sum(np.array(vol_errors)) / len(vol_errors)))
  print("Dim error  = " + str(np.sum(np.array(dim_errors)) / len(dim_errors)))
  print("Loc error  = " + str(np.sum(np.array(loc_errors)) / len(loc_errors)))
  print("Rot error  = " + str(np.sum(np.array(rot_errors)) / len(rot_errors)))

  #print(vol_errors_np.size)
  #print(rot_errors_np.size)
  #print(loc_errors_np.size)
  #print(len(iou_errors))

  #print("Volume error normalized in [0,1] = " + str(total_vol_error_normalized))
  #print("Dimension error normalized in [0,1]  = " + str(total_dim_error_normalized))
  #print("Location error normalized in [0,1]  = " + str(total_loc_error_normalized))
  #print("Rotation error normalized in [0,1]  = " + str(total_rot_error_normalized))

  #print("IoU = " + str(total_iou))

  #print("classification TP = " + str(true_positive))
  #print("classification FP = " + str(false_positive))
  #print("classification FN = " + str(false_negative))

  #print("detection TP - FP - FN = " + str(tp_fp_fn_det))

  #print("dim_loc_rot_tpfp = " + str(dim_loc_rot_tpfp))

  print("\n")

  print("Dimension precision = " + str(getPrecision(dim_loc_rot_tpfp[0], dim_loc_rot_tpfp[1])))
  print("Location precision = " + str(getPrecision(dim_loc_rot_tpfp[2], dim_loc_rot_tpfp[3])))
  print("Rotation precision = " + str(getPrecision(dim_loc_rot_tpfp[4], dim_loc_rot_tpfp[5])))

  print("\n")


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

  print("\n")

  print("precision classification = " + str(precision))
  print("recall classification= " + str(recall))

  precision_detection = tp_fp_fn_det[0]/float(tp_fp_fn_det[0] + tp_fp_fn_det[1])
  recall_detection = tp_fp_fn_det[0]/float(tp_fp_fn_det[0] + tp_fp_fn_det[2])

  #print("precision detection = " + str(precision_detection))
  #print("recall detection = " + str(recall_detection))

  print("\n")

  printThreshold()



    

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
