from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def printInfo(res):
  for i in res:
    if res[i].size != 0:
      cat = i
      break
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
  print("box2d = " + str(box2d))
  print("center = " + str(center))
  print("dim = " + str(dim))
  print("rot_y = " + str(rot_y))
  print("dir = " + str(opt.demo))

def readAnnotation(filePath, image_id):
  line_to_find = "\"image_id\": " + str(image_id)
  with open(filePath) as f:
    lines = f.readlines()
  for line in lines:
    if line_to_find in line:
      return line


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
      print(image_name)
      image_id = image_name[image_name.rfind('/') + 1: image_name.rfind('.')]
      print(image_id)
      print(readAnnotation("../data/my_dataset/annotations/annotations_3dop_train.json", image_id))
      ret = detector.run(image_name)
      res = ret['results']
      printInfo(res)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
