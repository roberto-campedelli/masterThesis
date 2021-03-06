import numpy as np

dim_threshold = loc_threshold = rot_threshold = 0.5
def getMultipleDetAnnotation(res):
    #define a list for storing all the detection for the image
    det_ann_list = []
    #for each category in the result detected
    for i in range(1, len(res) + 1):
        #if is detected at least one element of cat i
        if len(res[i]) > 0:
            for j in range(len(res[i])):
                info = res[i]
                cat = i
                alpha = info[j,0]
                topLeft_x = info[j,1]
                topLeft_y = info[j,2]
                topRight_x = info[j,3]
                bottomLeft_y = info[j,4]
                height_m = info[j,5]
                width_m = info[j,6]
                lentgh_m = info[j,7]
                center_x = info[j,8]
                center_y = info[j,9]
                center_z = info[j,10]
                rot_y = info[j,11]
  
                box2d = [topLeft_x, topLeft_y, topRight_x - topLeft_x, bottomLeft_y - topLeft_y]
                center = [center_x, center_y, center_z]
                dim = [height_m, width_m, lentgh_m]
                det_annotations = [cat, dim, box2d, alpha, center, rot_y]
                #i put all the det annotation of the image in a list
                if(center[2] < 100):
                  det_ann_list.append(det_annotations)
    return det_ann_list
    #now i have a list of detection of the image
    #i have to compare them with the gt_annotation of the same image

def getMultipleGtAnnotation(filePath, image_id):
    gt_ann_list = []
    line_to_find = "\"image_id\": " + str(image_id) + ","
    with open(filePath) as f:
        lines = f.readlines()
    for line in lines:
        if line_to_find in line:
            annotation = line

            cat_id = int(findBetween(annotation, "\"category_id\":", "\"dim\":")[:-1])
            dim = list(map(float, findBetween(annotation, "\"dim\":", "\"bbox\":")[1:-2].split(", ")))
            bbox = list(map(float, findBetween(annotation, "\"bbox\":", "\"depth\":")[1:-2].split(", ")))
            alpha = float(findBetween(annotation, "\"alpha\":", "\"truncated\":")[:-1])
            location = list(map(float,findBetween(annotation, "\"location\":", "\"rotation_y\":")[1:-2].split(", ")))
            rotation_y = float(findBetween(annotation, "\"rotation_y\":", "}"))
            
            ############################### KITTI ###########
            #i put all the gt annotations of the image in a list
            if(cat_id == 4 or cat_id == 5 or cat_id == 7 or cat_id == 8):
                cat_id = 2
            if(cat_id == 6):
                cat_id = 1  
        
            gt_annotations = [cat_id, dim, bbox, alpha, location, rotation_y]

            if(cat_id != 9 and location[2] < 100):
                gt_ann_list.append(gt_annotations)
    return gt_ann_list

def getMultipleStatistics(gt_ann_list, det_ann_list, true_positive, false_positive, false_negative, tp_fp_fn_det, dim_loc_rot_tpfn):
    
    #check if the list of the detection is empty, in that case update false negative values and return
    if len(det_ann_list) == 0 :
      while gt_ann_list :
        cat_not_det = gt_ann_list.pop()[0]
        false_negative[cat_not_det] = false_negative[cat_not_det] + 1
      return  float(0), float(0), float(0), float(0), float(0), [0,0,0]

    if(len(gt_ann_list) > len(det_ann_list)):
        tp_fp_fn_det[2] = tp_fp_fn_det[2] + (len(gt_ann_list) - len(det_ann_list))

    #for each element of the gt list i compare it with the closest in the det list, minimizing the distance of the centers
    right_det_element = det_ann_list[0]
    index_right_element = 0
    num_detection = 0
    tot_vol_err, tot_loc_err, tot_rot_err, tot_iou, tot_dim_err = float(0), float(0), float(0), float(0), float(0)
    tot_hwl_err = [0, 0, 0]

    for gt_element in gt_ann_list :
        min_loc_err = 1000
        if len(det_ann_list) <= 0:
            break
        for det_element in det_ann_list:
            center_diff = np.linalg.norm(np.array(det_element[4]) - np.array(gt_element[4]))
            if center_diff < min_loc_err :
                min_loc_err = center_diff
                right_det_element = det_element
                index_right_element = det_ann_list.index(det_element)
        #print(index_right_element)
        #det_ann_list.remove(right_det_element)
        #print("i remove element = " + str(right_det_element) + "of index " + str(index_right_element))
        det_ann_list.pop(index_right_element)
        num_detection = num_detection + 1
        vol_err, location_err, rot_err, iou, dim_err, hwl_err = getStatistics(gt_element, right_det_element, true_positive, false_positive, false_negative, tp_fp_fn_det, dim_loc_rot_tpfn)
        tot_vol_err, tot_loc_err, tot_rot_err, tot_iou, tot_dim_err =  tot_vol_err + vol_err, tot_loc_err + location_err, tot_rot_err + rot_err, tot_iou + iou, tot_dim_err + dim_err
        tot_hwl_err = [tot_hwl_err[0] + hwl_err[0],tot_hwl_err[1] + hwl_err[1],tot_hwl_err[2] + hwl_err[2]]
        gt_ann_list.remove(gt_element)

    #if det ann list is  not empty means that is detected something that is not in the gt, and this means a 
    #false positive for the category of the object detected
    #while det_ann_list :
    #    cat_det_wrong = det_ann_list.pop()[0]
    #    false_positive[cat_det_wrong] = false_positive[cat_det_wrong] + 1

    #if gt ann list is not empty means that something was not detected, and this means a 
    #false negative for the category of the object detected
    while gt_ann_list :
        cat_not_det = gt_ann_list.pop()[0]
        false_negative[cat_not_det] = false_negative[cat_not_det] + 1

   
    #for avoiding /0 error
    if num_detection == 0:
        num_detection = 1

    for i in range(len(tot_hwl_err)):
      tot_hwl_err[i] = tot_hwl_err[i]/float(num_detection)

    return  tot_vol_err/float(num_detection), tot_loc_err/float(num_detection), tot_rot_err/float(num_detection), tot_iou/float(num_detection), tot_dim_err/float(num_detection), tot_hwl_err



def getDetAnnotation(res):
  cat = -1
  for i in res:
    #if is detected at least one element of category i
    if len(res[i]) > 0:
      cat = i
      break
  if cat <= 0 : 
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
  return line[index_start:index_end].strip()#[:-1]

def getGtAnnotation(filePath, image_id):
  line_to_find = "\"image_id\": " + str(image_id) + ","
  with open(filePath) as f:
    lines = f.readlines()
  for line in lines:
    if line_to_find in line:
      annotation = line

  cat_id = int(findBetween(annotation, "\"category_id\":", "\"dim\":")[:-1])
  dim = list(map(float, findBetween(annotation, "\"dim\":", "\"bbox\":")[1:-2].split(", ")))
  bbox = list(map(float, findBetween(annotation, "\"bbox\":", "\"depth\":")[1:-2].split(", ")))
  alpha = float(findBetween(annotation, "\"alpha\":", "\"truncated\":")[:-1])
  location = list(map(float,findBetween(annotation, "\"location\":", "\"rotation_y\":")[1:-2].split(", ")))
  rotation_y = float(findBetween(annotation, "\"rotation_y\":", "}"))
############################### for kitti is cat_id for my dataset is cat_id +1
  gt_annotations = [cat_id + 1, dim, bbox, alpha, location, rotation_y]
  return gt_annotations

#calculation of the IoU of the front face of the bbox 3d
#the bbox2d is defined as [topLeft.x, topLeft.y, topRight.x - topLeft.x, bottomRight.y - topLeft.y]
def getIoU(det_bbox, gt_bbox, tp_fp_fn_det):
  #top left point of the intersaction square
  tl_int_x = max(det_bbox[0], gt_bbox[0])
  tl_int_y = max(det_bbox[1], gt_bbox[1])
  #top right x of the intersection square, the y is the same of the top left point
  tr_int_x = min(det_bbox[0] + det_bbox[2], gt_bbox[0] + gt_bbox[2])
  #bottom right y of the intersaction square, the x is the same of the top left point
  bl_int_y = min(det_bbox[1] + det_bbox[3], gt_bbox[1] + gt_bbox[3])

  int_area = (tr_int_x - tl_int_x)*(bl_int_y - tl_int_y)
  if int_area < 0:
    int_area = 0
        
  union_area = (det_bbox[2] * det_bbox[3]) + (gt_bbox[2] * gt_bbox[3]) - int_area

  iou = int_area / float(union_area)

  if iou > 0.5:
      tp_fp_fn_det[0] = tp_fp_fn_det[0] + 1
  else:
      tp_fp_fn_det[1] = tp_fp_fn_det[1] + 1

  return iou


#calculate statistics between ground truth and detection
#annotation order = [cat_id, dim, bbox, alpha, location, rotation_y]
def getStatistics(gt_annotation, det_annotation, true_positive, false_positive, false_negative, tp_fp_fn_det, dim_loc_rot_tpfp):
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
  vol_err = abs(det_vol - gt_vol)

  
  #check the difference between the dimension
  height_err = abs(det_dim[0] - gt_dim[0])
  width_err = abs(det_dim[1] - gt_dim[1])
  length_err = abs(det_dim[2] - gt_dim[2])

  dim_ratio = np.array(det_dim)/np.array(gt_dim)

  if (isTruePositive(dim_ratio[0], dim_threshold) and isTruePositive(dim_ratio[1], dim_threshold) and isTruePositive(dim_ratio[2], dim_threshold)):
      dim_loc_rot_tpfp[0] = dim_loc_rot_tpfp[0] + 1
  else:
      dim_loc_rot_tpfp[1] = dim_loc_rot_tpfp[1] + 1

  dim_err = [height_err, width_err, length_err]
  mean_dim_err = (height_err + width_err + length_err)/float(3)

  #check IoU of the front face of the 3dbbox
  iou = getIoU(det_bbox, gt_bbox, tp_fp_fn_det)

  #check the location of the center of the 3dbbox error using the euclidean distance
  location_err = np.linalg.norm(np.array(gt_location) - np.array(det_location))

  loc_ratio = np.array(det_location)/ np.array(gt_location)
  
  if( isTruePositive(loc_ratio[0], loc_threshold) and isTruePositive(loc_ratio[1], loc_threshold) and isTruePositive(loc_ratio[2], loc_threshold)):
      dim_loc_rot_tpfp[2] = dim_loc_rot_tpfp[2] + 1
  else:
      dim_loc_rot_tpfp[3] = dim_loc_rot_tpfp[3] +1  

  #check the error of the rotation 
  
  if gt_rot_y < 0 :
    gt_rot_y = gt_rot_y + ( 2 * 3.14159)
  if det_rot_y < 0 :
    det_rot_y = det_rot_y + ( 2 * 3.14159)

  #print("gt rotation = " + str(gt_rot_y))
  #print("det rotation = " + str(det_rot_y))
  
  rot_err = abs(gt_rot_y - det_rot_y)
  
  #print("rotation error = " + str(rot_err))

  rot_ratio = det_rot_y/gt_rot_y

  #print("rot ratio = " + str(rot_ratio))

  if( isTruePositive(rot_ratio, rot_threshold)):
      dim_loc_rot_tpfp[4] = dim_loc_rot_tpfp[4] + 1
  else:
      dim_loc_rot_tpfp[5] = dim_loc_rot_tpfp[5] +1

  return vol_err, location_err, rot_err, iou , mean_dim_err, dim_err

def getPrecisionPerCat(cat_id, true_positive, false_positive):
  return getPrecision(true_positive[cat_id], false_positive[cat_id])        
  
  #if (true_positive[cat_id] + false_positive[cat_id]) == 0 :
  #  return 0
  #precision = true_positive[cat_id]/float(true_positive[cat_id] + false_positive[cat_id])
  #return precision
 
def getRecallPerCat(cat_id, true_positive, false_negative):
  if (true_positive[cat_id] + false_negative[cat_id]) == 0 :
      return 0
  recall = true_positive[cat_id]/float(true_positive[cat_id] + false_negative[cat_id])
  return recall

def isTruePositive(value, threshold):
    if (value > threshold and value < (1 + 1 - threshold)):
        return True
    else:
        return False

def getPrecision(true_positive, false_positive):
    if(true_positive + false_positive == 0):
        return 0
    else:
        return(true_positive/float(true_positive + false_positive))

def getRMSE(list_of_value):
    arraynp = np.array(list_of_value)
    rmse = np.sqrt(np.sum(np.mean(np.square(arraynp))))#/arraynp.size)
    return rmse

def printThreshold():
  print("dimension threshold = " + str(dim_threshold))
  print("location threshold = " + str(loc_threshold))
  print("rotation threshold = " + str(rot_threshold))
 
