#!/usr/bin/env python
# coding: utf-8

from mmdet.apis import inference_detector, init_detector
import json
import numpy as np
import os
import glob
import cv2
from ensemble_boxes import *


#model 1
config_path_1 = 'myconfigs/cascade_convnext_base.py'
checkpoint_1 = 'work_dirs/cascade_convnext_base/epoch_24.pth'
device_1 = 'cuda:0'


#model_2
config_path_2 = 'myconfigs/cascade_swin_office.py'
checkpoint_2 = 'work_dirs/cascade_swin_office/epoch_24.pth'
device_2 = 'cuda:1'


output_root = 'data/detection-results/'
img_dir = 'data/test/JPEGImages'


model_1 = init_detector(config_path_1, checkpoint_1, device_1)
model_2 = init_detector(config_path_2, checkpoint_2, device_2)

img_list = os.listdir(img_dir)

cls_map = ['shoot']
weights = [1, 1] #weights for two models
iou_thr = 0.5
conf_thr = 0.001

for img in img_list:
    img_h, img_w = cv2.imread(os.path.join(img_dir , img)).shape[:2]
    predict_1 = inference_detector(model_1, os.path.join(img_dir , img))
    predict_2 = inference_detector(model_2, os.path.join(img_dir, img))

    bboxes_list = []
    score_list = []
    label_list= []

    with open(output_root + img.replace('.jpg','.txt'), 'w') as f:
        for i, cls in enumerate(cls_map):  #遍历预测类别

            pred_1 = np.asarray(predict_1[i])
            bbox_1, score_1 = pred_1[:, :4], pred_1[:, 4]
            bbox_1[..., [0, 2]], bbox_1[..., [1, 3]] = bbox_1[..., [0, 2]] / img_w, bbox_1[..., [1, 3]] / img_h
            label_1 = [i] * len(bbox_1)

            bboxes_list.append(bbox_1.tolist())
            score_list.append(score_1.tolist())
            label_list.append(label_1)

            pred_2 = np.asarray(predict_2[i])
            bbox_2, score_2 = pred_2[:, :4], pred_2[:, 4]
            bbox_2[..., [0,2]],  bbox_2[..., [1,3]] =  bbox_2[..., [0,2]] / img_w,  bbox_2[..., [1,3]] / img_h
            label_2 = [i] * len(bbox_2)

            bboxes_list.append(bbox_2.tolist())
            score_list.append(score_2.tolist())
            label_list.append(label_2)

            # print(bboxes_list, score_list, label_list)
            # assert 0
            bboxes, scores, labels =  weighted_boxes_fusion(bboxes_list, score_list, label_list,
                                                           weights=None, iou_thr=iou_thr, skip_box_thr=conf_thr)


            if len(bboxes) > 0:
                for j, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox

                    score = scores[j]
                    f.write(cls + ' ' + str(score) + ' ' + str(x1 * img_w) + ' ' + str(y1 * img_h)
                            + ' ' + str(x2 * img_w ) + ' ' + str(y2 * img_h))
                    f.write('\n')
            
    
    





