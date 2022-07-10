#!/usr/bin/env python
# coding: utf-8

from mmdet.apis import inference_detector, init_detector
import json
import numpy as np
import os
import glob
import cv2



config_path = 'myconfigs/cascade_swin_0.95111.py'
checkpoint = 'work_dirs/cascade_swin_0.95111/epoch_24.pth'
device = 'cuda:2'
output_root = 'data/detection-results/'
img_dir = 'data/test/JPEGImages'



model = init_detector(config_path, checkpoint, device)
img_list = os.listdir(img_dir)

for img in img_list:

    predict = inference_detector(model, os.path.join(img_dir , img))

    with open(output_root + img.replace('.jpg','.txt'), 'w') as f:
        for i, bboxes in enumerate(predict):  #遍历预测类别
            if len(bboxes) > 0:
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    f.write('shoot' + ' ' + str(score) + ' ' + str(x1) + ' ' + str(y1)
                            + ' ' + str(x2) + ' ' + str(y2))
                    f.write('\n')
            
    
    





