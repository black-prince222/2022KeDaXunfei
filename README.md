# 2022科大讯飞开发者大赛--柑橘花果梢识别
object detection  mmdetection

## 初赛!
训练集有420张图片，测试集有180张图片. 图片分辨率均为608x608

排行榜评价指标为mAp0.5.

初赛第四名  分数为0.95111



#### backbone
Cascade rcnn + swin-base + fcn

anchor 比例由[0.5 ，1.0， 2.0] 变成 [0.75  1.25,  1.75] （根据数据集分析统计）


#### 数据增强
1）Mixup

2）prob = 0.5的RandomFlip

3）prob = 0.5, level = 8的BrightnessTransoform

4）prob = 0.5, 默认level的EquailzeTransoform

#### 训练策略
1）480 - 800分辨率的多尺度训练

2）混合精度   

3）EMA   

4）AdamW   

5）全数据训练+补充训练集的缺失标注  **(5折交叉训练，对验证集进行预测。若预测结果中有某个框的置信度>0.9且和原始任意label bbox的iou都为0，则把这个框认为是缺失标注)**    

其余optimizer 和  learn rate设置保持不变

#### 测试阶段
1） TTA：608-672多尺度  Flip


#### 尝试了但无效的策略
1）测试集伪标签训练。 

2）swin 和 convnext的预测结果用[Weighted Bbox Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)作融合.

3)  GroupNormalization 代替Norm-cfg中的BN 和 SyncBN

4)   Mosaic、 bboxjitter、 AutoAugmentation数据增强(在mmdet/dataset/pipeline/transform.py中）

5)   分配策略：ATSS 替代RPN 中的 MaxiouAssigner(修改mmdet/models/dense_heads/anchor_head.py) 

6)   在bbox_roi_extractors全局上下文信息global context(修改mmdet/models/roi_heads/single_level_roi_extractor.py)