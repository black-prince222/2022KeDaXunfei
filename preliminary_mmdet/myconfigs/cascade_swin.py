#pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type = 'CascadeRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        #init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs = 4,
                num_shared_fcs = 1,
                in_channels=256,
                conv_out_channels = 256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg = dict(type = 'SyncBN', requires_grad = True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),

            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),

            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        ]),
# model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels = False,
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),

            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels = False,
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN


#half precision training
fp16 = dict(loss_scale = 512.)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type = 'MixUp', img_scale = (640,1333), ratio_range = (0.5,1.5)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5),
    dict(
        type='Resize',
        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        multiscale_mode='value',
        keep_ratio=True),
    # dict(
    #     type='AutoAugment',
    #     policies=[[
    #         dict(
    #             type='Resize',
    #             img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                        (736, 1333), (768, 1333), (800, 1333)],
    #             multiscale_mode='value',
    #             keep_ratio=True)
    #     ],
    #               [
    #                   dict(
    #                       type='Resize',
    #                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
    #                       multiscale_mode='value',
    #                       keep_ratio=True),
    #                   dict(
    #                       type='RandomCrop',
    #                       crop_type='absolute_range',
    #                       crop_size=(384, 600),
    #                       allow_negative_crop=True),
    #                   dict(
    #                       type='Resize',
    #                       img_scale=[(480, 1333), (512, 1333), (544, 1333),
    #                                  (576, 1333), (608, 1333), (640, 1333),
    #                                  (672, 1333), (704, 1333), (736, 1333),
    #                                  (768, 1333), (800, 1333)],
    #                       multiscale_mode='value',
    #                       override=True,
    #                       keep_ratio=True)
    #               ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 1333),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset settings
dataset_type = 'Fruit'
data_root = 'data/coco_fruit/'

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/train',
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        filter_empty_gt = False,
    ),
    pipeline=train_pipeline)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')



#ema
# custom_hooks = [dict(type='ExpMomentumEMAHook',
#                      resume_from = None,
#                      momentum = 0.0001,
#                      priority = 49)]


evaluation = dict(save_best = 'auto', interval=1, metric='bbox')
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=0.0001,   #0.0001 0.608 0.00005 = 0.451
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(policy='step',warmup='linear',warmup_iters=1000, step=[40,55])
runner = dict(type='EpochBasedRunner', max_epochs=60)



checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './pretrained/cascade_mask_rcnn_swin_tiny_patch4_window7.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)









# # model settings
#
# custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa
#
# model = dict(
#     type='CascadeRCNN',
#     backbone=dict(
#         #_delete_=True,
#         type='mmcls.ConvNeXt',
#         arch='small',
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         drop_path_rate=0.4,
#         layer_scale_init_value=1.0,
#         gap_before_final_norm=False,
#         init_cfg=dict(
#                 type='Pretrained', checkpoint=checkpoint_file,
#                 prefix='backbone.')),
#     neck=dict(
#         type='FPN',
#         in_channels=[96, 192, 384, 768],
#         out_channels=256,
#         num_outs=5),
#
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             #ratios=[0.5, 1.0, 2.0],
#             ratios = [0.5, 0.75, 1.0, 1.5, 2.5],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#     roi_head=dict(
#         type='CascadeRoIHead',
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=8,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 reg_decoded_bbox=True,
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=8,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 reg_decoded_bbox=True,
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=8,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 reg_decoded_bbox=True,
#                 loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
#         ]),
#
# # model training and testing settings
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.001,
#             nms=dict(type='soft_nms', iou_threshold=0.5),
#             max_per_img=100)))
#
#
# # dataset settings
# dataset_type = 'xrayDataset'
# data_root = 'datasets/xray/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     #dict(type='LoadImageFromFile'),
#     #dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type = 'MixUp', img_scale = (640,1333),
#     #      ratio_range = (0.5, 1.5), pad_val = 255.0),#填充白色
#
#     dict(
#         type='Resize',
#         img_scale=[(640, 1333)], #960有提升 0.523
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5), #horizontal
#
#
#     # dict(type='RandomShift', shift_ratio=0.5),
#     # dict(type='AutoAugment',
#     #      policies=[
#     #          [dict(type='Rotate', prob=0.5, level=5, max_rotate_angle=10),
#     #           dict(type='Shear', prob=0.3, level=5, max_shear_magnitude=0.2),
#     #           ],
#     #          [
#     #              dict(type='Albu',
#     #                   transforms=[
#     #                       dict(type='RandomScale', scale_limit=0.2, p=0.5),  # works
#     #                       dict(type='OneOf',
#     #                            transforms=[
#     #                                dict(type='MedianBlur', p=0.01),
#     #                                dict(type='Blur', p=0.01)
#     #                            ]),
#     #                   ],
#     #                   bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels'])
#     #                   ),
#     #          ]
#     #      ]
#     #      ),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(640, 1333),
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# #half precision training
# fp16 = dict(loss_scale = 512.)
#
# train_dataset = dict(
#     type = 'MultiImageMixDataset',
#     dataset = dict(
#         type=dataset_type,
#         ann_file=data_root + "annotations/train.json",
#         img_prefix=data_root + 'train/',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#         ],
#         filter_empty_gt = False,
#     ),
#     pipeline = train_pipeline)
#
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=8,
#     train= train_dataset,
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline))
#
#
# #ema
# custom_hooks = [dict(type='ExpMomentumEMAHook',
#                      resume_from = None,
#                      momentum = 0.0001,
#                      priority = 49)]
#
#
# evaluation = dict(save_best = 'auto', interval=1, metric='bbox')
# optimizer_config = dict(grad_clip=None)
# optimizer = dict(
#     #_delete_=True,
#     constructor='LearningRateDecayOptimizerConstructor',
#     type='AdamW',
#     lr=0.0004,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={
#         'decay_rate': 0.7,
#         'decay_type': 'layer_wise',
#         'num_layers': 6
#     })
# lr_config = dict(policy='step',warmup='linear',warmup_iters=1000, step=[8,11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=200,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]
#
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None#'pretrained/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'
# resume_from = None
# workflow = [('train', 1)]
#
# # disable opencv multithreading to avoid system being overloaded
# opencv_num_threads = 0
# # set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'
#
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)
#
