# 创建 configs_finetune/Swin_T/Swin_T_whu_4ch.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/standard_512x512_foundationdataset.py',
    '../_base_/schedules/schedule_default.py',
]

# 直接在配置文件中定义4通道模型
model = dict(
    type='FoundationEncoderDecoder',
    data_preprocessor=dict(
        type='FoundationInputSegDataPreProcessor',
        mean=[90.31, 92.44, 93.44, 131.03]*2,  # 4通道
        std=[57.93, 59.98, 61.62, 58.62]*2,        # 4通道
        bgr_to_rgb=False,
        size_divisor=32,
        pad_val=0,
        seg_pad_val=255,
        test_cfg=dict(size_divisor=32)),
    backbone=dict(
        type='mmseg.SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.15,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        in_channels=4,  # ⭐️ 关键：设置为4通道
    ),
    decode_head=dict(
        type='Foundation_Decoder_swin_v1_loss6',
        in_channels=[96, 192, 384, 768], 
        out_channels=256,
        drop=0.0,
        loss_type='BCELoss',
        loss_weight=[1,1,1],
        rect_fill_cfg=dict(
            type='AdaptiveRectFillLoss',
            base_weight=0.4,
            min_area=10,
            low_confidence_thresh=0.3,
            small_area_thresh=0.05,
            canny_thresh1=50,
            canny_thresh2=150,
            loss_type='mse',
            debug=False
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

find_unused_parameters=True

# 其他配置保持不变...
# You can change dataloader parameters here
bs=2
gpu_nums = 8
bs_mult = 1
num_workers = 8
persistent_workers = True

# data_list path
train_data_list = 'data_list/whu/train.txt'
test_data_list = 'data_list/whu/test.txt'

# ... 其余配置 ...
# training schedule for pretrain
max_iters = 4e4
val_interval = 200
logger_interval = 20
base_lr = 0.0001 * (bs * gpu_nums / 16) * bs_mult # lr is related to bs*gpu_num, default 16-0.0001


# If you want to train with some backbone init, you must change the dir for your personal save dir path
# But I think you will use our pretrained weight, you may do not need backbone_checkpoint
backbone_checkpoint = None
load_from = 'the checkpoint path' # !!!! must change this !!!!
resume_from = None

# which part you want to finetune
finetune_cfg = ['neck', 'decoder', 'ab mask head', 'cd mask head', 'ab query head', 'cd query head']

# If you want to use wandb, make it to 1
wandb = 0

# You can define which dir want to save checkpoint and loggings
names = 'Swin_T_whu'
work_dir = '/mnt/public/usr/wangmingze/work_dir/finetune/' + names



""" ************************** data **************************"""
train_dataset = dict(
    dataset=dict(
        data_list=train_data_list,
    )
)
train_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=bs,
    num_workers=num_workers,
    persistent_workers=persistent_workers, 
    dataset=dict(
        data_list=test_data_list,
    )
)
test_dataloader = val_dataloader

""" ************************** schedule **************************"""
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr
        ),
    # backbone lr_mult = 0.01
    )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=logger_interval, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, img_shape=(512, 512, 3)))

""" ************************** visualization **************************"""
if wandb:
    vis_backends = [dict(type='CDLocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        save_dir=
                        '/mnt/public/usr/wangmingze/opencd/wandb/try2',
                        init_kwargs={
                            'entity': "wangmingze",
                            'project': "opencd_all_v4",
                            'name': names,}
                            )
                    ]
