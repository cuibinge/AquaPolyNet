data_preprocessor = dict(
    type='FoundationInputSegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[105.30, 145.24, 125.11],
    std=[38.66, 39.85, 34.91],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='FoundationEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmseg.SwinTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint='/mnt/public/usr/wangmingze/pretrain/swin_T_mmseg.pth'),
        # you can download the pretrain weight from mmseg
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
        pretrain_img_size=224,

        # tiny
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
        ),
    decode_head=dict(
        type='Foundation_Decoder_swin_v1_loss6',
        in_channels=[96, 192, 384, 768], 
        out_channels=256,
        drop=0.0,
        loss_type='BCELoss',
        loss_weight=[1,1,1],
        # 新增填充损失配置
        # rect_fill_cfg=dict(
        #     type='RectFillLoss',
        #     weight=0.8,        # 损失权重
        #     min_area=10,       # 最小填充面积
        #     canny_thresh1=50,  # Canny低阈值
        #     canny_thresh2=150, # Canny高阈值
        #     loss_type='mse'    # 损失类型
        # ),
        # 使用自适应填充损失
        rect_fill_cfg=dict(
            type='AdaptiveRectFillLoss',
            base_weight=0.4,              # 基础权重
            min_area=10,                  # 最小填充面积
            low_confidence_thresh=0.3,    # 低置信度阈值
            small_area_thresh=0.05,       # 小面积阈值
            canny_thresh1=50,             # Canny低阈值
            canny_thresh2=150,            # Canny高阈值
            loss_type='mse',              # 损失类型
            debug=False                   # 调试模式
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(),)

find_unused_parameters=True