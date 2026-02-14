import os
from math import log

# 第三方库 (Third-party libraries)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.measure import label, regionprops

# PyTorch 相关
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# 本地项目模块 (Local application modules)
from hisup.backbones import build_backbone
from hisup.utils.polygon import generate_polygon, get_pred_junctions
from .stransconv import DeepSparse
def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])
    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - targets)
    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (t / w)
    return loss.mean()




class FDSM(nn.Module):
    def __init__(self, channel, num_filters=4, num_rows=128, num_cols=128):
        super(FDSM, self).__init__()
        self.channel = channel
        self.num_filters = num_filters
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Initialize DeepSparse filters with large receptive field
        self.large_filters = nn.ModuleList([
            DeepSparse(channel, channel, num_rows, num_cols) for _ in range(num_filters)
        ])
        self.filter_weights = nn.Parameter(torch.randn(num_filters, channel, 1, 1))
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # Feature aggregation
        self.agg_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channel * 2),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, num_filters, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # detector_fdsm.py 中的 FDSM.forward 方法

    def forward(self, features):
        B, C, H, W = features.shape

        # Aggregate features
        agg_features = self.agg_conv(torch.cat([features, features], dim=1))

        # Generate dynamic weights
        weights = self.weight_gen(agg_features)
        weights = weights.view(B, self.num_filters, 1, 1)

        # --- 这里的可视化代码建议在训练时注释掉，会显著拖慢速度并增加显存占用 ---
        # 仅保留计算逻辑

        filtered_features_list = []
        for i in range(self.num_filters):
            # DeepSparse 内部现在已经修正为 FP32 计算，返回的是 orig_dtype
            filtered = self.large_filters[i](features)
            filtered_features_list.append(filtered)

        filtered_features = torch.stack(filtered_features_list, dim=1)  # [B, num_filters, C, H, W]

        # === 核心修正开始 ===

        # 1. 保存原始类型 (FP16/FP32)
        orig_dtype = features.dtype

        # 2. 所有 FFT 相关操作必须在 float32 下进行
        features_f32 = features.float()
        filtered_features_f32 = filtered_features.float()
        weights_f32 = weights.float()

        # 3. FFT 变换 (使用 norm='ortho')
        # features_fft: [B, C, H, W//2+1] (Complex64)
        features_fft = torch.fft.rfftn(features_f32, s=(H, W), norm='ortho')

        # filtered_fft: [B, num_filters, C, H, W//2+1] (Complex64)
        filtered_fft = torch.fft.rfftn(filtered_features_f32, s=(H, W), norm='ortho')

        # 4. 频域计算 (完全在 Complex64/Float32 下进行，不要拆分 view_as_real 除非必要)
        # 广播 features 以匹配 filters 维度
        features_fft_expanded = features_fft.unsqueeze(1)  # [B, 1, C, H, W//2+1]

        # 复数乘法: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        # PyTorch 支持复数张量直接相乘，不需要手动拆解实部虚部，这样更稳定且快
        product = features_fft_expanded * filtered_fft  # Element-wise complex multiplication

        # 5. 加权求和
        # weights: [B, num_filters, 1, 1] -> 扩展到频域维度
        weights_expanded = weights_f32.unsqueeze(-1).expand(-1, -1, C, H, W // 2 + 1)

        # 这一步将 num_filters 维度规约掉
        # sum(weight * product)
        weighted_sum = torch.sum(weights_expanded * product, dim=1)  # [B, C, H, W//2+1] (Complex)

        # 6. 逆变换 (使用 norm='ortho')
        output_f32 = torch.fft.irfftn(weighted_sum, s=(H, W), norm='ortho')

        # 7. 残差连接 (此时 output_f32 还是 float32)
        # 必须确保 residual_weight 参与运算时也是 float32，避免被自动降级
        output = output_f32 + self.residual_weight.float() * features_f32

        # 8. 最后才转回 FP16
        output = output.to(orig_dtype)

        return output

class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.junc_loss = nn.CrossEntropyLoss()
        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        if not test:
            from hisup.encoder import Encoder
            self.encoder = Encoder(cfg)

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.fdsm = FDSM(dim_in, num_filters=4, num_rows=128, num_cols=128)
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        self.a2m_att = FDSM(dim_in)
        self.a2j_att = FDSM(dim_in)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        self.refuse_conv = self._make_conv(2, dim_in//2, dim_in)
        self.final_conv = self._make_conv(dim_in*2, dim_in, 2)

        self.train_step = 0

    def forward(self, images, annotations=None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations)

    def forward_test(self, images, annotations=None):
        device = images.device
        outputs, features = self.backbone(images)

        features = self.fdsm(features)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature+jloc_feature)
        jloc_att_feature = self.a2j_att(afm_feature+jloc_feature)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        joff_pred = outputs[:, :].sigmoid() - 0.5
        mask_pred = mask_pred.softmax(1)[:, 1:]
        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]
        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]
        remask_pred = remask_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):
            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])
            juncs_pred[:, 0] *= scale_x
            juncs_pred[:, 1] *= scale_y

            if not self.test_inria:
                polys, scores = [], []
                props = regionprops(label(mask_pred_per_im > 0.5))
                for prop in props:
                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, juncs_pred, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue
                    polys.append(poly)
                    scores.append(score)
                batch_scores.append(scores)
                batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        extra_info = {}
        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info

    def forward_train(self, images, annotations=None):
        torch.cuda.empty_cache()
        self.train_step += 1
        device = images.device
        targets, metas = self.encoder(annotations)
        outputs, features = self.backbone(images)

        features = self.fdsm(features)

        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm': 0.0,
            'loss_remask': 0.0
        }

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature+mask_feature)
        jloc_att_feature = self.a2j_att(afm_feature+jloc_feature)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        if targets is not None:
            loss_dict['loss_jloc'] += self.junc_loss(jloc_pred, targets['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += sigmoid_l1_loss(outputs[:, :], targets['joff'], -0.5, targets['jloc'])
            loss_dict['loss_mask'] += F.cross_entropy(mask_pred, targets['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += F.l1_loss(afm_pred, targets['afmap'])
            loss_dict['loss_remask'] += F.cross_entropy(remask_pred, targets['mask'].squeeze(dim=1).long())
        extra_info = {}
                
        return loss_dict, extra_info
        torch.cuda.empty_cache()
        
    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
            nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, dim_out, kernel_size=1),
        )
        return layer
