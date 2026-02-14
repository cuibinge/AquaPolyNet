import cv2
import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import canny
from scipy.ndimage import binary_dilation
from hisup.csrc.lib.afm_op import afm
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Encoder")

class Encoder(object):
    def __init__(self, cfg):
        self.target_h = cfg.DATASETS.TARGET.HEIGHT
        self.target_w = cfg.DATASETS.TARGET.WIDTH
        self.device = torch.device(cfg.MODEL.DEVICE)
        try:
            self.threshold = cfg.DATASETS.JLOC_THRESHOLD
        except AttributeError:
            self.threshold = 0.1
            logger.warning(f"DATASETS.JLOC_THRESHOLD not found in config. Using default value: {self.threshold}")

        self.out_dir = './out'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def __call__(self, annotations):
        targets = []
        metas = []
        for ann in annotations:
            # 检查 annotation 内容
            logger.debug(f"Annotation keys: {list(ann.keys())}")
            t, m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets), metas

    def _process_per_image(self, ann):
        junctions = ann.get('junctions', None)
        if junctions is None:
            logger.warning("Junctions not found in annotation, skipping processing")
            return self._dummy_target(ann)

        device = junctions.device
        height, width = ann['height'], ann['width']
        junc_tag = ann['juncs_tag']
        jmap = torch.zeros((height, width), device=device, dtype=torch.long)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)

        edges_positive = ann['edges_positive']
        if len(edges_positive) == 0:
            afmap = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
        else:
            lines = torch.cat((junctions[edges_positive[:, 0]], junctions[edges_positive[:, 1]]), dim=-1)
            shape_info = torch.IntTensor([[0, lines.size(0), height, width]])
            afmap, label = afm(lines, shape_info.to(device), height, width)

        xint, yint = junctions[:, 0].long(), junctions[:, 1].long()
        xint = torch.clamp(xint, 0, width - 1)
        yint = torch.clamp(yint, 0, height - 1)
        valid_indices = (xint < width) & (yint < height) & (xint >= 0) & (yint >= 0)
        if not valid_indices.all():
            logger.warning("Invalid indices detected, clamping and filtering")
            xint = xint[valid_indices]
            yint = yint[valid_indices]
            junctions = junctions[valid_indices]
            junc_tag = junc_tag[valid_indices]
            if len(xint) == 0:
                logger.warning("No valid junctions after filtering, skipping processing")
                return self._dummy_target(ann)
            else:
                off_x = junctions[:, 0] - xint.float() - 0.5
                off_y = junctions[:, 1] - yint.float() - 0.5
        else:
            off_x = junctions[:, 0] - xint.float() - 0.5
            off_y = junctions[:, 1] - yint.float() - 0.5

        # 添加日志检查 junctions 坐标
        logger.debug(f"Image {ann.get('image_id', 'unknown')}: Junctions shape: {junctions.shape}, "
                     f"xint range: [{xint.min().item()}, {xint.max().item()}], "
                     f"yint range: [{yint.min().item()}, {yint.max().item()}]")

        image = ann.get('image', None)
        if image is None:
            logger.warning("Image not found in annotation, skipping processing")
            return self._dummy_target(ann)

        # 将图像从 tensor 转换为 numpy 格式，并归一化到 [0, 255]
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        if image_np.shape[2] == 3:  # 如果是 RGB 图像，转换为灰度图
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_np.squeeze()  # 如果已经是灰度图，直接使用
        image_gray = (image_gray * 255).astype(np.uint8)  # 归一化到 [0, 255]

        # 使用 Sobel 算子计算梯度
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 提取角点位置的梯度强度
        gradient_magnitudes = np.zeros(len(xint), dtype=np.float32)
        for i in range(len(xint)):
            y, x = int(yint[i]), int(xint[i])
            gradient_magnitudes[i] = gradient_magnitude[y, x]

        # 归一化梯度强度到 [0, 1]
        max_magnitude = np.max(gradient_magnitudes) if len(gradient_magnitudes) > 0 else 1.0
        gradient_magnitudes = gradient_magnitudes / (max_magnitude + 1e-6)
        gradient_magnitudes = torch.tensor(gradient_magnitudes, dtype=torch.float32, device=device)

        # 添加日志检查 gradient_magnitudes
        logger.debug(f"Image {ann.get('image_id', 'unknown')}: Gradient magnitudes range: "
                     f"[{gradient_magnitudes.min().item():.4f}, {gradient_magnitudes.max().item():.4f}], "
                     f"non-zero count: {(gradient_magnitudes > 0).sum().item()}")

        # 动态调整阈值
        gradient_values = gradient_magnitudes.cpu().numpy()
        valid_values = gradient_values[gradient_values > 0]
        if len(valid_values) > 0:
            threshold = np.percentile(valid_values, 75)  # 使用 75 分位数作为阈值
            threshold = max(threshold, 0.1)
        else:
            threshold = self.threshold
            logger.warning("No valid gradient values, using default threshold")

        # 可视化梯度强度分布
        if 'image_id' in ann:
            plt.figure(figsize=(10, 5))
            plt.hist(gradient_values, bins=50, range=(0, 1), alpha=0.5, label='Gradient Magnitude', color='blue')
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
            plt.title(f"Gradient Magnitude Distribution - Image {ann['image_id']}")
            plt.xlabel("Gradient Magnitude")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, f"gradient_magnitude_hist_{ann['image_id']}.png"))
            plt.close()

        # 获取当前 epoch
        current_epoch = ann.get('current_epoch', 1)

        # 移除 epoch 限制，直接进行分类（临时调试）
        # 根据梯度强度区分清晰和半淹没角点
        clear_mask = (gradient_magnitudes >= threshold)
        submerged_mask = (gradient_magnitudes < threshold) & (gradient_magnitudes > 0)
        if len(xint) > 0:
            jmap[yint[clear_mask], xint[clear_mask]] = 1
            jmap[yint[submerged_mask], xint[submerged_mask]] = 2
            joff[0, yint, xint] = off_x
            joff[1, yint, xint] = off_y
        else:
            logger.warning("No valid indices for jmap assignment, using zeros")

        logger.info(f"jmap distribution for image {ann.get('image_id', 'unknown')}: "
                    f"background: {(jmap == 0).sum().item()}, "
                    f"clear: {(jmap == 1).sum().item()}, "
                    f"submerged: {(jmap == 2).sum().item()}")

        line_map_np = np.zeros((height, width), dtype=np.uint8)
        if len(edges_positive) > 0:
            for edge in edges_positive:
                start = junctions[edge[0]].cpu().numpy().astype(int)
                end = junctions[edge[1]].cpu().numpy().astype(int)
                cv2.line(line_map_np, (start[0], start[1]), (end[0], end[1]), 1, thickness=2)
        mask_np = ann['mask'].cpu().numpy()
        edges = canny(mask_np, sigma=0.5, low_threshold=0.1, high_threshold=0.2)
        edges = binary_dilation(edges, iterations=2)
        line_map_np = np.logical_or(line_map_np, edges).astype(np.float32)
        line_map = torch.tensor(line_map_np, dtype=torch.float32, device=device)[None]

        if 'image_id' in ann:
            cv2.imwrite(os.path.join(self.out_dir, f"line_map_{ann['image_id']}.png"), (line_map_np * 255).astype(np.uint8))

        meta = {
            'junc': junctions,
            'junc_index': ann['juncs_index'],
            'bbox': ann['bbox'],
        }

        mask = ann['mask'].float()
        target = {
            'jloc': jmap[None],
            'joff': joff,
            'mask': mask[None],
            'afmap': afmap[0],
            'line_map': line_map
        }
        return target, meta

    def _dummy_target(self, ann):
        """返回一个空的 target 和 meta，用于跳过无效样本"""
        device = torch.device(self.device)
        height, width = ann['height'], ann['width']
        jmap = torch.zeros((height, width), device=device, dtype=torch.long)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)
        afmap = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
        line_map = torch.zeros((1, height, width), device=device, dtype=torch.float32)
        mask = ann['mask'].float()

        meta = {
            'junc': torch.tensor([], device=device),
            'junc_index': ann['juncs_index'],
            'bbox': ann['bbox'],
        }

        target = {
            'jloc': jmap[None],
            'joff': joff,
            'mask': mask[None],
            'afmap': afmap[0],
            'line_map': line_map
        }
        return target, meta