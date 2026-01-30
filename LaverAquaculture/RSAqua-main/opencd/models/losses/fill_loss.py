import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch import Tensor
from opencd.registry import MODELS


@MODELS.register_module()
class RectFillLoss(nn.Module):
    """
    基础矩形填充损失 - 参考RSBuilding模型中的实现
    专门用于填充前景建筑物的空洞
    """
    
    def __init__(self, 
                 weight=0.4,
                 min_area=10,
                 canny_thresh1=50,
                 canny_thresh2=150,
                 loss_type='mse'):
        super(RectFillLoss, self).__init__()
        self.weight = weight
        self.min_area = min_area
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def min_area_rect_fill(self, binary_mask):
        """
        最小外接矩形填充 - 核心填充算法
        """
        filled_mask = np.zeros_like(binary_mask)
        
        # 如果掩码全为0，直接返回
        if np.sum(binary_mask) == 0:
            return filled_mask
        
        # 边缘检测 - 提取建筑物轮廓
        edges = cv2.Canny(binary_mask, self.canny_thresh1, self.canny_thresh2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # 最小外接矩形 - 适合建筑物形状
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(filled_mask, [box], color=1)
        
        return filled_mask
    
    def forward(self, pred_mask):
        """
        Args:
            pred_mask: 预测掩码 [B, 1, H, W] 或 [B, H, W]
        Returns:
            loss: 填充损失
        """
        if pred_mask.dim() == 4:
            pred_mask = pred_mask.squeeze(1)  # [B, H, W]
        
        batch_size = pred_mask.shape[0]
        total_loss = 0
        valid_batches = 0
        
        for i in range(batch_size):
            # 获取单张预测并转换为numpy
            single_pred = pred_mask[i].detach().cpu().numpy()
            
            # 归一化到0-1范围
            pred_norm = (single_pred - single_pred.min()) / (single_pred.max() - single_pred.min() + 1e-8)
            
            # 二值化
            binary_pred = (pred_norm > 0.5).astype(np.uint8)
            
            # 矩形填充
            filled_mask = self.min_area_rect_fill(binary_pred)
            
            # 如果填充后没有变化，跳过这个样本
            if np.array_equal(binary_pred, filled_mask):
                continue
            
            # 转换回tensor并计算损失
            filled_tensor = torch.from_numpy(filled_mask.astype(np.float32)).to(pred_mask.device)
            pred_tensor = torch.from_numpy(pred_norm).to(pred_mask.device)
            
            loss = self.loss_fn(pred_tensor, filled_tensor)
            total_loss += loss
            valid_batches += 1
        
        # 如果没有有效样本，返回0损失
        if valid_batches == 0:
            return pred_mask.new_zeros(1)
        
        return (total_loss / valid_batches) * self.weight

# @MODELS.register_module()
# class AdaptiveRectFillLoss(nn.Module):
#     """
#     自适应矩形填充损失 - 备用方案
#     根据预测质量动态调整填充强度
#     """
    
#     def __init__(self, 
#                  base_weight=0.4,
#                  min_area=10,
#                  low_confidence_thresh=0.3,
#                  small_area_thresh=0.05):
#         super(AdaptiveRectFillLoss, self).__init__()
#         self.base_weight = base_weight
#         self.min_area = min_area
#         self.low_confidence_thresh = low_confidence_thresh
#         self.small_area_thresh = small_area_thresh
#         self.loss_fn = nn.MSELoss()
    
#     def compute_prediction_quality(self, pred_mask):
#         """评估预测质量"""
#         pred_prob = torch.sigmoid(pred_mask)
        
#         # 置信度：离0.5越远，置信度越高
#         confidence = torch.abs(pred_prob - 0.5).mean()
        
#         # 区域面积比例
#         area_ratio = (pred_prob > 0.5).float().mean()
        
#         return confidence, area_ratio
    
#     def min_area_rect_fill(self, binary_mask):
#         """矩形填充实现"""
#         filled_mask = np.zeros_like(binary_mask)
        
#         if np.sum(binary_mask) == 0:
#             return filled_mask
        
#         edges = cv2.Canny(binary_mask, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area < self.min_area:
#                 continue
            
#             rect = cv2.minAreaRect(contour)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             cv2.fillPoly(filled_mask, [box], color=1)
        
#         return filled_mask
    
#     def forward(self, pred_mask):
#         if pred_mask.dim() == 4:
#             pred_mask = pred_mask.squeeze(1)
        
#         batch_size = pred_mask.shape[0]
#         total_loss = 0
#         valid_batches = 0
        
#         for i in range(batch_size):
#             # 评估预测质量
#             confidence, area_ratio = self.compute_prediction_quality(pred_mask[i])
            
#             # 自适应权重：低置信度或小区域时加强约束
#             if confidence < self.low_confidence_thresh or area_ratio < self.small_area_thresh:
#                 adaptive_weight = self.base_weight * 2.0
#             else:
#                 adaptive_weight = self.base_weight
            
#             # 矩形填充
#             single_pred = pred_mask[i].detach().cpu().numpy()
#             pred_norm = (single_pred - single_pred.min()) / (single_pred.max() - single_pred.min() + 1e-8)
#             binary_pred = (pred_norm > 0.5).astype(np.uint8)
#             filled_mask = self.min_area_rect_fill(binary_pred)
            
#             if np.array_equal(binary_pred, filled_mask):
#                 continue
            
#             # 计算损失
#             filled_tensor = torch.from_numpy(filled_mask.astype(np.float32)).to(pred_mask.device)
#             pred_tensor = torch.from_numpy(pred_norm).to(pred_mask.device)
            
#             loss = self.loss_fn(pred_tensor, filled_tensor)
#             total_loss += loss * adaptive_weight
#             valid_batches += 1
        
#         if valid_batches == 0:
#             return pred_mask.new_zeros(1)
        
#         return total_loss / valid_batches