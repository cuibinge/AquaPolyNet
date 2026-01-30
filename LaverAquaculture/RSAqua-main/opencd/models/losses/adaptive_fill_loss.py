import torch
import torch.nn as nn
import cv2
import numpy as np
from torch import Tensor
from opencd.registry import MODELS


@MODELS.register_module()
class AdaptiveRectFillLoss(nn.Module):
    """
    自适应矩形填充损失
    根据预测质量动态调整填充强度
    - 低置信度区域：加强填充约束
    - 小面积区域：加强填充约束  
    - 高置信度/大面积区域：减弱填充约束
    """
    
    def __init__(self, 
                 base_weight=0.4,
                 min_area=10,
                 low_confidence_thresh=0.3,
                 small_area_thresh=0.05,
                 canny_thresh1=50,
                 canny_thresh2=150,
                 loss_type='mse',
                 debug=False):
        super(AdaptiveRectFillLoss, self).__init__()
        self.base_weight = base_weight
        self.min_area = min_area
        self.low_confidence_thresh = low_confidence_thresh
        self.small_area_thresh = small_area_thresh
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.debug = debug
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_prediction_quality(self, pred_mask):
        """
        评估预测质量
        Returns:
            confidence: 预测置信度 (0-1, 越高越好)
            area_ratio: 预测区域面积比例 (0-1)
        """
        pred_prob = torch.sigmoid(pred_mask)
        
        # 置信度：离0.5越远，置信度越高
        confidence = torch.abs(pred_prob - 0.5).mean()
        
        # 区域面积比例
        area_ratio = (pred_prob > 0.5).float().mean()
        
        return confidence, area_ratio
    
    def compute_adaptive_weight(self, confidence, area_ratio):
        """
        根据预测质量计算自适应权重
        """
        # 基础权重
        weight = self.base_weight
        
        # 低置信度时加强填充约束
        if confidence < self.low_confidence_thresh:
            weight *= 2.0
            if self.debug:
                print(f"低置信度区域，权重加倍: {confidence:.3f} -> 权重: {weight}")
        
        # 小面积区域时加强填充约束
        if area_ratio < self.small_area_thresh:
            weight *= 1.5
            if self.debug:
                print(f"小面积区域，权重增加: {area_ratio:.3f} -> 权重: {weight}")
        
        # 高置信度且大面积时减弱约束
        if confidence > 0.4 and area_ratio > 0.1:
            weight *= 0.5
            if self.debug:
                print(f"高质量预测，权重减半 -> 权重: {weight}")
        
        return weight
    
    def min_area_rect_fill(self, binary_mask):
        """
        最小外接矩形填充
        """
        filled_mask = np.zeros_like(binary_mask)
        
        # 如果掩码全为0，直接返回
        if np.sum(binary_mask) == 0:
            return filled_mask
        
        # 边缘检测
        edges = cv2.Canny(binary_mask, self.canny_thresh1, self.canny_thresh2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # 最小外接矩形
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
            loss: 自适应填充损失
        """
        if pred_mask.dim() == 4:
            pred_mask = pred_mask.squeeze(1)  # [B, H, W]
        
        batch_size = pred_mask.shape[0]
        total_loss = 0
        valid_batches = 0
        
        for i in range(batch_size):
            # 评估预测质量
            confidence, area_ratio = self.compute_prediction_quality(pred_mask[i])
            
            # 计算自适应权重
            adaptive_weight = self.compute_adaptive_weight(confidence, area_ratio)
            
            # 矩形填充
            single_pred = pred_mask[i].detach().cpu().numpy()
            pred_norm = (single_pred - single_pred.min()) / (single_pred.max() - single_pred.min() + 1e-8)
            binary_pred = (pred_norm > 0.5).astype(np.uint8)
            filled_mask = self.min_area_rect_fill(binary_pred)
            
            # 如果填充后没有变化，跳过这个样本
            if np.array_equal(binary_pred, filled_mask):
                if self.debug:
                    print("填充后无变化，跳过样本")
                continue
            
            # 计算损失
            filled_tensor = torch.from_numpy(filled_mask.astype(np.float32)).to(pred_mask.device)
            pred_tensor = torch.from_numpy(pred_norm).to(pred_mask.device)
            
            base_loss = self.loss_fn(pred_tensor, filled_tensor)
            weighted_loss = base_loss * adaptive_weight
            
            total_loss += weighted_loss
            valid_batches += 1
            
            if self.debug:
                print(f"样本 {i}: 置信度={confidence:.3f}, 面积比={area_ratio:.3f}, "
                      f"权重={adaptive_weight:.3f}, 损失={base_loss:.4f}")
        
        # 如果没有有效样本，返回0损失
        if valid_batches == 0:
            return pred_mask.new_zeros(1)
        
        return total_loss / valid_batches


@MODELS.register_module()
class ProgressiveAdaptiveFillLoss(nn.Module):
    """
    渐进式自适应填充损失
    根据训练epoch动态调整填充策略
    """
    
    def __init__(self, 
                 base_weight=0.4,
                 min_area=10,
                 total_epochs=100,
                 warmup_epochs=10,
                 canny_thresh1=50,
                 canny_thresh2=150):
        super(ProgressiveAdaptiveFillLoss, self).__init__()
        self.base_weight = base_weight
        self.min_area = min_area
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.current_epoch = 0
        
        self.mse_loss = nn.MSELoss()
    
    def set_epoch(self, epoch):
        """设置当前epoch，用于调整策略"""
        self.current_epoch = epoch
    
    def get_progressive_weight(self):
        """根据训练进度获取渐进权重"""
        if self.current_epoch < self.warmup_epochs:
            # 预热期：权重较小
            return self.base_weight * 0.3
        elif self.current_epoch < self.total_epochs * 0.7:
            # 中期：正常权重
            return self.base_weight
        else:
            # 后期：加强权重
            return self.base_weight * 1.5
    
    def min_area_rect_fill(self, binary_mask):
        """矩形填充实现"""
        filled_mask = np.zeros_like(binary_mask)
        
        if np.sum(binary_mask) == 0:
            return filled_mask
        
        edges = cv2.Canny(binary_mask, self.canny_thresh1, self.canny_thresh2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(filled_mask, [box], color=1)
        
        return filled_mask
    
    def forward(self, pred_mask):
        if pred_mask.dim() == 4:
            pred_mask = pred_mask.squeeze(1)
        
        progressive_weight = self.get_progressive_weight()
        batch_size = pred_mask.shape[0]
        total_loss = 0
        valid_batches = 0
        
        for i in range(batch_size):
            single_pred = pred_mask[i].detach().cpu().numpy()
            pred_norm = (single_pred - single_pred.min()) / (single_pred.max() - single_pred.min() + 1e-8)
            binary_pred = (pred_norm > 0.5).astype(np.uint8)
            filled_mask = self.min_area_rect_fill(binary_pred)
            
            if np.array_equal(binary_pred, filled_mask):
                continue
            
            filled_tensor = torch.from_numpy(filled_mask.astype(np.float32)).to(pred_mask.device)
            pred_tensor = torch.from_numpy(pred_norm).to(pred_mask.device)
            
            loss = self.mse_loss(pred_tensor, filled_tensor)
            total_loss += loss * progressive_weight
            valid_batches += 1
        
        if valid_batches == 0:
            return pred_mask.new_zeros(1)
        
        return total_loss / valid_batches