# Copyright (c) Open-CD. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmseg.utils import SampleList
from opencd.registry import MODELS


def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            if 'gt_seg_map' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
            if 'gt_seg_map_from' in data_sample:
                gt_seg_map_from = data_sample.gt_seg_map_from.data
                del data_sample.gt_seg_map_from.data
                data_sample.gt_seg_map_from.data = F.pad(
                    gt_seg_map_from, padding_size, value=seg_pad_val)
            if 'gt_seg_map_to' in data_sample:
                gt_seg_map_to = data_sample.gt_seg_map_to.data
                del data_sample.gt_seg_map_to.data
                data_sample.gt_seg_map_to.data = F.pad(
                    gt_seg_map_to, padding_size, value=seg_pad_val)
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': 'no use',
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples


@MODELS.register_module()
class FoundationInputSegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for change detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    # def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
    #     """Perform normalization、padding and bgr2rgb conversion based on
    #     ``BaseDataPreprocessor``.

    #     Args:
    #         data (dict): data sampled from dataloader.
    #         training (bool): Whether to enable training time augmentation.

    #     Returns:
    #         Dict: Data in the same format as the model input.
    #     """
    #     data = self.cast_data(data)  # type: ignore
    #     inputs = data['inputs']
    #     data_samples = data.get('data_samples', None)
    #     # TODO: whether normalize should be after stack_batch
    #     if self.channel_conversion and inputs[0].size(0) == 6:
    #         inputs = [_input[[2, 1, 0, 5, 4, 3], ...] for _input in inputs]
    #         inputs = [_input.float() for _input in inputs]
    #         if self._enable_normalize:
    #             inputs = [(_input - self.mean) / self.std for _input in inputs]
    #     else:
    #         inputs = [_input[[2, 1, 0], ...] for _input in inputs]
    #         mean = self.mean[:3]
    #         std = self.std[:3]
    #         inputs = [_input.float() for _input in inputs]
    #         if self._enable_normalize:
    #             inputs = [(_input - mean) / std for _input in inputs]

    #     if training:
    #         assert data_samples is not None, ('During training, ',
    #                                           '`data_samples` must be define.')
    #         inputs, data_samples = stack_batch(
    #             inputs=inputs,
    #             data_samples=data_samples,
    #             size=self.size,
    #             size_divisor=self.size_divisor,
    #             pad_val=self.pad_val,
    #             seg_pad_val=self.seg_pad_val)

    #         if self.batch_augments is not None:
    #             inputs, data_samples = self.batch_augments(
    #                 inputs, data_samples)
    #     else:
    #         # assert len(inputs) == 1, (
    #         #     'Batch inference is not support currently, '
    #         #     'as the image size might be different in a batch')
    #         # pad images when testing
    #         if self.test_cfg:
    #             inputs, padded_samples = stack_batch(
    #                 inputs=inputs,
    #                 size=self.test_cfg.get('size', None),
    #                 size_divisor=self.test_cfg.get('size_divisor', None),
    #                 pad_val=self.pad_val,
    #                 seg_pad_val=self.seg_pad_val)
    #             for data_sample, pad_info in zip(data_samples, padded_samples):
    #                 data_sample.set_metainfo({**pad_info})
    #         else:
    #             inputs = torch.stack(inputs, dim=0)

    #     return dict(inputs=inputs, data_samples=data_samples)


    
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.
    
        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
    
        Returns:
            Dict: Data in the same format as the model input.
        """
        
        # 🔍 添加调试信息
        # print(f"\n🔍 DataPreprocessor.forward [开始]")
        
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        
        # # 检查输入
        # if isinstance(inputs, list):
        #     print(f"   输入是列表，长度: {len(inputs)}")
        #     for i, img_tensor in enumerate(inputs):
        #         if isinstance(img_tensor, torch.Tensor):
        #             print(f"   图像{i}: shape={img_tensor.shape}")
        
        # ⭐️⭐️⭐️ 关键修复：确保所有图像都是4通道 ⭐️⭐️⭐️
        fixed_inputs = []
        for i, img_tensor in enumerate(inputs):
            if isinstance(img_tensor, torch.Tensor):
                if len(img_tensor.shape) == 3:  # [C, H, W]
                    channels = img_tensor.shape[0]
                    
                    if channels == 4:
                        # print(f"   图像{i}: ✅ 4通道，正常")
                        fixed_inputs.append(img_tensor)
                    elif channels == 3:
                        # print(f"   图像{i}: ⚠️ 只有3通道，强制修复为4通道")
                        # 复制第3通道作为第4通道
                        channel_3 = img_tensor[2:3, :, :]
                        fixed_tensor = torch.cat([img_tensor, channel_3], dim=0)
                        # print(f"       修复后: shape={fixed_tensor.shape}")
                        fixed_inputs.append(fixed_tensor)
                    else:
                        # print(f"   图像{i}: ⚠️ {channels}通道，创建4通道tensor")
                        # 创建4通道tensor
                        height = img_tensor.shape[1]
                        width = img_tensor.shape[2]
                        fixed_tensor = torch.zeros(4, height, width,
                                                  dtype=img_tensor.dtype, device=img_tensor.device)
                        channels_to_copy = min(4, channels)
                        fixed_tensor[:channels_to_copy] = img_tensor[:channels_to_copy]
                        fixed_inputs.append(fixed_tensor)
                else:
                    fixed_inputs.append(img_tensor)
            else:
                fixed_inputs.append(img_tensor)
        
        inputs = fixed_inputs
        
        # TODO: whether normalize should be after stack_batch
        
        # ⭐️ 修改归一化逻辑，确保通道数匹配
        if self.channel_conversion:
            # 这个分支不应该执行，因为您的配置中 bgr_to_rgb=False
            # print(f"   ⚠️ channel_conversion为True，但配置中bgr_to_rgb=False")
            if inputs[0].size(0) == 6:
                inputs = [_input[[2, 1, 0, 5, 4, 3], ...] for _input in inputs]
        else:
            # print(f"   channel_conversion为False，跳过BGR->RGB转换")
            None
        
        # 转换为float
        inputs = [_input.float() for _input in inputs]
        
        # ⭐️ 执行归一化
        if self._enable_normalize:
            # print(f"   执行归一化")
            # print(f"   mean形状: {self.mean.shape}, 值: {self.mean.view(-1).tolist()}")
            # print(f"   std形状: {self.std.shape}, 值: {self.std.view(-1).tolist()}")
            
            # 确保所有图像都是4通道
            normalized_inputs = []
            for i, img_tensor in enumerate(inputs):
                if img_tensor.shape[0] == 4:
                    # 使用前4个通道的mean/std
                    if len(self.mean) >= 4:
                        mean_to_use = self.mean[:4]
                        std_to_use = self.std[:4]
                        # print(f"   图像{i}: 使用前4通道mean/std")
                        normalized_img = (img_tensor - mean_to_use) / std_to_use
                        normalized_inputs.append(normalized_img)
                    else:
                        # print(f"   ⚠️ 图像{i}: mean/std少于4个值，跳过归一化")
                        normalized_inputs.append(img_tensor)
                elif img_tensor.shape[0] == 3:
                    # print(f"   ⚠️ 图像{i}: 仍有3个通道，使用前3通道mean/std")
                    if len(self.mean) >= 3:
                        mean_to_use = self.mean[:3]
                        std_to_use = self.std[:3]
                        normalized_img = (img_tensor - mean_to_use) / std_to_use
                        # 转换为4通道
                        channel_3 = normalized_img[2:3, :, :]
                        normalized_img = torch.cat([normalized_img, channel_3], dim=0)
                        normalized_inputs.append(normalized_img)
                    else:
                        # 转换为4通道但不归一化
                        channel_3 = img_tensor[2:3, :, :]
                        normalized_img = torch.cat([img_tensor, channel_3], dim=0)
                        normalized_inputs.append(normalized_img)
                else:
                    # print(f"   ⚠️ 图像{i}: {img_tensor.shape[0]}通道，跳过归一化")
                    normalized_inputs.append(img_tensor)
            
            inputs = normalized_inputs
        
        # 🔍 检查最终输出
        # print(f"\n🔍 DataPreprocessor.forward [处理后]")
        for i, img_tensor in enumerate(inputs):
            if isinstance(img_tensor, torch.Tensor):
                # print(f"   输出图像{i}: shape={img_tensor.shape}")
                if img_tensor.shape[0] != 4:
                    # print(f"     ❌ 错误: 输出图像{i}只有{img_tensor.shape[0]}个通道!")
                    None
        
        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)
    
            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)
    
        return dict(inputs=inputs, data_samples=data_samples)
