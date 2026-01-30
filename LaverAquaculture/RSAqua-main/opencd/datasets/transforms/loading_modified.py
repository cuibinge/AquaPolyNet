# Copyright (c) Open-CD. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import os
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from opencd.registry import TRANSFORMS


# @TRANSFORMS.register_module()
# class MultiImgLoadImageFromFile_Modified(MMCV_LoadImageFromFile):

#     def __init__(self, force_channels=4, target_dtype='float32', **kwargs) -> None:
#          super().__init__(**kwargs)
#          self.force_channels = force_channels
#          self.target_dtype = target_dtype  # 添加目标数据类型参数

#     def transform(self, results: dict) -> Optional[dict]:
#         """Functions to load image.

#         Args:
#             results (dict): Result dict from
#                 :class:`mmengine.dataset.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         data_type = results['type']
#         if data_type == 'only_building_label':
#             try:
#                 filenames = [results['img_a_path']]
#                 # print(f"🔍 单图模式 - 加载: {results['img_a_path']}")
#             except:
#                 filenames = [results['img_path']]
#                 # print(f"🔍 单图模式 - 加载: {results['img_path']}")
#         else:
#             filenames = [results['img_a_path'], results['img_b_path']]
#             # print(f"🔍 双图模式 - 加载 A: {results['img_a_path']}, B: {results['img_b_path']}")
        
#         imgs = []
#         try:
#             for filename in filenames:
#                 if self.file_client_args is not None:
#                     file_client = fileio.FileClient.infer_client(
#                         self.file_client_args, filename)
#                     img_bytes = file_client.get(filename)
#                 else:
#                     img_bytes = fileio.get(
#                         filename, backend_args=self.backend_args)
                
#                 # 使用rasterio读取多波段TIFF文件
#                 img = self._load_image_with_rasterio(img_bytes, filename)

#                 # 转换数据类型
#                 if img.dtype != np.dtype(self.target_dtype):
#                     if img.dtype == np.float64:
#                         # 64位转32位
#                         img = img.astype(np.float32)
#                         print(f"Converted {filename} from float64 to float32")
#                     elif img.dtype == np.uint16:
#                         # 16位转32位浮点
#                         img = img.astype(np.float32) / 65535.0
#                         print(f"Converted {filename} from uint16 to float32")
#                     else:
#                         img = img.astype(np.float32)
                        
#                 if self.to_float32:
#                     img = img.astype(np.float32)
#                 imgs.append(img)
                
#         except Exception as e:
#             if self.ignore_empty:
#                 return None
#             else:
#                 raise e
                
#         results['img'] = imgs
#         results['img_shape'] = imgs[0].shape[:2]
#         results['ori_shape'] = imgs[0].shape[:2]
#         return results

@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile_Modified(MMCV_LoadImageFromFile):

    def __init__(self, force_channels=4, target_dtype='float32', **kwargs) -> None:
         super().__init__(**kwargs)
         self.force_channels = force_channels
         self.target_dtype = target_dtype
         self.debug_count = 0  # 添加调试计数器

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image."""
        
        data_type = results['type']
        if data_type == 'only_building_label':
            try:
                filenames = [results['img_a_path']]
            except:
                filenames = [results['img_path']]
        else:
            filenames = [results['img_a_path'], results['img_b_path']]
        
        imgs = []
        try:
            for filename in filenames:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                
                # 使用rasterio读取多波段TIFF文件
                img = self._load_image_with_rasterio(img_bytes, filename)

                # 🔍 添加调试信息
                self.debug_count += 1
                # if self.debug_count <= 10:  # 只打印前10次
                #     print(f"\n🔍 DEBUG [加载图像 {self.debug_count}]: {os.path.basename(filename)}")
                #     print(f"   原始形状: {img.shape}")
                #     print(f"   原始dtype: {img.dtype}")
                
                # 转换数据类型
                if img.dtype != np.dtype(self.target_dtype):
                    if img.dtype == np.float64:
                        # 64位转32位
                        img = img.astype(np.float32)
                        if self.debug_count <= 10:
                            print(f"   转换: float64 → float32")
                    elif img.dtype == np.uint16:
                        # 16位转32位浮点
                        img = img.astype(np.float32) / 65535.0
                        if self.debug_count <= 10:
                            print(f"   转换: uint16 → float32 (/65535)")
                    else:
                        img = img.astype(np.float32)
                        if self.debug_count <= 10:
                            print(f"   转换: {img.dtype} → float32")
                            
                if self.to_float32:
                    img = img.astype(np.float32)
                    if self.debug_count <= 10:
                        print(f"   应用to_float32: {img.dtype}")
                
                # 🔍 检查转换后
                # if self.debug_count <= 10:
                #     print(f"   转换后形状: {img.shape}")
                #     print(f"   转换后dtype: {img.dtype}")
                #     if len(img.shape) == 3:
                #         print(f"   通道数: {img.shape[2]}")
                #         # 检查每个通道的范围
                #         print(f"   各通道范围:")
                #         for i in range(min(4, img.shape[2])):
                #             channel = img[:, :, i]
                #             print(f"     通道{i}: [{channel.min():.3f}, {channel.max():.3f}]")
                
                imgs.append(img)
                
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
                
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        
        # # 🔍 最终检查
        # if self.debug_count <= 10:
        #     print(f"\n🔍 DEBUG [加载器输出]:")
        #     for i, img in enumerate(imgs):
        #         print(f"   输出图像{i}: 形状={img.shape}, dtype={img.dtype}")
        #         if len(img.shape) == 3:
        #             print(f"     通道数: {img.shape[2]}")
        
        # 在方法结束前添加通道数修复
        # 在方法结束前添加最终验证
        fixed_imgs = []
        for i, img in enumerate(results['img']):
            if len(img.shape) == 3:
                if img.shape[2] != 4:
                    # 强制修复
                    if img.shape[2] == 3:
                        channel_3 = img[:, :, 2:3]
                        img = np.concatenate([img, channel_3], axis=2)
                    elif img.shape[2] < 3:
                        h, w = img.shape[:2]
                        img_4ch = np.zeros((h, w, 4), dtype=img.dtype)
                        img_4ch[:, :, :img.shape[2]] = img
                        for ch in range(img.shape[2], 4):
                            if img.shape[2] > 0:
                                img_4ch[:, :, ch] = img[:, :, -1]
                        img = img_4ch
                    else:
                        img = img[:, :, :4]
                else:
                    None
            fixed_imgs.append(img)
        
        results['img'] = fixed_imgs
        return results

    # _load_image_with_rasterio 和 _adjust_channels 方法保持不变

    def _load_image_with_rasterio(self, img_bytes, filename):
        """使用rasterio加载多波段图像并调整通道数"""
        try:
            import rasterio
            from io import BytesIO
            
            with rasterio.open(BytesIO(img_bytes)) as src:
                img_data = src.read()  # 读取所有波段 (C, H, W)
                
                # 如果只有一个波段，扩展为2D
                if len(img_data.shape) == 2:
                    img_data = np.expand_dims(img_data, axis=0)
                
                # 转换为 (H, W, C) 格式
                img = np.transpose(img_data, (1, 2, 0))
                
                # print(f"Loaded image {filename}: original shape {img.shape}")
                
                # 调整通道数到目标通道数
                img = self._adjust_channels(img, self.force_channels)
                
                return img
                
        except Exception as e:
            print(f"Warning: Failed to load {filename} with rasterio: {e}")
            print("Falling back to mmcv imfrombytes...")
            
            # 回退到原来的mmcv方式
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            
            # 调整通道数
            img = self._adjust_channels(img, self.force_channels)
            
            return img

    def _adjust_channels(self, img, target_channels):
        """调整图像通道数到目标通道数"""
        # 确保图像是3维的 (H, W, C)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            
        current_channels = img.shape[2] if len(img.shape) == 3 else 1
        
        if current_channels == target_channels:
            return img
            
        if current_channels > target_channels:
            # 裁剪多余的通道
            print(f"Warning: Cropping image from {current_channels} to {target_channels} channels")
            return img[:, :, :target_channels]
        else:
            # 填充不足的通道
            print(f"Warning: Padding image from {current_channels} to {target_channels} channels")
            padded_img = np.zeros((img.shape[0], img.shape[1], target_channels), dtype=img.dtype)
            padded_img[:, :, :current_channels] = img
            
            # 用最后一个通道填充剩余通道
            for i in range(current_channels, target_channels):
                if current_channels > 0:
                    padded_img[:, :, i] = img[:, :, -1]  # 复制最后一个通道
                else:
                    padded_img[:, :, i] = 0  # 如果是单通道，填充0
            
            return padded_img

# @TRANSFORMS.register_module()
# class MultiImgLoadImageFromFile_Modified(MMCV_LoadImageFromFile):

#     def __init__(self, **kwargs) -> None:
#          super().__init__(**kwargs)

#     def transform(self, results: dict) -> Optional[dict]:
#         """Functions to load image.

#         Args:
#             results (dict): Result dict from
#                 :class:`mmengine.dataset.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         data_type = results['type']
#         if data_type == 'only_building_label':
#             try:
#                 filenames = [results['img_a_path']]
#                 # print(f"🔍 单图模式 - 加载: {results['img_a_path']}")
#             except:
#                 filenames = [results['img_path']]
#                 # print(f"🔍 单图模式 - 加载: {results['img_path']}")
#         else:
#             filenames = [results['img_a_path'], results['img_b_path']]
#             # print(f"🔍 双图模式 - 加载 A: {results['img_a_path']}, B: {results['img_b_path']}")
#         imgs = []
#         try:
#             for filename in filenames:
#                 if self.file_client_args is not None:
#                     file_client = fileio.FileClient.infer_client(
#                         self.file_client_args, filename)
#                     img_bytes = file_client.get(filename)
#                 else:
#                     img_bytes = fileio.get(
#                         filename, backend_args=self.backend_args)
#                 img = mmcv.imfrombytes(
#                 img_bytes, flag=self.color_type, backend=self.imdecode_backend)
#                 if self.to_float32:
#                     img = img.astype(np.float32)
#                 imgs.append(img)
#         except Exception as e:
#             if self.ignore_empty:
#                 return None
#             else:
#                 raise e
#         # results['img'] = imgs
#         # results['img_shape'] = imgs[0].shape[:2]
#         # results['ori_shape'] = imgs[0].shape[:2]
#         results['img'] = imgs
#         results['img_shape'] = imgs[0].shape[:2]
#         results['ori_shape'] = imgs[0].shape[:2]
#         return results
    
    
@TRANSFORMS.register_module()
class MultiImgMultiAnnLoadAnnotations_Modified(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_semantic_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_semantic_zero_label = reduce_semantic_zero_label
        if self.reduce_semantic_zero_label is not None:
            warnings.warn('`reduce_semantic_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_semantic_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend
        # 初始化默认黑图和相关属性
        self._default_black_gt = None
        self._missing_files_logged = set()

    def _create_default_black_gt(self, shape=(512, 512)):
        """创建默认的全黑GT图像"""
        if self._default_black_gt is None or self._default_black_gt.shape != shape:
            self._default_black_gt = np.zeros(shape, dtype=np.uint8)
        return self._default_black_gt.copy()

    def _load_seg_with_fallback(self, seg_path, results, seg_type='default'):
        """加载分割图，如果文件不存在则使用默认黑图"""
        try:
            img_bytes = fileio.get(seg_path, backend_args=self.backend_args)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='grayscale', 
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            
            # 应用二值化阈值
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
            gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            
            return gt_semantic_seg
            
        except FileNotFoundError:
            # 文件不存在，使用默认黑图
            if seg_path not in self._missing_files_logged:
                print(f"Warning: {seg_path} not found, using default black GT")
                self._missing_files_logged.add(seg_path)
            
            # 根据输入图像尺寸创建黑图，如果没有输入图像信息则使用默认尺寸
            if 'img' in results and results['img'] is not None:
                img_shape = results['img'][0].shape[:2]  # 取第一张图像的尺寸
                black_gt = self._create_default_black_gt(img_shape)
            else:
                black_gt = self._create_default_black_gt()
            
            # 标记使用了假数据
            results['is_fake_gt'] = True
            return black_gt

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results['type'] == 'only_cd_label':
            # 加载变化检测标签
            gt_semantic_seg = self._load_seg_with_fallback(results['label_cd'], results)
            results['gt_seg_map'] = gt_semantic_seg
            results['seg_fields'].extend(['gt_seg_map',])

        elif results['type'] == 'only_building_label':
            # 加载单时相建筑标签
            gt_semantic_seg_from = self._load_seg_with_fallback(results['label_a'], results)
            results['gt_seg_map_from'] = gt_semantic_seg_from
            results['seg_fields'].extend(['gt_seg_map_from'])

        else:
            # 加载多时相的所有标签
            gt_semantic_seg = self._load_seg_with_fallback(results['label_cd'], results)
            gt_semantic_seg_from = self._load_seg_with_fallback(results['label_a'], results)
            gt_semantic_seg_to = self._load_seg_with_fallback(results['label_b'], results)
            
            results['gt_seg_map'] = gt_semantic_seg
            results['gt_seg_map_from'] = gt_semantic_seg_from
            results['gt_seg_map_to'] = gt_semantic_seg_to
            results['seg_fields'].extend(['gt_seg_map', 'gt_seg_map_from', 'gt_seg_map_to'])
        

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_semantic_zero_label={self.reduce_semantic_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str