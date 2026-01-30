# Copyright (c) Open-CD. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile_Modified(MMCV_LoadImageFromFile):

    def __init__(self, **kwargs) -> None:
         super().__init__(**kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        data_type = results['type']
        if data_type == 'only_building_label':
            try:
                filenames = [results['img_a_path']]
                # print(f"🔍 单图模式 - 加载: {results['img_a_path']}")
            except:
                filenames = [results['img_path']]
                # print(f"🔍 单图模式 - 加载: {results['img_path']}")
        else:
            filenames = [results['img_a_path'], results['img_b_path']]
            # print(f"🔍 双图模式 - 加载 A: {results['img_a_path']}, B: {results['img_b_path']}")
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
                img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)
                imgs.append(img)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # results['img'] = imgs
        # results['img_shape'] = imgs[0].shape[:2]
        # results['ori_shape'] = imgs[0].shape[:2]
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results
    
    
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