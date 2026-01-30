# from typing import Sequence, Optional, Union  # 添加必要的类型导入
# import mmcv
# import numpy as np
# from mmengine.dist import master_only
#
# from mmseg.structures import SegDataSample
# from mmseg.visualization import SegLocalVisualizer
# from opencd.registry import VISUALIZERS
#
#
# @VISUALIZERS.register_module()
# class CDLocalVisualizer(SegLocalVisualizer):
#     """Change Detection Local Visualizer. """
#     print("9")
#     @master_only
#     def add_datasample(
#             self,
#             name: str,
#             image: np.ndarray,
#             image_from_to: Sequence[np.array],
#             data_sample: Optional[SegDataSample] = None,
#             draw_gt: bool = True,
#             draw_pred: bool = True,
#             show: bool = False,
#             wait_time: float = 0,
#             # TODO: Supported in mmengine's Viusalizer.
#             out_file: Optional[str] = None,
#             step: int = 0) -> None:
#         """Draw datasample and save to all backends.
#
#         - If GT and prediction are plotted at the same time, they are
#         displayed in a stitched image where the left image is the
#         ground truth and the right image is the prediction.
#         - If ``show`` is True, all storage backends are ignored, and
#         the images will be displayed in a local window.
#         - If ``out_file`` is specified, the drawn image will be
#         saved to ``out_file``. it is usually used when the display
#         is not available.
#
#         Args:
#             name (str): The image identifier.
#             image (np.ndarray): The image to draw.
#             image_from_to (Sequence[np.array]): The image pairs to draw.
#             gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
#                 Defaults to None.
#             pred_sample (:obj:`SegDataSample`, optional): Prediction
#                 SegDataSample. Defaults to None.
#             draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
#             draw_pred (bool): Whether to draw Prediction SegDataSample.
#                 Defaults to True.
#             show (bool): Whether to display the drawn image. Default to False.
#             wait_time (float): The interval of show (s). Defaults to 0.
#             out_file (str): Path to output file. Defaults to None.
#             step (int): Global step value to record. Defaults to 0.
#         """
#         # === 新增：从 test.txt 加载文件名映射 ===
#         import os
#         print("10")
#         # 保存原始名称
#         original_name = name
#         print("original_name:",original_name)
#         # 如果名称是 batchX_outputX 格式，从 test.txt 获取真实文件名
#         if 'batch' in name and 'output' in name:
#             print("1")
#             try:
#                 print("2")
#                 # 解析批次和输出索引
#                 parts = name.split('_')
#                 batch_idx = int(parts[0].replace('batch', ''))
#                 output_idx = int(parts[1].replace('output', ''))
#
#                 # 计算全局索引 (假设每个批次2个样本)
#                 global_idx = batch_idx * 2 + output_idx
#
#                 # 加载 test.txt 文件
#                 test_txt_path = "data_dir/whubuilding/test.txt"
#                 if os.path.exists(test_txt_path):
#                     print("3")
#                     with open(test_txt_path, 'r') as f:
#                         print("4")
#                         lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
#
#                     print(f"🔍 从 {test_txt_path} 加载了 {len(lines)} 行")
#
#                     if global_idx < len(lines):
#                         line = lines[global_idx]
#                         print(f"🔍 原始行内容: {repr(line)}")
#
#                         # 从行中提取纯文件名
#                         # 方法1: 从路径中提取
#                         if '/' in line:
#                             # 假设格式: path/to/image/0_x_y.tif ** ** path/to/label/0_x_y.tif **
#                             parts = line.split()
#                             for part in parts:
#                                 if '.tif' in part and '/' in part:
#                                     # 从路径中提取文件名（不含扩展名）
#                                     filename = os.path.splitext(os.path.basename(part))[0]
#                                     name = filename
#                                     print(f"🎯 从路径提取文件名: {filename}")
#                                     break
#                         else:
#                             # 方法2: 直接使用第一列
#                             name = line.split()[0] if line.split() else line
#
#                         print(f"🎯 映射文件名: {original_name} -> {name} (索引: {global_idx})")
#                     else:
#                         print(f"⚠️ 索引 {global_idx} 超出范围")
#                 else:
#                     print("5")
#                     print(f"⚠️ test.txt 文件不存在: {test_txt_path}")
#
#             except Exception as e:
#                 print("6")
#                 print(f"⚠️ 文件名映射失败: {e}")
#
#         print(f"🎯 最终使用的文件名: {name}")
#
#         exist_img_from_to = True if len(image_from_to) > 0 else False
#         if exist_img_from_to:
#             assert len(image_from_to) == 2, '`image_from_to` contains `from` ' 'and `to` images'
#
#         classes = self.dataset_meta.get('classes', None)
#         palette = self.dataset_meta.get('palette', None)
#         semantic_classes = self.dataset_meta.get('semantic_classes', None)
#         semantic_palette = self.dataset_meta.get('semantic_palette', None)
#
#         gt_img_data = None
#         gt_img_data_from = None
#         gt_img_data_to = None
#         pred_img_data = None
#         pred_img_data_from = None
#         pred_img_data_to = None
#
#         drawn_img_from = None
#         drawn_img_to = None
#
#         if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
#             gt_img_data = image
#             assert classes is not None, 'class information is ' 'not provided when ' 'visualizing change ' 'deteaction results.'
#             gt_img_data = self._draw_sem_seg(gt_img_data,
#                                              data_sample.gt_sem_seg, classes,
#                                              palette)
#         if draw_gt and data_sample is not None and 'gt_sem_seg_from' in data_sample \
#             and 'gt_sem_seg_to' in data_sample:
#             if exist_img_from_to:
#                 gt_img_data_from = image_from_to[0]
#                 gt_img_data_to = image_from_to[1]
#             else:
#                 gt_img_data_from = np.zeros_like(image)
#                 gt_img_data_to = np.zeros_like(image)
#             assert semantic_classes is not None, 'class information is ' 'not provided when ' 'visualizing change ' 'deteaction results.'
#             gt_img_data_from = self._draw_sem_seg(gt_img_data_from,
#                                              data_sample.gt_sem_seg_from, semantic_classes,
#                                              semantic_palette)
#             gt_img_data_to = self._draw_sem_seg(gt_img_data_to,
#                                              data_sample.gt_sem_seg_to, semantic_classes,
#                                              semantic_palette)
#
#         if (draw_pred and data_sample is not None
#                 and 'pred_sem_seg' in data_sample):
#             pred_img_data = image
#             assert classes is not None, 'class information is ' \
#                                         'not provided when ' \
#                                         'visualizing semantic ' \
#                                         'segmentation results.'
#             pred_img_data = self._draw_sem_seg(pred_img_data,
#                                                data_sample.pred_sem_seg,
#                                                classes, palette)
#
#         if (draw_pred and data_sample is not None and 'pred_sem_seg_from' in data_sample \
#             and 'pred_sem_seg_to' in data_sample):
#             if exist_img_from_to:
#                 pred_img_data_from = image_from_to[0]
#                 pred_img_data_to = image_from_to[1]
#             else:
#                 pred_img_data_from = np.zeros_like(image)
#                 pred_img_data_to = np.zeros_like(image)
#             assert semantic_classes is not None, 'class information is ' \
#                                         'not provided when ' \
#                                         'visualizing change ' \
#                                         'deteaction results.'
#             pred_img_data_from = self._draw_sem_seg(pred_img_data_from,
#                                              data_sample.pred_sem_seg_from, semantic_classes,
#                                              semantic_palette)
#             pred_img_data_to = self._draw_sem_seg(pred_img_data_to,
#                                              data_sample.pred_sem_seg_to, semantic_classes,
#                                              semantic_palette)
#
#         if gt_img_data is not None and pred_img_data is not None:
#             drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
#         elif gt_img_data is not None:
#             drawn_img = gt_img_data
#         else:
#             drawn_img = pred_img_data
#
#         if gt_img_data_from is not None and pred_img_data_from is not None:
#             drawn_img_from = np.concatenate((gt_img_data_from, pred_img_data_from), axis=1)
#         elif gt_img_data_from is not None:
#             drawn_img_from = gt_img_data_from
#         else:
#             drawn_img_from = pred_img_data_from
#
#         if gt_img_data_to is not None and pred_img_data_to is not None:
#             drawn_img_to = np.concatenate((gt_img_data_to, pred_img_data_to), axis=1)
#         elif gt_img_data_to is not None:
#             drawn_img_to = gt_img_data_to
#         else:
#             drawn_img_to = pred_img_data_to
#
#         if show:
#             if drawn_img_from is not None and drawn_img_to is not None:
#                 drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
#                 self.show(drawn_img_cat, win_name=name, wait_time=wait_time)
#             else:
#                 self.show(drawn_img, win_name=name, wait_time=wait_time)
#
#         if out_file is not None:
#             if drawn_img_from is not None and drawn_img_to is not None:
#                 drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
#                 mmcv.imwrite(mmcv.bgr2rgb(drawn_img_cat), out_file)
#             else:
#                 mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
#         else:
#             self.add_image(name, drawn_img, drawn_img_from, drawn_img_to, step)
#
#     @master_only
#     def add_image(self, name: str,
#                   image: np.ndarray,
#                   image_from: np.ndarray = None,
#                   image_to: np.ndarray = None,
#                   step: int = 0) -> None:
#         """Record the image.
#
#         Args:
#             name (str): The image identifier.
#             image (np.ndarray, optional): The image to be saved. The format
#                 should be RGB. Defaults to None.
#             step (int): Global step value to record. Defaults to 0.
#         """
#         for vis_backend in self._vis_backends.values():
#             vis_backend.add_image(name, image, image_from, image_to, step)  # type: ignore
#
#     @master_only
#     def set_image(self, image: np.ndarray) -> None:
#         """Set the image to draw.
#
#         Args:
#             image (np.ndarray): The image to draw.
#         """
#         assert image is not None
#         image = image.astype('uint8')
#         self._image = image
#         self.width, self.height = image.shape[1], image.shape[0]
#         # print(image.shape)
#         self._default_font_size = max(
#             np.sqrt(self.height * self.width) // 90, 10)
#
#         self.fig_save.set_size_inches(  # type: ignore
#             self.width / self.dpi, self.height / self.dpi)
#         # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
#         self.ax_save.cla()
#         self.ax_save.axis(False)
#         self.ax_save.imshow(
#             image,
#             extent=(0, self.width, self.height, 0),
#             interpolation='none')
#

from typing import Sequence, Optional, Union  # 添加必要的类型导入
import mmcv
import numpy as np
from mmengine.dist import master_only

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from opencd.registry import VISUALIZERS


@VISUALIZERS.register_module()
class CDLocalVisualizer(SegLocalVisualizer):
    """Change Detection Local Visualizer. """

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            image_from_to: Sequence[np.array],
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            image_from_to (Sequence[np.array]): The image pairs to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        # === 新增：从 test.txt 加载文件名映射 ===
        import os

        # 保存原始名称
        original_name = name

        # 如果名称是 batchX_outputX 格式，从 test.txt 获取真实文件名
        if 'batch' in name and 'output' in name:
            try:
                # 解析批次和输出索引
                parts = name.split('_')
                batch_idx = int(parts[0].replace('batch', ''))
                output_idx = int(parts[1].replace('output', ''))

                # 计算全局索引 (假设每个批次2个样本)
                global_idx = batch_idx * 2 + output_idx

                # 加载 test.txt 文件
                test_txt_path = "data_dir/whubuilding/test.txt"
                if os.path.exists(test_txt_path):
                    with open(test_txt_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

                    print(f"🔍 从 {test_txt_path} 加载了 {len(lines)} 行")

                    if global_idx < len(lines):
                        line = lines[global_idx]
                        print(f"🔍 原始行内容: {repr(line)}")

                        # 从行中提取纯文件名
                        # 方法1: 从路径中提取
                        if '/' in line:
                            # 假设格式: path/to/image/0_x_y.tif ** ** path/to/label/0_x_y.tif **
                            parts = line.split()
                            for part in parts:
                                if '.tif' in part and '/' in part:
                                    # 从路径中提取文件名（不含扩展名）
                                    filename = os.path.splitext(os.path.basename(part))[0]
                                    name = filename
                                    print(f"🎯 从路径提取文件名: {filename}")
                                    break
                        else:
                            # 方法2: 直接使用第一列
                            name = line.split()[0] if line.split() else line

                        print(f"🎯 映射文件名: {original_name} -> {name} (索引: {global_idx})")
                    else:
                        print(f"⚠️ 索引 {global_idx} 超出范围")
                else:
                    print(f"⚠️ test.txt 文件不存在: {test_txt_path}")

            except Exception as e:
                print(f"⚠️ 文件名映射失败: {e}")

        print(f"🎯 最终使用的文件名: {name}")

        exist_img_from_to = True if len(image_from_to) > 0 else False
        if exist_img_from_to:
            assert len(image_from_to) == 2, '`image_from_to` contains `from` ' \
                                            'and `to` images'

        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)
        semantic_classes = self.dataset_meta.get('semantic_classes', None)
        semantic_palette = self.dataset_meta.get('semantic_palette', None)

        gt_img_data = None
        gt_img_data_from = None
        gt_img_data_to = None
        pred_img_data = None
        pred_img_data_from = None
        pred_img_data_to = None

        drawn_img_from = None
        drawn_img_to = None

        if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
            gt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            gt_img_data = self._draw_sem_seg(gt_img_data,
                                             data_sample.gt_sem_seg, classes,
                                             palette)
        if draw_gt and data_sample is not None and 'gt_sem_seg_from' in data_sample \
                and 'gt_sem_seg_to' in data_sample:
            if exist_img_from_to:
                gt_img_data_from = image_from_to[0]
                gt_img_data_to = image_from_to[1]
            else:
                gt_img_data_from = np.zeros_like(image)
                gt_img_data_to = np.zeros_like(image)
            assert semantic_classes is not None, 'class information is ' \
                                                 'not provided when ' \
                                                 'visualizing change ' \
                                                 'deteaction results.'
            gt_img_data_from = self._draw_sem_seg(gt_img_data_from,
                                                  data_sample.gt_sem_seg_from, semantic_classes,
                                                  semantic_palette)
            gt_img_data_to = self._draw_sem_seg(gt_img_data_to,
                                                data_sample.gt_sem_seg_to, semantic_classes,
                                                semantic_palette)

        if (draw_pred and data_sample is not None
                and 'pred_sem_seg' in data_sample):
            pred_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_img_data = self._draw_sem_seg(pred_img_data,
                                               data_sample.pred_sem_seg,
                                               classes, palette)

        if (draw_pred and data_sample is not None and 'pred_sem_seg_from' in data_sample \
                and 'pred_sem_seg_to' in data_sample):
            if exist_img_from_to:
                pred_img_data_from = image_from_to[0]
                pred_img_data_to = image_from_to[1]
            else:
                pred_img_data_from = np.zeros_like(image)
                pred_img_data_to = np.zeros_like(image)
            assert semantic_classes is not None, 'class information is ' \
                                                 'not provided when ' \
                                                 'visualizing change ' \
                                                 'deteaction results.'
            pred_img_data_from = self._draw_sem_seg(pred_img_data_from,
                                                    data_sample.pred_sem_seg_from, semantic_classes,
                                                    semantic_palette)
            pred_img_data_to = self._draw_sem_seg(pred_img_data_to,
                                                  data_sample.pred_sem_seg_to, semantic_classes,
                                                  semantic_palette)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if gt_img_data_from is not None and pred_img_data_from is not None:
            drawn_img_from = np.concatenate((gt_img_data_from, pred_img_data_from), axis=1)
        elif gt_img_data_from is not None:
            drawn_img_from = gt_img_data_from
        else:
            drawn_img_from = pred_img_data_from

        if gt_img_data_to is not None and pred_img_data_to is not None:
            drawn_img_to = np.concatenate((gt_img_data_to, pred_img_data_to), axis=1)
        elif gt_img_data_to is not None:
            drawn_img_to = gt_img_data_to
        else:
            drawn_img_to = pred_img_data_to

        if show:
            if drawn_img_from is not None and drawn_img_to is not None:
                drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
                self.show(drawn_img_cat, win_name=name, wait_time=wait_time)
            else:
                self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            if drawn_img_from is not None and drawn_img_to is not None:
                drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
                mmcv.imwrite(mmcv.bgr2rgb(drawn_img_cat), out_file)
            else:
                mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, drawn_img_from, drawn_img_to, step)

    @master_only
    def add_image(self, name: str,
                  image: np.ndarray,
                  image_from: np.ndarray = None,
                  image_to: np.ndarray = None,
                  step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, image_from, image_to, step)  # type: ignore

    @master_only
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        # print(image.shape)
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        self.fig_save.set_size_inches(  # type: ignore
            self.width / self.dpi, self.height / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')