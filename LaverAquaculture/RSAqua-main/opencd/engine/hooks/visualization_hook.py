# # Copyright (c) Open-CD. All rights reserved.
# import os.path as osp
# import warnings
# from typing import Optional, Sequence
#
# import mmcv
# import mmengine.fileio as fileio
# import numpy as np
# from mmengine.runner import Runner
#
# from mmseg.engine import SegVisualizationHook
# from mmseg.structures import SegDataSample
# from opencd.registry import HOOKS
# from opencd.visualization import CDLocalVisualizer
#
#
# @HOOKS.register_module()
# class CDVisualizationHook(SegVisualizationHook):
#     """Change Detection Visualization Hook. Used to visualize validation and
#     testing process prediction results.
#
#     Args:
#         img_shape (tuple): if img_shape is given and `draw_on_from_to_img` is
#             False, the original images will not be read.
#         draw_on_from_to_img (bool): whether to draw semantic prediction results
#             on the original images. If it is False, it means that drawing on
#             the black board. Defaults to False.
#
#     """
#
#     def __init__(self,
#                  img_shape: tuple = None,
#                  draw_on_from_to_img: bool = False,
#                  draw: bool = False,
#                  interval: int = 50,
#                  show: bool = False,
#                  wait_time: float = 0.,
#                  backend_args: Optional[dict] = None):
#         # 必须调用父类的 __init__，传递所有必要参数
#         super().__init__(
#             draw=draw,
#             interval=interval,
#             show=show,
#             wait_time=wait_time,
#             backend_args=backend_args
#         )
#
#         self.img_shape = img_shape
#         self.draw_on_from_to_img = draw_on_from_to_img
#         if self.draw_on_from_to_img:
#             warnings.warn('`draw_on_from_to_img` works only in '
#                           'semantic change detection.')
#         # self._visualizer: CDLocalVisualizer = CDLocalVisualizer.get_current_instance()
#         self._visualizer = CDLocalVisualizer()
#         self.interval = interval
#         self.show = show
#         if self.show:
#             # No need to think about vis backends.
#             self._visualizer._vis_backends = {}
#             warnings.warn('The show is True, it means that only '
#                           'the prediction results are visualized '
#                           'without storing data, so vis_backends '
#                           'needs to be excluded.')
#
#         self.wait_time = wait_time
#         self.backend_args = backend_args.copy() if backend_args else None
#         self.draw = draw
#         if not self.draw:
#             warnings.warn('The draw is False, it means that the '
#                           'hook for visualization will not take '
#                           'effect. The results will NOT be '
#                           'visualized or stored.')
#
#     def _after_iter(self,
#                     runner: Runner,
#                     batch_idx: int,
#                     data_batch: dict,
#                     outputs: Sequence[SegDataSample],
#                     mode: str = 'val') -> None:
#         """Run after every ``self.interval`` validation iterations.
#
#         Args:
#             runner (:obj:`Runner`): The runner of the validation process.
#             batch_idx (int): The index of the current batch in the val loop.
#             data_batch (dict): Data from dataloader.
#             outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
#             mode (str): mode (str): Current mode of runner. Defaults to 'val'.
#         """
#         # 初始化统计信息
#         if not hasattr(self, 'total_batches'):
#             self.total_batches = 0
#             self.total_outputs = 0
#             self.total_visualized = 0
#
#         self.total_batches += 1
#         self.total_outputs += len(outputs) if outputs else 0
#
#         # print(f"=== 可视化钩子调用 ===")
#         # print(f"模式: {mode}, 批次: {batch_idx}, 间隔: {self.interval}")
#         # print(f"当前批次输出数量: {len(outputs) if outputs else 0}")
#         # print(f"累计批次: {self.total_batches}, 累计输出: {self.total_outputs}")
#
#         if self.draw is False or mode == 'train':
#             return
#
#         if self.every_n_inner_iters(batch_idx, self.interval):
#             print(f"✓ 执行可视化: batch_idx={batch_idx} 满足间隔条件")
#             batch_visualized = 0
#
#             for i, output in enumerate(outputs):
#                 # 使用原始图像文件名作为预测结果文件名
#                 if hasattr(output, "img_path") and output.img_path:
#                     if isinstance(output.img_path, (list, tuple)):
#                         # 对于变化检测，使用第一张图像的文件名
#                         original_img_path = output.img_path[0]
#                     else:
#                         original_img_path = output.img_path
#
#                     # 提取原始文件名（不含路径和扩展名）
#                     original_filename = osp.splitext(osp.basename(original_img_path))[0]
#                     # 使用原始文件名作为窗口名
#                     window_name = original_filename
#                     print(f"  使用原始文件名: {original_filename}")
#
#                     # 设置图像来源
#                     img_from_to = []
#                     if self.draw_on_from_to_img:
#                         # 加载原始图像对
#                         for _img_path in output.img_path:
#                             try:
#                                 _img_bytes = fileio.get(_img_path, backend_args=self.backend_args)
#                                 _img = mmcv.imfrombytes(_img_bytes, channel_order='rgb')
#                                 img_from_to.append(_img)
#                                 print(f"    加载图像: {osp.basename(_img_path)}")
#                             except Exception as e:
#                                 print(f"    加载图像失败 {_img_path}: {e}")
#                                 # 创建灰色背景作为备用
#                                 if self.img_shape is not None:
#                                     img_from_to.append(np.ones(self.img_shape, dtype=np.uint8) * 128)
#                                 else:
#                                     img_from_to.append(np.ones((512, 512, 3), dtype=np.uint8) * 128)
#                 else:
#                     # 备用方案
#                     window_name = f"batch{batch_idx}_output{i}"
#                     img_from_to = []
#                     print(f"  使用备用文件名: {window_name}")
#
#                 # 创建背景图像
#                 if img_from_to and len(img_from_to) > 0:
#                     img = img_from_to[0]  # 使用第一张输入图像作为背景
#                 else:
#                     if self.img_shape is not None:
#                         img = np.ones(self.img_shape, dtype=np.uint8) * 128
#                     else:
#                         img = np.ones((512, 512, 3), dtype=np.uint8) * 128
#                 print("6")
#                 # 调用可视化器
#                 try:
#                     print("7")
#                     self._visualizer.add_datasample(
#                         window_name,
#                         img,
#                         img_from_to,
#                         data_sample=output,
#                         show=self.show,
#                         wait_time=self.wait_time,
#                         step=runner.iter,
#                         draw_gt=False)
#                     batch_visualized += 1
#                     self.total_visualized += 1
#                     print(f"  ✓ 成功可视化: {window_name}")
#                 except Exception as e:
#                     print("8")
#                     print(f"  ✗ 可视化失败: {window_name}, 错误: {e}")
#
#             print(f"本批次可视化数量: {batch_visualized}")
#             print(f"累计可视化总数: {self.total_visualized}")
#
#         else:
#             print(f"✗ 跳过可视化: batch_idx={batch_idx} 不满足间隔条件")
#
#     def after_test_epoch(self, runner, metrics=None):
#         """在测试结束后打印统计摘要"""
#         print("\n" + "=" * 50)
#         print("可视化统计摘要")
#         print("=" * 50)
#         print(f"总处理批次: {self.total_batches}")
#         print(f"总输出样本: {self.total_outputs}")
#         print(f"总可视化样本: {self.total_visualized}")
#
#         if hasattr(self, 'total_outputs') and self.total_outputs > 0:
#             visualization_ratio = (self.total_visualized / self.total_outputs) * 100
#             print(f"可视化比例: {visualization_ratio:.1f}%")
#
#         print("=" * 50)

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

#
# # Copyright (c) Open-CD. All rights reserved.
# import os.path as osp
# import warnings
# from typing import Optional, Sequence
#
# import mmcv
# import mmengine.fileio as fileio
# import numpy as np
# from mmengine.runner import Runner
#
# from mmseg.engine import SegVisualizationHook
# from mmseg.structures import SegDataSample
# from opencd.registry import HOOKS
# from opencd.visualization import CDLocalVisualizer
#
#
# @HOOKS.register_module()
# class CDVisualizationHook(SegVisualizationHook):
#     """Change Detection Visualization Hook."""
#
#     def __init__(self,
#                  img_shape: tuple = None,
#                  draw_on_from_to_img: bool = False,
#                  draw: bool = False,
#                  interval: int = 50,
#                  show: bool = False,
#                  wait_time: float = 0.,
#                  backend_args: Optional[dict] = None):
#
#         # 手动设置基本属性，避免调用父类的初始化
#         self.draw = draw
#         self.interval = interval
#         self.show = show
#         self.wait_time = wait_time
#
#         self.img_shape = img_shape
#         self.draw_on_from_to_img = draw_on_from_to_img
#         if self.draw_on_from_to_img:
#             warnings.warn('`draw_on_from_to_img` works only in '
#                           'semantic change detection.')
#
#         # 直接创建可视化器实例
#         self._visualizer = CDLocalVisualizer()
#
#         if self.show:
#             # 如果只是显示，不需要后端
#             self._visualizer._vis_backends = {}
#
#         self.backend_args = backend_args.copy() if backend_args else None
#
#         # 初始化统计信息
#         self.total_batches = 0
#         self.total_outputs = 0
#         self.total_visualized = 0
#
#     def _after_iter(self,
#                     runner: Runner,
#                     batch_idx: int,
#                     data_batch: dict,
#                     outputs: Sequence[SegDataSample],
#                     mode: str = 'val') -> None:
#         """Run after every ``self.interval`` validation iterations."""
#         print(f"=== 进入 _after_iter ===")
#         print(f"模式: {mode}, 批次: {batch_idx}, 输出数量: {len(outputs)}")
#
#         self.total_batches += 1
#         self.total_outputs += len(outputs) if outputs else 0
#
#         if self.draw is False:
#             print("❌ 跳过: draw=False")
#             return
#
#         if mode == 'train':
#             print("❌ 跳过: mode=train")
#             return
#
#         print(f"✅ 基础条件满足，检查间隔: batch_idx={batch_idx}, interval={self.interval}")
#
#         # 检查间隔条件
#         if batch_idx % self.interval != 0:
#             print(f"❌ 不满足间隔条件: {batch_idx} % {self.interval} != 0")
#             return
#
#         print(f"✅ 满足间隔条件，开始可视化处理")
#         batch_visualized = 0
#
#         for i, output in enumerate(outputs):
#             print(f"  处理输出 {i}")
#
#             # 创建窗口名
#             window_name = f"sample_{batch_idx}_{i}"
#             img_from_to = []
#
#             # 创建背景图像
#             if self.img_shape is not None:
#                 img = np.ones(self.img_shape, dtype=np.uint8) * 128
#             else:
#                 img = np.ones((512, 512, 3), dtype=np.uint8) * 128
#
#             print(f"  准备调用 add_datasample: {window_name}")
#             print("7")
#             # 调用可视化器
#             try:
#                 print("8")
#                 self._visualizer.add_datasample(
#                     window_name,
#                     img,
#                     img_from_to,
#                     data_sample=output,
#                     show=self.show,
#                     wait_time=self.wait_time,
#                     step=runner.iter,
#                     draw_gt=False)
#                 batch_visualized += 1
#                 self.total_visualized += 1
#                 print(f"  ✅ 成功可视化: {window_name}")
#             except Exception as e:
#                 print("9")
#                 print(f"  ❌ 可视化失败: {window_name}, 错误: {e}")
#                 import traceback
#                 print(f"  详细堆栈: {traceback.format_exc()}")
#
#         print(f"本批次可视化数量: {batch_visualized}")
#
#     def after_test_epoch(self, runner, metrics=None):
#         """在测试结束后打印统计摘要"""
#         print("\n" + "=" * 50)
#         print("可视化统计摘要")
#         print("=" * 50)
#         print(f"总处理批次: {self.total_batches}")
#         print(f"总输出样本: {self.total_outputs}")
#         print(f"总可视化样本: {self.total_visualized}")
#
#         if self.total_outputs > 0:
#             visualization_ratio = (self.total_visualized / self.total_outputs) * 100
#             print(f"可视化比例: {visualization_ratio:.1f}%")
#
#         print("=" * 50)

# Copyright (c) Open-CD. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmengine.runner import Runner

from mmseg.engine import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer


@HOOKS.register_module()
class CDVisualizationHook(SegVisualizationHook):
    """Change Detection Visualization Hook. Used to visualize validation and
    testing process prediction results.

    Args:
        img_shape (tuple): if img_shape is given and `draw_on_from_to_img` is
            False, the original images will not be read.
        draw_on_from_to_img (bool): whether to draw semantic prediction results
            on the original images. If it is False, it means that drawing on
            the black board. Defaults to False.

    """

    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self.img_shape = img_shape
        self.draw_on_from_to_img = draw_on_from_to_img
        if self.draw_on_from_to_img:
            warnings.warn('`draw_on_from_to_img` works only in '
                          'semantic change detection.')
        self._visualizer: CDLocalVisualizer = \
            CDLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        # 初始化统计信息
        if not hasattr(self, 'total_batches'):
            self.total_batches = 0
            self.total_outputs = 0
            self.total_visualized = 0

        self.total_batches += 1
        self.total_outputs += len(outputs) if outputs else 0

        # print(f"=== 可视化钩子调用 ===")
        # print(f"模式: {mode}, 批次: {batch_idx}, 间隔: {self.interval}")
        # print(f"当前批次输出数量: {len(outputs) if outputs else 0}")
        # print(f"累计批次: {self.total_batches}, 累计输出: {self.total_outputs}")

        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            print(f"✓ 执行可视化: batch_idx={batch_idx} 满足间隔条件")
            batch_visualized = 0

            for i, output in enumerate(outputs):
                # 使用原始图像文件名作为预测结果文件名
                if hasattr(output, "img_path") and output.img_path:
                    if isinstance(output.img_path, (list, tuple)):
                        # 对于变化检测，使用第一张图像的文件名
                        original_img_path = output.img_path[0]
                    else:
                        original_img_path = output.img_path

                    # 提取原始文件名（不含路径和扩展名）
                    original_filename = osp.splitext(osp.basename(original_img_path))[0]
                    # 使用原始文件名作为窗口名
                    window_name = original_filename
                    print(f"  使用原始文件名: {original_filename}")

                    # 设置图像来源
                    img_from_to = []
                    if self.draw_on_from_to_img:
                        # 加载原始图像对
                        for _img_path in output.img_path:
                            try:
                                _img_bytes = fileio.get(_img_path, backend_args=self.backend_args)
                                _img = mmcv.imfrombytes(_img_bytes, channel_order='rgb')
                                img_from_to.append(_img)
                                print(f"    加载图像: {osp.basename(_img_path)}")
                            except Exception as e:
                                print(f"    加载图像失败 {_img_path}: {e}")
                                # 创建灰色背景作为备用
                                if self.img_shape is not None:
                                    img_from_to.append(np.ones(self.img_shape, dtype=np.uint8) * 128)
                                else:
                                    img_from_to.append(np.ones((512, 512, 3), dtype=np.uint8) * 128)
                else:
                    # 备用方案
                    window_name = f"batch{batch_idx}_output{i}"
                    img_from_to = []
                    print(f"  使用备用文件名: {window_name}")

                # 创建背景图像
                if img_from_to and len(img_from_to) > 0:
                    img = img_from_to[0]  # 使用第一张输入图像作为背景
                else:
                    if self.img_shape is not None:
                        img = np.ones(self.img_shape, dtype=np.uint8) * 128
                    else:
                        img = np.ones((512, 512, 3), dtype=np.uint8) * 128

                # 调用可视化器
                try:
                    self._visualizer.add_datasample(
                        window_name,
                        img,
                        img_from_to,
                        data_sample=output,
                        show=self.show,
                        wait_time=self.wait_time,
                        step=runner.iter,
                        draw_gt=False)
                    batch_visualized += 1
                    self.total_visualized += 1
                    print(f"  ✓ 成功可视化: {window_name}")
                except Exception as e:
                    print(f"  ✗ 可视化失败: {window_name}, 错误: {e}")

            print(f"本批次可视化数量: {batch_visualized}")
            print(f"累计可视化总数: {self.total_visualized}")

        else:
            print(f"✗ 跳过可视化: batch_idx={batch_idx} 不满足间隔条件")

    def after_test_epoch(self, runner, metrics=None):
        """在测试结束后打印统计摘要"""
        print("\n" + "=" * 50)
        print("可视化统计摘要")
        print("=" * 50)
        print(f"总处理批次: {self.total_batches}")
        print(f"总输出样本: {self.total_outputs}")
        print(f"总可视化样本: {self.total_visualized}")

        if hasattr(self, 'total_outputs') and self.total_outputs > 0:
            visualization_ratio = (self.total_visualized / self.total_outputs) * 100
            print(f"可视化比例: {visualization_ratio:.1f}%")

        print("=" * 50)