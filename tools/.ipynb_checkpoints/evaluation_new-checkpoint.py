# -*- coding: UTF-8 -*-

import argparse
import numpy as np
from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from hisup.utils.metrics.polis import PolisEval
from hisup.utils.metrics.angle_eval import ContourEval
from hisup.utils.metrics.cIoU import compute_IoU_cIoU
import traceback
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure
from scipy.ndimage import binary_dilation
import logging
import os

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)  # 定义全局 log 对象

def coco_eval(annFile, resFile):
    type_idx = 1
    annType = ['bbox', 'segm']
    print(f'\nRunning COCO evaluation for *{annType[type_idx]}* results.')

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type_idx])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats

def boundary_eval(annFile, resFile):
    print('\nRunning Boundary IoU evaluation.')
    dilation_ratio = 0.02  # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def polis_eval(annFile, resFile):
    print('\nRunning Polis evaluation.')
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()

def pixel_boundary_eval(annFile, resFile):
    """
    像素级和边界评估，计算 Precision、Recall、F1-score、mIoU 和总体准确率 (OAbdy)
    边界指标基于 5x5 窗口膨胀后的边界区域，平均前景和背景评分
    """
    print('\n运行像素级和边界评估.')
    
    annFile_processed = annFile
    resFile_processed = resFile

    try:
        gt_coco = COCO(annFile_processed)
        dt_coco = gt_coco.loadRes(resFile_processed)
        
        img_ids = gt_coco.getImgIds()
        # 像素级指标列表
        pixel_precision_list, pixel_recall_list, pixel_f1_list, pixel_miou_list = [], [], [], []
        # 边界级指标列表（前景、背景、平均）
        bdy_oa_list, bdy_precision_fg_list, bdy_recall_fg_list, bdy_f1_fg_list, bdy_miou_fg_list = [], [], [], [], []
        bdy_precision_bg_list, bdy_recall_bg_list, bdy_f1_bg_list, bdy_miou_bg_list = [], [], [], []
        bdy_precision_avg_list, bdy_recall_avg_list, bdy_f1_avg_list, bdy_miou_avg_list = [], [], [], []

        # 定义 5x5 膨胀核
        dilation_kernel = np.ones((5, 5), dtype=np.uint8)

        for img_id in img_ids:
            img_info = gt_coco.loadImgs(img_id)[0]
            img_name = img_info['file_name']
            
            ann_ids_gt = gt_coco.getAnnIds(imgIds=img_id)
            anns_gt = gt_coco.loadAnns(ann_ids_gt)
            ann_ids_dt = dt_coco.getAnnIds(imgIds=img_id)
            anns_dt = dt_coco.loadAnns(ann_ids_dt)

            if not anns_gt or not anns_dt:
                log.info(f"{img_name} 无有效标注，跳过.")
                continue

            height, width = img_info['height'], img_info['width']
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            dt_mask = np.zeros((height, width), dtype=np.uint8)

            for ann in anns_gt:
                mask = gt_coco.annToMask(ann)
                gt_mask = np.logical_or(gt_mask, mask)

            for ann in anns_dt:
                mask = dt_coco.annToMask(ann)
                dt_mask = np.logical_or(dt_mask, mask)

            # 像素级指标计算
            intersection = np.logical_and(gt_mask, dt_mask).sum()
            union = np.logical_or(gt_mask, dt_mask).sum()
            gt_sum = gt_mask.sum()
            dt_sum = dt_mask.sum()

            pixel_precision = intersection / dt_sum if dt_sum > 0 else 0
            pixel_recall = intersection / gt_sum if gt_sum > 0 else 0
            pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0
            pixel_miou = intersection / union if union > 0 else 0

            pixel_precision_list.append(pixel_precision)
            pixel_recall_list.append(pixel_recall)
            pixel_f1_list.append(pixel_f1)
            pixel_miou_list.append(pixel_miou)

            # 边界区域掩码生成
            # 真实边界 (GT)：边缘检测 + 膨胀
            gt_edges = measure.find_contours(gt_mask, 0.5)
            gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
            if gt_edges:
                for contour in gt_edges:
                    contour = contour.astype(int)
                    gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
                gt_bdy_mask = binary_dilation(gt_bdy_mask, structure=dilation_kernel).astype(np.uint8)

            # 预测边界 (DT)：边缘检测 + 膨胀
            dt_edges = measure.find_contours(dt_mask, 0.5)
            dt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
            if dt_edges:
                for contour in dt_edges:
                    contour = contour.astype(int)
                    dt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
                dt_bdy_mask = binary_dilation(dt_bdy_mask, structure=dilation_kernel).astype(np.uint8)

            if not gt_bdy_mask.any() or not dt_bdy_mask.any():
                log.info(f"{img_name} 无有效边界区域，跳过.")
                continue

            # 前景边界指标
            fg_intersection = np.logical_and(gt_bdy_mask, dt_bdy_mask).sum()
            fg_union = np.logical_or(gt_bdy_mask, dt_bdy_mask).sum()
            fg_gt_sum = gt_bdy_mask.sum()
            fg_dt_sum = dt_bdy_mask.sum()

            fg_precision = fg_intersection / fg_dt_sum if fg_dt_sum > 0 else 0
            fg_recall = fg_intersection / fg_gt_sum if fg_gt_sum > 0 else 0
            fg_f1 = 2 * (fg_precision * fg_recall) / (fg_precision + fg_recall) if (fg_precision + fg_recall) > 0 else 0
            fg_miou = fg_intersection / fg_union if fg_union > 0 else 0

            # 背景边界指标
            gt_bdy_mask_bg = 1 - gt_bdy_mask
            dt_bdy_mask_bg = 1 - dt_bdy_mask
            bg_intersection = np.logical_and(gt_bdy_mask_bg, dt_bdy_mask_bg).sum()
            bg_union = np.logical_or(gt_bdy_mask_bg, dt_bdy_mask_bg).sum()
            bg_gt_sum = gt_bdy_mask_bg.sum()
            bg_dt_sum = dt_bdy_mask_bg.sum()

            bg_precision = bg_intersection / bg_dt_sum if bg_dt_sum > 0 else 0
            bg_recall = bg_intersection / bg_gt_sum if bg_gt_sum > 0 else 0
            bg_f1 = 2 * (bg_precision * bg_recall) / (bg_precision + bg_recall) if (bg_precision + bg_recall) > 0 else 0
            bg_miou = bg_intersection / bg_union if bg_union > 0 else 0

            # 总体准确率 (OAbdy)
            oa = (fg_intersection + bg_intersection) / (height * width)

            # 平均边界指标
            avg_precision = (fg_precision + bg_precision) / 2
            avg_recall = (fg_recall + bg_recall) / 2
            avg_f1 = (fg_f1 + bg_f1) / 2
            avg_miou = (fg_miou + bg_miou) / 2

            # 存储指标
            bdy_oa_list.append(oa)
            bdy_precision_fg_list.append(fg_precision)
            bdy_recall_fg_list.append(fg_recall)
            bdy_f1_fg_list.append(fg_f1)
            bdy_miou_fg_list.append(fg_miou)
            bdy_precision_bg_list.append(bg_precision)
            bdy_recall_bg_list.append(bg_recall)
            bdy_f1_bg_list.append(bg_f1)
            bdy_miou_bg_list.append(bg_miou)
            bdy_precision_avg_list.append(avg_precision)
            bdy_recall_avg_list.append(avg_recall)
            bdy_f1_avg_list.append(avg_f1)
            bdy_miou_avg_list.append(avg_miou)

            # 输出每张图像的指标
            log.info(f"{img_name} 的像素级 Precision: {pixel_precision:.4f}")
            log.info(f"{img_name} 的像素级 Recall: {pixel_recall:.4f}")
            log.info(f"{img_name} 的像素级 F1-score: {pixel_f1:.4f}")
            log.info(f"{img_name} 的像素级 mIoU: {pixel_miou:.4f}")
            log.info(f"{img_name} 的边界 OAbdy: {oa:.4f}")
            log.info(f"{img_name} 的前景边界 Precision: {fg_precision:.4f}")
            log.info(f"{img_name} 的前景边界 Recall: {fg_recall:.4f}")
            log.info(f"{img_name} 的前景边界 F1-score: {fg_f1:.4f}")
            log.info(f"{img_name} 的前景边界 mIoU: {fg_miou:.4f}")
            log.info(f"{img_name} 的背景边界 Precision: {bg_precision:.4f}")
            log.info(f"{img_name} 的背景边界 Recall: {bg_recall:.4f}")
            log.info(f"{img_name} 的背景边界 F1-score: {bg_f1:.4f}")
            log.info(f"{img_name} 的背景边界 mIoU: {bg_miou:.4f}")
            log.info(f"{img_name} 的平均边界 Precision: {avg_precision:.4f}")
            log.info(f"{img_name} 的平均边界 Recall: {avg_recall:.4f}")
            log.info(f"{img_name} 的平均边界 F1-score: {avg_f1:.4f}")
            log.info(f"{img_name} 的平均边界 mIoU: {avg_miou:.4f}")

        # 计算平均指标
        avg_pixel_precision = np.mean(pixel_precision_list) if pixel_precision_list else 0
        avg_pixel_recall = np.mean(pixel_recall_list) if pixel_recall_list else 0
        avg_pixel_f1 = np.mean(pixel_f1_list) if pixel_f1_list else 0
        avg_pixel_miou = np.mean(pixel_miou_list) if pixel_miou_list else 0

        avg_bdy_oa = np.mean(bdy_oa_list) if bdy_oa_list else 0
        avg_bdy_precision_fg = np.mean(bdy_precision_fg_list) if bdy_precision_fg_list else 0
        avg_bdy_recall_fg = np.mean(bdy_recall_fg_list) if bdy_recall_fg_list else 0
        avg_bdy_f1_fg = np.mean(bdy_f1_fg_list) if bdy_f1_fg_list else 0
        avg_bdy_miou_fg = np.mean(bdy_miou_fg_list) if bdy_miou_fg_list else 0
        avg_bdy_precision_bg = np.mean(bdy_precision_bg_list) if bdy_precision_bg_list else 0
        avg_bdy_recall_bg = np.mean(bdy_recall_bg_list) if bdy_recall_bg_list else 0
        avg_bdy_f1_bg = np.mean(bdy_f1_bg_list) if bdy_f1_bg_list else 0
        avg_bdy_miou_bg = np.mean(bdy_miou_bg_list) if bdy_miou_bg_list else 0
        avg_bdy_precision = np.mean(bdy_precision_avg_list) if bdy_precision_avg_list else 0
        avg_bdy_recall = np.mean(bdy_recall_avg_list) if bdy_recall_avg_list else 0
        avg_bdy_f1 = np.mean(bdy_f1_avg_list) if bdy_f1_avg_list else 0
        avg_bdy_miou = np.mean(bdy_miou_avg_list) if bdy_miou_avg_list else 0

        # 输出平均指标
        print(f"\n平均像素级 Precision: {avg_pixel_precision:.4f}")
        print(f"平均像素级 Recall: {avg_pixel_recall:.4f}")
        print(f"平均像素级 F1-score: {avg_pixel_f1:.4f}")
        print(f"平均像素级 mIoU: {avg_pixel_miou:.4f}")
        print(f"\n平均边界 OAbdy: {avg_bdy_oa:.4f}")
        print(f"平均前景边界 Precision: {avg_bdy_precision_fg:.4f}")
        print(f"平均前景边界 Recall: {avg_bdy_recall_fg:.4f}")
        print(f"平均前景边界 F1-score: {avg_bdy_f1_fg:.4f}")
        print(f"平均前景边界 mIoU: {avg_bdy_miou_fg:.4f}")
        print(f"平均背景边界 Precision: {avg_bdy_precision_bg:.4f}")
        print(f"平均背景边界 Recall: {avg_bdy_recall_bg:.4f}")
        print(f"平均背景边界 F1-score: {avg_bdy_f1_bg:.4f}")
        print(f"平均背景边界 mIoU: {avg_bdy_miou_bg:.4f}")
        print(f"平均边界 Precision: {avg_bdy_precision:.4f}")
        print(f"平均边界 Recall: {avg_bdy_recall:.4f}")
        print(f"平均边界 F1-score: {avg_bdy_f1:.4f}")
        print(f"平均边界 mIoU: {avg_bdy_miou:.4f}")

    except Exception as e:
        print(f"像素级和边界评估出错: {str(e)}")
        traceback.print_exc()
    
    finally:
        if annFile_processed != annFile and os.path.exists(annFile_processed):
            os.remove(annFile_processed)
        if resFile_processed != resFile and os.path.exists(resFile_processed):
            os.remove(resFile_processed)

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

def ciou_eval(annFile, resFile):
    print('\nRunning Contour IoU (cIoU) evaluation.')
    try:
        compute_IoU_cIoU(resFile, annFile)
    except Exception as e:
        print(f"Error in cIoU evaluation: {str(e)}")
        traceback.print_exc()

def evaluate_all(annFile, resFile):
    print('='*50)
    print('Starting comprehensive evaluation')
    print('='*50)
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script for polygon-based instance segmentation')
    parser.add_argument("--gt-file", required=True, help="Path to ground truth annotations file")
    parser.add_argument("--dt-file", required=True, help="Path to detection results file")
    parser.add_argument("--eval-type", default="all", 
                        choices=["all", "coco_iou", "boundary_iou", "polis", "angle", "ciou", "pixel_boundary"],
                        help="Evaluation type to run")
    args = parser.parse_args()

    if args.eval_type == 'all':
        evaluate_all(args.gt_file, args.dt_file)
    else:
        if args.eval_type == 'coco_iou':
            coco_eval(args.gt_file, args.dt_file)
        elif args.eval_type == 'boundary_iou':
            boundary_eval(args.gt_file, args.dt_file)
        elif args.eval_type == 'polis':
            polis_eval(args.gt_file, args.dt_file)
        elif args.eval_type == 'angle':
            max_angle_error_eval(args.gt_file, args.dt_file)
        elif args.eval_type == 'ciou':
            ciou_eval(args.gt_file, args.dt_file)
        elif args.eval_type == 'pixel_boundary':
            pixel_boundary_eval(args.gt_file, args.dt_file)  # Assuming same as boundary_ioustances_val2017.json --dt-file results/seg_results.json            #  elif eval_type == 'angle':
            max_angle_error_eval(gt_file, dt_file)
        elif eval_type == 'ciou':
            ciou_eval(gt_file, dt_file)
        elif eval_type == 'pixel_boundary':
            pixel_boundary_eval(gt_file, dt_file)
        else:
            raise RuntimeError('Invalid evaluation type')
# import numpy as np
# from multiprocess import Pool
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from boundary_iou.coco_instance_api.coco import COCO as BCOCO
# from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
# from hisup.utils.metrics.polis import PolisEval
# from hisup.utils.metrics.angle_eval import ContourEval
# from hisup.utils.metrics.cIoU import compute_IoU_cIoU
# import traceback
# from shapely.geometry import MultiPolygon, Polygon
# from skimage import measure  # 用于边界提取
# import logging
# import os

# # 配置日志记录器
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# log = logging.getLogger(__name__)  # 定义全局 log 对象

# def coco_eval(annFile, resFile):
#     type_idx = 1
#     annType = ['bbox', 'segm']
#     print(f'\nRunning COCO evaluation for *{annType[type_idx]}* results.')

#     cocoGt = COCO(annFile)
#     cocoDt = cocoGt.loadRes(resFile)

#     imgIds = cocoGt.getImgIds()
#     imgIds = imgIds[:]

#     cocoEval = COCOeval(cocoGt, cocoDt, annType[type_idx])
#     cocoEval.params.imgIds = imgIds
#     cocoEval.params.catIds = [1]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#     return cocoEval.stats

# def boundary_eval(annFile, resFile):
#     print('\nRunning Boundary IoU evaluation.')
#     dilation_ratio = 0.02  # default settings 0.02
#     cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
#     cocoDt = cocoGt.loadRes(resFile)
#     cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

# def polis_eval(annFile, resFile):
#     print('\nRunning Polis evaluation.')
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     polisEval = PolisEval(gt_coco, dt_coco)
#     polisEval.evaluate()

# def pixel_boundary_eval(annFile, resFile):
#     """
#     像素级和边界评估，计算 Precision、Recall、F1-score 和 mIoU
#     """
#     print('\n运行像素级和边界评估.')
    
#     annFile_processed = annFile
#     resFile_processed = resFile

#     try:
#         gt_coco = COCO(annFile_processed)
#         dt_coco = gt_coco.loadRes(resFile_processed)
        
#         img_ids = gt_coco.getImgIds()
#         # 像素级指标列表
#         pixel_precision_list, pixel_recall_list, pixel_f1_list, pixel_miou_list = [], [], [], []
#         # 边界级指标列表
#         bdy_precision_list, bdy_recall_list, bdy_f1_list, bdy_miou_list = [], [], [], []

#         for img_id in img_ids:
#             img_info = gt_coco.loadImgs(img_id)[0]
#             img_name = img_info['file_name']
            
#             ann_ids_gt = gt_coco.getAnnIds(imgIds=img_id)
#             anns_gt = gt_coco.loadAnns(ann_ids_gt)
#             ann_ids_dt = dt_coco.getAnnIds(imgIds=img_id)
#             anns_dt = dt_coco.loadAnns(ann_ids_dt)

#             if not anns_gt or not anns_dt:
#                 log.info(f"{img_name} 无有效标注，跳过.")
#                 continue

#             height, width = img_info['height'], img_info['width']
#             gt_mask = np.zeros((height, width), dtype=np.uint8)
#             dt_mask = np.zeros((height, width), dtype=np.uint8)

#             for ann in anns_gt:
#                 mask = gt_coco.annToMask(ann)
#                 gt_mask = np.logical_or(gt_mask, mask)

#             for ann in anns_dt:
#                 mask = dt_coco.annToMask(ann)
#                 dt_mask = np.logical_or(dt_mask, mask)

#             # 像素级指标计算
#             intersection = np.logical_and(gt_mask, dt_mask).sum()
#             union = np.logical_or(gt_mask, dt_mask).sum()
#             gt_sum = gt_mask.sum()
#             dt_sum = dt_mask.sum()

#             pixel_precision = intersection / dt_sum if dt_sum > 0 else 0
#             pixel_recall = intersection / gt_sum if gt_sum > 0 else 0
#             pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0
#             pixel_miou = intersection / union if union > 0 else 0

#             pixel_precision_list.append(pixel_precision)
#             pixel_recall_list.append(pixel_recall)
#             pixel_f1_list.append(pixel_f1)
#             pixel_miou_list.append(pixel_miou)

#             # 边界提取
#             gt_boundary = measure.find_contours(gt_mask, 0.5)
#             dt_boundary = measure.find_contours(dt_mask, 0.5)

#             if not gt_boundary or not dt_boundary:
#                 log.info(f"{img_name} 无有效边界，跳过.")
#                 continue

#             # 边界级指标计算（基于边界掩码）
#             gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
#             dt_bdy_mask = np.zeros((height, width), dtype=np.uint8)

#             # 将边界点转换为掩码
#             for contour in gt_boundary:
#                 contour = contour.astype(int)
#                 gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
#             for contour in dt_boundary:
#                 contour = contour.astype(int)
#                 dt_bdy_mask[contour[:, 0], contour[:, 1]] = 1

#             bdy_intersection = np.logical_and(gt_bdy_mask, dt_bdy_mask).sum()
#             bdy_union = np.logical_or(gt_bdy_mask, dt_bdy_mask).sum()
#             gt_bdy_sum = gt_bdy_mask.sum()
#             dt_bdy_sum = dt_bdy_mask.sum()

#             bdy_precision = bdy_intersection / dt_bdy_sum if dt_bdy_sum > 0 else 0
#             bdy_recall = bdy_intersection / gt_bdy_sum if gt_bdy_sum > 0 else 0
#             bdy_f1 = 2 * (bdy_precision * bdy_recall) / (bdy_precision + bdy_recall) if (bdy_precision + bdy_recall) > 0 else 0
#             bdy_miou = bdy_intersection / bdy_union if bdy_union > 0 else 0

#             bdy_precision_list.append(bdy_precision)
#             bdy_recall_list.append(bdy_recall)
#             bdy_f1_list.append(bdy_f1)
#             bdy_miou_list.append(bdy_miou)

#             # 输出每张图像的指标
#             log.info(f"{img_name} 的像素级 Precision: {pixel_precision:.4f}")
#             log.info(f"{img_name} 的像素级 Recall: {pixel_recall:.4f}")
#             log.info(f"{img_name} 的像素级 F1-score: {pixel_f1:.4f}")
#             log.info(f"{img_name} 的像素级 mIoU: {pixel_miou:.4f}")
#             log.info(f"{img_name} 的边界 Precision: {bdy_precision:.4f}")
#             log.info(f"{img_name} 的边界 Recall: {bdy_recall:.4f}")
#             log.info(f"{img_name} 的边界 F1-score: {bdy_f1:.4f}")
#             log.info(f"{img_name} 的边界 mIoU: {bdy_miou:.4f}")

#         # 计算平均指标
#         avg_pixel_precision = np.mean(pixel_precision_list) if pixel_precision_list else 0
#         avg_pixel_recall = np.mean(pixel_recall_list) if pixel_recall_list else 0
#         avg_pixel_f1 = np.mean(pixel_f1_list) if pixel_f1_list else 0
#         avg_pixel_miou = np.mean(pixel_miou_list) if pixel_miou_list else 0

#         avg_bdy_precision = np.mean(bdy_precision_list) if bdy_precision_list else 0
#         avg_bdy_recall = np.mean(bdy_recall_list) if bdy_recall_list else 0
#         avg_bdy_f1 = np.mean(bdy_f1_list) if bdy_f1_list else 0
#         avg_bdy_miou = np.mean(bdy_miou_list) if bdy_miou_list else 0

#         # 输出平均指标
#         print(f"\n平均像素级 Precision: {avg_pixel_precision:.4f}")
#         print(f"平均像素级 Recall: {avg_pixel_recall:.4f}")
#         print(f"平均像素级 F1-score: {avg_pixel_f1:.4f}")
#         print(f"平均像素级 mIoU: {avg_pixel_miou:.4f}")
#         print(f"\n平均边界 Precision: {avg_bdy_precision:.4f}")
#         print(f"平均边界 Recall: {avg_bdy_recall:.4f}")
#         print(f"平均边界 F1-score: {avg_bdy_f1:.4f}")
#         print(f"平均边界 mIoU: {avg_bdy_miou:.4f}")

#     except Exception as e:
#         print(f"像素级和边界评估出错: {str(e)}")
#         traceback.print_exc()
    
#     finally:
#         if annFile_processed != annFile and os.path.exists(annFile_processed):
#             os.remove(annFile_processed)
#         if resFile_processed != resFile and os.path.exists(resFile_processed):
#             os.remove(resFile_processed)

# def max_angle_error_eval(annFile, resFile):
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     contour_eval = ContourEval(gt_coco, dt_coco)
#     pool = Pool(processes=20)
#     max_angle_diffs = contour_eval.evaluate(pool=pool)
#     print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

# def ciou_eval(annFile, resFile):
#     print('\nRunning Contour IoU (cIoU) evaluation.')
#     try:
#         compute_IoU_cIoU(resFile, annFile)
#     except Exception as e:
#         print(f"Error in cIoU evaluation: {str(e)}")
#         traceback.print_exc()

# def evaluate_all(annFile, resFile):
#     print('='*50)
#     print('Starting comprehensive evaluation')
#     print('='*50)
    
#     # Run all evaluations
#     coco_eval(annFile, resFile)
#     boundary_eval(annFile, resFile)
#     polis_eval(annFile, resFile)
#     max_angle_error_eval(annFile, resFile)
#     ciou_eval(annFile, resFile)
    
#     print('\n'+'='*50)
#     print('All evaluations completed!')
#     print('='*50)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gt-file", required=True, help="Path to ground truth annotations file")
#     parser.add_argument("--dt-file", required=True, help="Path to detection results file")
#     parser.add_argument("--eval-type", default="all", 
#                         choices=["all", "coco_iou", "boundary_iou", "polis", "angle", "ciou", "pixel_boundary"],
#                         help="Evaluation type to run")
#     args = parser.parse_args()

#     if args.eval_type == 'all':
#         evaluate_all(args.gt_file, args.dt_file)
#     else:
#         eval_type = args.eval_type
#         gt_file = args.gt_file
#         dt_file = args.dt_file
#         if eval_type == 'coco_iou':
#             coco_eval(gt_file, dt_file)
#         elif eval_type == 'boundary_iou':
#             boundary_eval(gt_file, dt_file)
#         elif eval_type == 'polis':
#             polis_eval(gt_file, dt_file)
#         elif eval_type == 'angle':
#             max_angle_error_eval(gt_file, dt_file)
#         elif eval_type == 'ciou':
#             ciou_eval(gt_file, dt_file)
#         elif eval_type == 'pixel_boundary':
#             pixel_boundary_eval(gt_file, dt_file)
#         else:
#             raise RuntimeError('Invalid evaluation type')
# import argparse
# import numpy as np
# from multiprocess import Pool
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from boundary_iou.coco_instance_api.coco import COCO as BCOCO
# from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
# from hisup.utils.metrics.polis import PolisEval
# from hisup.utils.metrics.angle_eval import ContourEval
# from hisup.utils.metrics.cIoU import compute_IoU_cIoU
# import traceback
# from shapely.geometry import MultiPolygon, Polygon
# from skimage import measure  # 用于边界提取
# import logging

# # 配置日志记录器
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# log = logging.getLogger(__name__)  # 定义全局 log 对象
# def coco_eval(annFile, resFile):
#     type_idx = 1
#     annType = ['bbox', 'segm']
#     print(f'\nRunning COCO evaluation for *{annType[type_idx]}* results.')

#     cocoGt = COCO(annFile)
#     cocoDt = cocoGt.loadRes(resFile)

#     imgIds = cocoGt.getImgIds()
#     imgIds = imgIds[:]

#     cocoEval = COCOeval(cocoGt, cocoDt, annType[type_idx])
#     cocoEval.params.imgIds = imgIds
#     cocoEval.params.catIds = [1]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#     return cocoEval.stats

# def boundary_eval(annFile, resFile):
#     print('\nRunning Boundary IoU evaluation.')
#     dilation_ratio = 0.02  # default settings 0.02
#     cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
#     cocoDt = cocoGt.loadRes(resFile)
#     cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

# def polis_eval(annFile, resFile):
#     print('\nRunning Polis evaluation.')
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     polisEval = PolisEval(gt_coco, dt_coco)
#     polisEval.evaluate()

# import numpy as np
# import os
# import traceback
# from pycocotools.coco import COCO
# from skimage import measure
# import logging as log

# def pixel_boundary_eval(annFile, resFile):
#     """
#     像素级和边界评估，计算 Precision、Recall、F1-score 和 mIoU
#     """
#     print('\n运行像素级和边界评估.')
    
#     annFile_processed = annFile
#     resFile_processed = resFile

#     try:
#         gt_coco = COCO(annFile_processed)
#         dt_coco = gt_coco.loadRes(resFile_processed)
        
#         img_ids = gt_coco.getImgIds()
#         # 像素级指标列表
#         pixel_precision_list, pixel_recall_list, pixel_f1_list, pixel_miou_list = [], [], [], []
#         # 边界级指标列表
#         bdy_precision_list, bdy_recall_list, bdy_f1_list, bdy_miou_list = [], [], [], []

#         for img_id in img_ids:
#             img_info = gt_coco.loadImgs(img_id)[0]
#             img_name = img_info['file_name']
            
#             ann_ids_gt = gt_coco.getAnnIds(imgIds=img_id)
#             anns_gt = gt_coco.loadAnns(ann_ids_gt)
#             ann_ids_dt = dt_coco.getAnnIds(imgIds=img_id)
#             anns_dt = dt_coco.loadAnns(ann_ids_dt)

#             if not anns_gt or not anns_dt:
#                 log.info(f"{img_name} 无有效标注，跳过.")
#                 continue

#             height, width = img_info['height'], img_info['width']
#             gt_mask = np.zeros((height, width), dtype=np.uint8)
#             dt_mask = np.zeros((height, width), dtype=np.uint8)

#             for ann in anns_gt:
#                 mask = gt_coco.annToMask(ann)
#                 gt_mask = np.logical_or(gt_mask, mask)

#             for ann in anns_dt:
#                 mask = dt_coco.annToMask(ann)
#                 dt_mask = np.logical_or(dt_mask, mask)

#             # 像素级指标计算
#             intersection = np.logical_and(gt_mask, dt_mask).sum()
#             union = np.logical_or(gt_mask, dt_mask).sum()
#             gt_sum = gt_mask.sum()
#             dt_sum = dt_mask.sum()

#             pixel_precision = intersection / dt_sum if dt_sum > 0 else 0
#             pixel_recall = intersection / gt_sum if gt_sum > 0 else 0
#             pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0
#             pixel_miou = intersection / union if union > 0 else 0

#             pixel_precision_list.append(pixel_precision)
#             pixel_recall_list.append(pixel_recall)
#             pixel_f1_list.append(pixel_f1)
#             pixel_miou_list.append(pixel_miou)

#             # 边界提取
#             gt_boundary = measure.find_contours(gt_mask, 0.5)
#             dt_boundary = measure.find_contours(dt_mask, 0.5)

#             if not gt_boundary or not dt_boundary:
#                 log.info(f"{img_name} 无有效边界，跳过.")
#                 continue

#             # 边界级指标计算（基于边界掩码）
#             gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
#             dt_bdy_mask = np.zeros((height, width), dtype=np.uint8)

#             # 将边界点转换为掩码
#             for contour in gt_boundary:
#                 contour = contour.astype(int)
#                 gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
#             for contour in dt_boundary:
#                 contour = contour.astype(int)
#                 dt_bdy_mask[contour[:, 0], contour[:, 1]] = 1

#             bdy_intersection = np.logical_and(gt_bdy_mask, dt_bdy_mask).sum()
#             bdy_union = np.logical_or(gt_bdy_mask, dt_bdy_mask).sum()
#             gt_bdy_sum = gt_bdy_mask.sum()
#             dt_bdy_sum = dt_bdy_mask.sum()

#             bdy_precision = bdy_intersection / dt_bdy_sum if dt_bdy_sum > 0 else 0
#             bdy_recall = bdy_intersection / gt_bdy_sum if gt_bdy_sum > 0 else 0
#             bdy_f1 = 2 * (bdy_precision * bdy_recall) / (bdy_precision + bdy_recall) if (bdy_precision + bdy_recall) > 0 else 0
#             bdy_miou = bdy_intersection / bdy_union if bdy_union > 0 else 0

#             bdy_precision_list.append(bdy_precision)
#             bdy_recall_list.append(bdy_recall)
#             bdy_f1_list.append(bdy_f1)
#             bdy_miou_list.append(bdy_miou)

#             # 输出每张图像的指标
#             log.info(f"{img_name} 的像素级 Precision: {pixel_precision:.4f}")
#             log.info(f"{img_name} 的像素级 Recall: {pixel_recall:.4f}")
#             log.info(f"{img_name} 的像素级 F1-score: {pixel_f1:.4f}")
#             log.info(f"{img_name} 的像素级 mIoU: {pixel_miou:.4f}")
#             log.info(f"{img_name} 的边界 Precision: {bdy_precision:.4f}")
#             log.info(f"{img_name} 的边界 Recall: {bdy_recall:.4f}")
#             log.info(f"{img_name} 的边界 F1-score: {bdy_f1:.4f}")
#             log.info(f"{img_name} 的边界 mIoU: {bdy_miou:.4f}")

#         # 计算平均指标
#         avg_pixel_precision = np.mean(pixel_precision_list) if pixel_precision_list else 0
#         avg_pixel_recall = np.mean(pixel_recall_list) if pixel_recall_list else 0
#         avg_pixel_f1 = np.mean(pixel_f1_list) if pixel_f1_list else 0
#         avg_pixel_miou = np.mean pixel_miou_list) if pixel_miou_list else 0

#         avg_bdy_precision = np.mean(bdy_precision_list) if bdy_precision_list else 0
#         avg_bdy_recall = np.mean(bdy_recall_list) if bdy_recall_list else 0
#         avg_bdy_f1 = np.mean(bdy_f1_list) if bdy_f1_list else 0
#         avg_bdy_miou = np.mean(bdy_miou_list) if bdy_miou_list else 0

#         # 输出平均指标
#         print(f"\n平均像素级 Precision: {avg_pixel_precision:.4f}")
#         print(f"平均像素级 Recall: {avg_pixel_recall:.4f}")
#         print(f"平均像素级 F1-score: {avg_pixel_f1:.4f}")
#         print(f"平均像素级 mIoU: {avg_pixel_miou:.4f}")
#         print(f"\n平均边界 Precision: {avg_bdy_precision:.4f}")
#         print(f"平均边界 Recall: {avg_bdy_recall:.4f}")
#         print(f"平均边界 F1-score: {avg_bdy_f1:.4f}")
#         print(f"平均边界 mIoU: {avg_bdy_miou:.4f}")

#     except Exception as e:
#         print(f"像素级和边界评估出错: {str(e)}")
#         traceback.print_exc()
    
#     finally:
#         if annFile_processed != annFile and os.path.exists(annFile_processed):
#             os.remove(annFile_processed)
#         if resFile_processed != resFile and os.path.exists(resFile_processed):
#             os.remove(resFile_processed)
# # def pixel_boundary_eval(annFile, resFile):
# #     """
# #     像素级和边界评估，计算 Precision、Recall、F1-score 和 mIoU
# #     """
# #     print('\n运行像素级和边界评估.')
    
# #     annFile_processed = annFile
# #     resFile_processed = resFile

# #     try:
# #         gt_coco = COCO(annFile_processed)
# #         dt_coco = gt_coco.loadRes(resFile_processed)
        
# #         img_ids = gt_coco.getImgIds()
# #         precision_bdy_list, recall_bdy_list, f1_bdy_list, miou_bdy_list = [], [], [], []

# #         for img_id in img_ids:
# #             img_info = gt_coco.loadImgs(img_id)[0]
# #             img_name = img_info['file_name']
            
# #             ann_ids_gt = gt_coco.getAnnIds(imgIds=img_id)
# #             anns_gt = gt_coco.loadAnns(ann_ids_gt)
# #             ann_ids_dt = dt_coco.getAnnIds(imgIds=img_id)
# #             anns_dt = dt_coco.loadAnns(ann_ids_dt)

# #             if not anns_gt or not anns_dt:
# #                 log.info(f"{img_name} 无有效标注，跳过.")
# #                 continue

# #             height, width = img_info['height'], img_info['width']
# #             gt_mask = np.zeros((height, width), dtype=np.uint8)
# #             dt_mask = np.zeros((height, width), dtype=np.uint8)

# #             for ann in anns_gt:
# #                 mask = gt_coco.annToMask(ann)
# #                 gt_mask = np.logical_or(gt_mask, mask)

# #             for ann in anns_dt:
# #                 mask = dt_coco.annToMask(ann)
# #                 dt_mask = np.logical_or(dt_mask, mask)

# #             gt_boundary = measure.find_contours(gt_mask, 0.5)
# #             dt_boundary = measure.find_contours(dt_mask, 0.5)

# #             if not gt_boundary or not dt_boundary:
# #                 log.info(f"{img_name} 无有效边界，跳过.")
# #                 continue

# #             intersection = np.logical_and(gt_mask, dt_mask).sum()
# #             union = np.logical_or(gt_mask, dt_mask).sum()
# #             gt_sum = gt_mask.sum()
# #             dt_sum = dt_mask.sum()

# #             precision = intersection / dt_sum if dt_sum > 0 else 0
# #             recall = intersection / gt_sum if gt_sum > 0 else 0
# #             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# #             miou = intersection / union if union > 0 else 0

# #             precision_bdy_list.append(precision)
# #             recall_bdy_list.append(recall)
# #             f1_bdy_list.append(f1)
# #             miou_bdy_list.append(miou)

# #             log.info(f"{img_name} 的边界 Precision: {precision:.4f}")
# #             log.info(f"{img_name} 的边界 Recall: {recall:.4f}")
# #             log.info(f"{img_name} 的边界 F1-score: {f1:.4f}")
# #             log.info(f"{img_name} 的边界 mIoU: {miou:.4f}")

# #         avg_precision = np.mean(precision_bdy_list) if precision_bdy_list else 0
# #         avg_recall = np.mean(recall_bdy_list) if recall_bdy_list else 0
# #         avg_f1 = np.mean(f1_bdy_list) if f1_bdy_list else 0
# #         avg_miou = np.mean(miou_bdy_list) if miou_bdy_list else 0

# #         print(f"\n平均边界 Precision: {avg_precision:.4f}")
# #         print(f"平均边界 Recall: {avg_recall:.4f}")
# #         print(f"平均边界 F1-score: {avg_f1:.4f}")
# #         print(f"平均边界 mIoU: {avg_miou:.4f}")

# #     except Exception as e:
# #         print(f"像素级和边界评估出错: {str(e)}")
# #         traceback.print_exc()
    
# #     finally:
# #         if annFile_processed != annFile and os.path.exists(annFile_processed):
# #             os.remove(annFile_processed)
# #         if resFile_processed != resFile and os.path.exists(resFile_processed):
# #             os.remove(resFile_processed)
            
# def max_angle_error_eval(annFile, resFile):
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     contour_eval = ContourEval(gt_coco, dt_coco)
#     pool = Pool(processes=20)
#     max_angle_diffs = contour_eval.evaluate(pool=pool)
#     print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

# def ciou_eval(annFile, resFile):
#     print('\nRunning Contour IoU (cIoU) evaluation.')
#     try:
#         compute_IoU_cIoU(resFile, annFile)
#     except Exception as e:
#         print(f"Error in cIoU evaluation: {str(e)}")
#         traceback.print_exc()

# def evaluate_all(annFile, resFile):
#     print('='*50)
#     print('Starting comprehensive evaluation')
#     print('='*50)
    
#     # Run all evaluations
#     coco_eval(annFile, resFile)
#     boundary_eval(annFile, resFile)
#     polis_eval(annFile, resFile)
#     max_angle_error_eval(annFile, resFile)
#     ciou_eval(annFile, resFile)
    
#     print('\n'+'='*50)
#     print('All evaluations completed!')
#     print('='*50)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gt-file", required=True, help="Path to ground truth annotations file")
#     parser.add_argument("--dt-file", required=True, help="Path to detection results file")
#     parser.add_argument("--eval-type", default="all", 
#                         choices=["all", "coco_iou", "boundary_iou", "polis", "angle", "ciou",'pixel_boundary'],
#                         help="Evaluation type to run")
#     args = parser.parse_args()

#     if args.eval_type == 'all':
#         evaluate_all(args.gt_file, args.dt_file)
#     else:
#         eval_type = args.eval_type
#         gt_file = args.gt_file
#         dt_file = args.dt_file
#         if eval_type == 'coco_iou':
#             coco_eval(gt_file, dt_file)
#         elif eval_type == 'boundary_iou':
#             boundary_eval(gt_file, dt_file)
#         elif eval_type == 'polis':
#             polis_eval(gt_file, dt_file)
#         elif eval_type == 'angle':
#             max_angle_error_eval(gt_file, dt_file)
#         elif eval_type == 'ciou':
#             ciou_eval(gt_file, dt_file)
#         elif eval_type == 'pixel_boundary':
#             pixel_boundary_eval(gt_file, dt_file)
#         else:
#             raise RuntimeError('Invalid evaluation type')