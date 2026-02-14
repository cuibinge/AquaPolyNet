import argparse
from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from hisup.utils.metrics.polis import PolisEval
from hisup.utils.metrics.angle_eval import ContourEval
from hisup.utils.metrics.cIoU import compute_IoU_cIoU
import numpy as np

def coco_eval(annFile, resFile):
    type = 1  # segmentation
    annType = ['bbox', 'segm']
    print('Running demo for *%s* results.' % (annType[type]))

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats = cocoEval.stats
    print("\nSegmentation Metrics:")
    print(f"AP: {stats[0]:.3f}")
    print(f"AP50: {stats[1]:.3f}")
    print(f"AP75: {stats[2]:.3f}")
    print(f"AR: {stats[8]:.3f}")
    print(f"AR50: {stats[9]:.3f}")
    print(f"AR75: {stats[10]:.3f}")
    return stats

def boundary_eval(annFile, resFile):
    dilation_ratio = 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats = cocoEval.stats
    print("\nBoundary Metrics:")
    print(f"AP: {stats[0]:.3f}")
    print(f"AP50: {stats[1]:.3f}")
    print(f"AP75: {stats[2]:.3f}")
    print(f"AR: {stats[8]:.3f}")
    print(f"AR50: {stats[9]:.3f}")
    print(f"AR75: {stats[10]:.3f}")
    return stats

def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()
    
    print("\nPolygon (Polis) Metrics:")
    
    cocoEval = COCOeval(gt_coco, dt_coco, iouType="segm")
    cocoEval.params.imgIds = gt_coco.getImgIds()
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    
    iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    ap_scores = []
    ar_scores = []
    
    for iouThr in iouThrs:
        cocoEval.params.iouThrs = np.array([iouThr])
        cocoEval.accumulate()
        precision = cocoEval.eval['precision'][0, :, 0, 0, -1]
        recall = cocoEval.eval['recall'][0, :, 0, -1]
        
        valid_precision = precision[precision > -1]
        ap = np.mean(valid_precision) if valid_precision.size > 0 else 0.0
        ap_scores.append(ap)
        
        valid_recall = recall[recall > -1]
        ar = np.mean(valid_recall) if valid_recall.size > 0 else 0.0
        ar_scores.append(ar)
    
    ap = np.mean(ap_scores) if ap_scores else 0.0
    ap50 = ap_scores[0] if ap_scores else 0.0
    ap75 = ap_scores[5] if len(ap_scores) > 5 else 0.0
    ar = np.mean(ar_scores) if ar_scores else 0.0
    ar50 = ar_scores[0] if ar_scores else 0.0
    ar75 = ar_scores[5] if len(ar_scores) > 5 else 0.0
    
    print(f"AP: {ap:.3f}")
    print(f"AP50: {ap50:.3f}")
    print(f"AP75: {ap75:.3f}")
    print(f"AR: {ar:.3f}")
    print(f"AR50: {ar50:.3f}")
    print(f"AR75: {ar75:.3f}")
    
    stats = [ap, ap50, ap75, 0, 0, 0, 0, 0, ar, ar50, ar75, 0]
    return stats

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument("--eval-type", default="coco_iou", choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"])
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == 'coco_iou':
        coco_eval(gt_file, dt_file)
    elif eval_type == 'boundary_iou':
        boundary_eval(gt_file, dt_file)
    elif eval_type == 'polis':
        polis_eval(gt_file, dt_file)
    elif eval_type == 'angle':
        max_angle_error_eval(gt_file, dt_file)
    elif eval_type == 'ciou':
        compute_IoU_cIoU(dt_file, gt_file)
    else:
        raise RuntimeError('please choose a correct type from \
                            ["coco_iou", "boundary_iou", "polis", "angle", "ciou"]')

if __name__ == "__main__":
    main()
# import argparse
# from multiprocess import Pool
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from boundary_iou.coco_instance_api.coco import COCO as BCOCO
# from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
# from hisup.utils.metrics.polis import PolisEval
# from hisup.utils.metrics.angle_eval import ContourEval
# from hisup.utils.metrics.cIoU import compute_IoU_cIoU

# def coco_eval(annFile, resFile):
#     type = 1  # segmentation
#     annType = ['bbox', 'segm']
#     print('Running demo for *%s* results.' % (annType[type]))

#     cocoGt = COCO(annFile)
#     cocoDt = cocoGt.loadRes(resFile)

#     imgIds = cocoGt.getImgIds()
#     imgIds = imgIds[:]

#     cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
#     cocoEval.params.imgIds = imgIds
#     cocoEval.params.catIds = [1]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

#     # Extract and print AP and AR metrics
#     stats = cocoEval.stats
#     print("\nSegmentation Metrics:")
#     print(f"AP: {stats[0]:.3f}")
#     print(f"AP50: {stats[1]:.3f}")
#     print(f"AP75: {stats[2]:.3f}")
#     print(f"AR: {stats[8]:.3f}")
#     print(f"AR50: {stats[9]:.3f}")
#     print(f"AR75: {stats[10]:.3f}")
#     return stats

# def boundary_eval(annFile, resFile):
#     dilation_ratio = 0.02  # default settings
#     cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
#     cocoDt = cocoGt.loadRes(resFile)
#     cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()

#     # Extract and print AP and AR metrics
#     stats = cocoEval.stats
#     print("\nBoundary Metrics:")
#     print(f"AP: {stats[0]:.3f}")
#     print(f"AP50: {stats[1]:.3f}")
#     print(f"AP75: {stats[2]:.3f}")
#     print(f"AR: {stats[8]:.3f}")
#     print(f"AR50: {stats[9]:.3f}")
#     print(f"AR75: {stats[10]:.3f}")
#     return stats

# def polis_eval(annFile, resFile):
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
    
#     # Run PolisEval to get the average Polis score
#     polisEval = PolisEval(gt_coco, dt_coco)
#     polisEval.evaluate()
    
#     # Assuming PolisEval prints the average Polis score but doesn't return stats
#     print("\nPolygon (Polis) Metrics:")
#     # If PolisEval doesn't provide stats, use COCOeval for AP/AR metrics
#     cocoEval = COCOeval(gt_coco, dt_coco, iouType="segm")  # Use segmentation for polygon evaluation
#     cocoEval.params.imgIds = gt_coco.getImgIds()
#     cocoEval.params.catIds = [1]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
    
#     # Extract and print AP and AR metrics
#     stats = cocoEval.stats
#     print(f"AP: {stats[0]:.3f}")
#     print(f"AP50: {stats[1]:.3f}")
#     print(f"AP75: {stats[2]:.3f}")
#     print(f"AR: {stats[8]:.3f}")
#     print(f"AR50: {stats[9]:.3f}")
#     print(f"AR75: {stats[10]:.3f}")
#     return stats

# def max_angle_error_eval(annFile, resFile):
#     gt_coco = COCO(annFile)
#     dt_coco = gt_coco.loadRes(resFile)
#     contour_eval = ContourEval(gt_coco, dt_coco)
#     pool = Pool(processes=20)
#     max_angle_diffs = contour_eval.evaluate(pool=pool)
#     print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gt-file", default="")
#     parser.add_argument("--dt-file", default="")
#     parser.add_argument("--eval-type", default="coco_iou", choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"])
#     args = parser.parse_args()

#     eval_type = args.eval_type
#     gt_file = args.gt_file
#     dt_file = args.dt_file
#     if eval_type == 'coco_iou':
#         coco_eval(gt_file, dt_file)
#     elif eval_type == 'boundary_iou':
#         boundary_eval(gt_file, dt_file)
#     elif eval_type == 'polis':
#         polis_eval(gt_file, dt_file)
#     elif eval_type == 'angle':
#         max_angle_error_eval(gt_file, dt_file)
#     elif eval_type == 'ciou':
#         compute_IoU_cIoU(dt_file, gt_file)
#     else:
#         raise RuntimeError('please choose a correct type from \
#                             ["coco_iou", "boundary_iou", "polis", "angle", "ciou"]')

# if __name__ == "__main__":
#     main()