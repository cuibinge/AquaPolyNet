import os.path
import torch
import imageio
import numpy as np
from utils.SegmentationMetric import SegmentationMetric
import glob
from datetime import datetime
from utils.util import Logger
from pathlib import Path
import sys
import io
import cv2
import json
import shapely.geometry
import networkx as nx
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.measure import label as sk_label, regionprops
import tempfile
from skimage import measure
from scipy.ndimage import binary_dilation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8", line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.now()
now = str(datetime.now().date()) + "-" + str(datetime.now().time())[:5].replace(':', '-')

def readimage(dir):
    images_path_list = glob.glob(os.path.join(dir, "*.tif")) \
                       + glob.glob(os.path.join(dir, "*.png")) \
                       + glob.glob(os.path.join(dir, "*.tiff"))
    return images_path_list

def calculate_boundary_metrics(pred, gt):
    """Calculate boundary metrics between prediction and ground truth based on boundary region maps"""
    pred_mask = (pred == 1).astype(np.uint8)
    gt_mask = (gt == 1).astype(np.uint8)
    
    height, width = pred_mask.shape
    dilation_kernel = np.ones((5, 5), dtype=np.uint8)
    
    gt_edges = measure.find_contours(gt_mask, 0.5)
    gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
    if gt_edges:
        for contour in gt_edges:
            contour = contour.astype(int)
            gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
        gt_bdy_mask = binary_dilation(gt_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
    pred_edges = measure.find_contours(pred_mask, 0.5)
    pred_bdy_mask = np.zeros((height, width), dtype=np.uint8)
    if pred_edges:
        for contour in pred_edges:
            contour = contour.astype(int)
            pred_bdy_mask[contour[:, 0], contour[:, 1]] = 1
        pred_bdy_mask = binary_dilation(pred_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
    fg_intersection = np.logical_and(gt_bdy_mask, pred_bdy_mask).sum()
    fg_union = np.logical_or(gt_bdy_mask, pred_bdy_mask).sum()
    fg_gt_sum = gt_bdy_mask.sum()
    fg_pred_sum = pred_bdy_mask.sum()
    
    precision_fg = fg_intersection / (fg_pred_sum + 1e-6) if fg_pred_sum > 0 else 0
    recall_fg = fg_intersection / (fg_gt_sum + 1e-6) if fg_gt_sum > 0 else 0
    f1_fg = 2 * (precision_fg * recall_fg) / (precision_fg + recall_fg + 1e-6) if (precision_fg + recall_fg) > 0 else 0
    miou_fg = fg_intersection / (fg_union + 1e-6) if fg_union > 0 else 0
    
    gt_bdy_mask_bg = 1 - gt_bdy_mask
    pred_bdy_mask_bg = 1 - pred_bdy_mask
    bg_intersection = np.logical_and(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
    bg_union = np.logical_or(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
    bg_gt_sum = gt_bdy_mask_bg.sum()
    bg_pred_sum = pred_bdy_mask_bg.sum()
    
    precision_bg = bg_intersection / (bg_pred_sum + 1e-6) if bg_pred_sum > 0 else 0
    recall_bg = bg_intersection / (bg_gt_sum + 1e-6) if bg_gt_sum > 0 else 0
    f1_bg = 2 * (precision_bg * recall_bg) / (precision_bg + recall_bg + 1e-6) if (precision_bg + recall_bg) > 0 else 0
    miou_bg = bg_intersection / (bg_union + 1e-6) if bg_union > 0 else 0
    
    precision_bdy = (precision_fg + precision_bg) / 2
    recall_bdy = (recall_fg + recall_bg) / 2
    f1_bdy = (f1_fg + f1_bg) / 2
    miou_bdy = (miou_fg + miou_bg) / 2
    
    oabdy = (fg_intersection + bg_intersection) / (height * width + 1e-6)
    
    return oabdy, precision_bdy, recall_bdy, f1_bdy, miou_bdy

def calculate_apls(pred, gt):
    """Calculate APLS metric between prediction and ground truth"""
    pred_mask = (pred == 1).astype(np.uint8)
    gt_mask = (gt == 1).astype(np.uint8)
    
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pred_lines = [shapely.geometry.LineString(c[:,0,:]) for c in pred_contours if len(c) >= 2]
    gt_lines = [shapely.geometry.LineString(c[:,0,:]) for c in gt_contours if len(c) >= 2]
    
    if not pred_lines or not gt_lines:
        return 0.0
    
    G_pred = nx.Graph()
    G_gt = nx.Graph()
    
    for line in pred_lines:
        for i in range(len(line.coords)-1):
            G_pred.add_edge(line.coords[i], line.coords[i+1])
    
    for line in gt_lines:
        for i in range(len(line.coords)-1):
            G_gt.add_edge(line.coords[i], line.coords[i+1])
    
    try:
        pred_paths = list(nx.all_pairs_shortest_path_length(G_pred))
        gt_paths = list(nx.all_pairs_shortest_path_length(G_gt))
        
        score = 0.0
        count = 0
        for u1, paths1 in pred_paths:
            for u2, dist1 in paths1.items():
                if u1 == u2:
                    continue
                try:
                    dist2 = nx.shortest_path_length(G_gt, u1, u2)
                    score += min(1.0, abs(dist1 - dist2) / max(dist1, dist2))
                    count += 1
                except:
                    pass
        
        if count > 0:
            apls = 1.0 - (score / count)
        else:
            apls = 0.0
    except:
        apls = 0.0
    
    return apls

def calculate_ap_metrics(pred, gt):
    """Calculate APmask, APpoly, and additional MS-COCO metrics"""
    pred_mask = (pred == 1).astype(np.uint8)
    gt_mask = (gt == 1).astype(np.uint8)
    
    pred_labels = sk_label(pred_mask, connectivity=2)
    gt_labels = sk_label(gt_mask, connectivity=2)
    
    coco_gt = {
        "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    coco_pred = {
        "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    
    gt_props = regionprops(gt_labels)
    for idx, prop in enumerate(gt_props, 1):
        coords = prop.coords
        if len(coords) >= 4:
            poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
            if poly.is_valid:
                coco_gt["annotations"].append({
                    "id": idx,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": shapely.geometry.mapping(poly)["coordinates"],
                    "area": prop.area,
                    "bbox": list(prop.bbox),
                    "iscrowd": 0
                })
    
    pred_props = regionprops(pred_labels)
    for idx, prop in enumerate(pred_props, 1):
        coords = prop.coords
        if len(coords) >= 4:
            poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
            if poly.is_valid:
                coco_pred["annotations"].append({
                    "id": idx,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": shapely.geometry.mapping(poly)["coordinates"],
                    "area": prop.area,
                    "bbox": list(prop.bbox),
                    "iscrowd": 0,
                    "score": 0.9
                })
    
    if not coco_gt["annotations"] or not coco_pred["annotations"]:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_gt, f)
        gt_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_pred, f)
        pred_file = f.name
    
    coco_gt_obj = COCO(gt_file)
    coco_pred_obj = coco_gt_obj.loadRes(pred_file)
    
    coco_eval_mask = COCOeval(coco_gt_obj, coco_pred_obj, 'segm')
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()
    
    ap_mask = coco_eval_mask.stats[0]
    ap_mask_50 = coco_eval_mask.stats[1]
    ap_mask_75 = coco_eval_mask.stats[2]
    ar_mask = coco_eval_mask.stats[8]
    ar_mask_50 = coco_eval_mask.stats[9]
    ar_mask_75 = coco_eval_mask.stats[10]
    
    ap_poly = ap_mask
    ap_poly_50 = ap_mask_50
    ap_poly_75 = ap_mask_75
    ar_poly = ar_mask
    ar_poly_50 = ar_mask_50
    ar_poly_75 = ar_mask_75
    
    os.remove(gt_file)
    os.remove(pred_file)
    
    return ap_mask, ap_poly, ap_poly_50, ap_poly_75, ar_poly, ar_poly_50, ar_poly_75

def calculate_polisi(pred, gt):
    """Calculate Polisi metric for instance segmentation"""
    pred_mask = (pred == 1).astype(np.uint8)
    gt_mask = (gt == 1).astype(np.uint8)
    
    pred_labels = sk_label(pred_mask, connectivity=2)
    gt_labels = sk_label(gt_mask, connectivity=2)
    
    pred_instances = len(np.unique(pred_labels)) - 1
    gt_instances = len(np.unique(gt_labels)) - 1
    
    if gt_instances == 0:
        return 0.0
    
    matched = 0
    for pred_id in np.unique(pred_labels):
        if pred_id == 0:
            continue
        pred_instance = (pred_labels == pred_id).astype(np.uint8)
        best_iou = 0
        for gt_id in np.unique(gt_labels):
            if gt_id == 0:
                continue
            gt_instance = (gt_labels == gt_id).astype(np.uint8)
            intersection = np.logical_and(pred_instance, gt_instance).sum()
            union = np.logical_or(pred_instance, gt_instance).sum()
            iou = intersection / (union + 1e-6)
            best_iou = max(best_iou, iou)
        if best_iou > 0.5:
            matched += 1
    
    polisi = matched / gt_instances
    return polisi

def visualize_boundaries(image_dir, pred_mask, save_path, img_name):
    """Draw predicted boundaries on the original image and save"""
    original_img_path = os.path.join(image_dir, img_name + '.png')
    original_img = cv2.imread(original_img_path)
    
    if original_img is None:
        log.logger.warning(f"Cannot load original image {original_img_path}, skipping visualization")
        return
    
    img_color = original_img if original_img.shape[-1] == 3 else cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    pred_mask = (pred_mask == 1).astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img_color, contours, -1, (0, 0, 255), 1)
    
    cv2.imwrite(save_path, img_color)

def evaluate_segmentation(pred_path, gt_path, image_dir, output_dir):
    """Evaluate segmentation results by comparing prediction and ground truth"""
    pred_list = sorted(readimage(pred_path))
    gt_list = sorted(readimage(gt_path))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, now + '.log')
    log = Logger(log_path, level='debug')
    log.logger.info('Starting evaluation of segmentation results')
    
    # Global metric lists
    global_pa_list = []
    global_mpa_list = []
    global_miou_list = []
    global_oabdy_list = []
    global_precision_bdy_list = []
    global_recall_bdy_list = []
    global_f1_bdy_list = []
    global_miou_bdy_list = []
    global_apls_list = []
    global_ap_mask_list = []
    global_ap_poly_list = []
    global_ap_mask_50_list = []
    global_ap_mask_75_list = []
    global_ar_mask_list = []
    global_ar_mask_50_list = []
    global_ar_mask_75_list = []
    global_polisi_list = []
    
    for pred_file, gt_file in zip(pred_list, gt_list):
        start_time = datetime.now()
        
        # Load prediction and ground truth
        pred = imageio.imread(pred_file)
        gt = imageio.imread(gt_file)
        pred = (pred == 255).astype(np.uint8)
        
        # Basic segmentation metrics
        metric = SegmentationMetric(2)
        metric.addBatch(pred, gt)
        pa = metric.pixelAccuracy()
        mpa, _ = metric.meanPixelAccuracy()
        miou, _ = metric.meanIntersectionOverUnion()
        
        # Boundary metrics
        oabdy, precision_bdy, recall_bdy, f1_bdy, miou_bdy = calculate_boundary_metrics(pred, gt)
        
        # APLS metric
        apls = calculate_apls(pred, gt)
        
        # COCO-style metrics
        ap_mask, ap_poly, ap_poly_50, ap_poly_75, ar_poly, ar_poly_50, ar_poly_75 = calculate_ap_metrics(pred, gt)
        
        # Polisi metric
        polisi = calculate_polisi(pred, gt)
        
        end_time = datetime.now()
        eval_time = (end_time - start_time).total_seconds()
        
        # Save visualization
        img_name = Path(pred_file).stem
        boundary_save = os.path.join(output_dir, now + "_" + img_name + "-boundary.png")
        visualize_boundaries(image_dir, pred, boundary_save, img_name)
        
        # Log per-image results
        log.logger.info(f"Evaluation for {img_name}:")
        log.logger.info(f"Pixel Accuracy: {pa:.4f}")
        log.logger.info(f"Mean Pixel Accuracy: {mpa:.4f}")
        log.logger.info(f"Mean IoU: {miou:.4f}")
        log.logger.info(f"Boundary OA: {oabdy:.4f}")
        log.logger.info(f"Boundary Precision: {precision_bdy:.4f}")
        log.logger.info(f"Boundary Recall: {recall_bdy:.4f}")
        log.logger.info(f"Boundary F1: {f1_bdy:.4f}")
        log.logger.info(f"Boundary mIoU: {miou_bdy:.4f}")
        log.logger.info(f"APLS: {apls:.4f}")
        log.logger.info(f"APmask: {ap_mask:.4f}")
        log.logger.info(f"APpoly: {ap_poly:.4f}")
        log.logger.info(f"APpoly@50: {ap_poly_50:.4f}")
        log.logger.info(f"APpoly@75: {ap_poly_75:.4f}")
        log.logger.info(f"ARpoly: {ar_poly:.4f}")
        log.logger.info(f"ARpoly@50: {ar_poly_50:.4f}")
        log.logger.info(f"ARpoly@75: {ar_poly_75:.4f}")
        log.logger.info(f"Polisi: {polisi:.4f}")
        log.logger.info(f"Evaluation time: {eval_time:.3f}s")
        log.logger.info("\n" + "="*80 + "\n")
        
        # Store global metrics
        global_pa_list.append(pa)
        global_mpa_list.append(mpa)
        global_miou_list.append(miou)
        global_oabdy_list.append(oabdy)
        global_precision_bdy_list.append(precision_bdy)
        global_recall_bdy_list.append(recall_bdy)
        global_f1_bdy_list.append(f1_bdy)
        global_miou_bdy_list.append(miou_bdy)
        global_apls_list.append(apls)
        global_ap_mask_list.append(ap_mask)
        global_ap_poly_list.append(ap_poly)
        global_ap_mask_50_list.append(ap_poly_50)
        global_ap_mask_75_list.append(ap_poly_75)
        global_ar_mask_list.append(ar_poly)
        global_ar_mask_50_list.append(ar_poly_50)
        global_ar_mask_75_list.append(ar_poly_75)
        global_polisi_list.append(polisi)
    
    # Print and log global evaluation results
    print("\n全局评估结果:")
    print(f"平均PA: {np.mean(global_pa_list):.4f}")
    print(f"平均MPA: {np.mean(global_mpa_list):.4f}")
    print(f"平均mIoU: {np.mean(global_miou_list):.4f}")
    print(f"平均边界OA: {np.mean(global_oabdy_list):.4f}")
    print(f"平均边界Precision: {np.mean(global_precision_bdy_list):.4f}")
    print(f"平均边界Recall: {np.mean(global_recall_bdy_list):.4f}")
    print(f"平均边界F1-score: {np.mean(global_f1_bdy_list):.4f}")
    print(f"平均边界mIoU: {np.mean(global_miou_bdy_list):.4f}")
    print(f"平均APLS: {np.mean(global_apls_list):.4f}")
    print(f"平均APmask: {np.mean(global_ap_mask_list):.4f}")
    print(f"平均APpoly: {np.mean(global_ap_poly_list):.4f}")
    print(f"平均APpoly@50: {np.mean(global_ap_mask_50_list):.4f}")
    print(f"平均APpoly@75: {np.mean(global_ap_mask_75_list):.4f}")
    print(f"平均ARpoly: {np.mean(global_ar_mask_list):.4f}")
    print(f"平均ARpoly@50: {np.mean(global_ar_mask_50_list):.4f}")
    print(f"平均ARpoly@75: {np.mean(global_ar_mask_75_list):.4f}")
    print(f"平均Polisi: {np.mean(global_polisi_list):.4f}")
    
    log.logger.info("\n全局评估结果:")
    log.logger.info(f"平均PA: {np.mean(global_pa_list):.4f}")
    log.logger.info(f"平均MPA: {np.mean(global_mpa_list):.4f}")
    log.logger.info(f"平均mIoU: {np.mean(global_miou_list):.4f}")
    log.logger.info(f"平均边界OA: {np.mean(global_oabdy_list):.4f}")
    log.logger.info(f"平均边界Precision: {np.mean(global_precision_bdy_list):.4f}")
    log.logger.info(f"平均边界Recall: {np.mean(global_recall_bdy_list):.4f}")
    log.logger.info(f"平均边界F1-score: {np.mean(global_f1_bdy_list):.4f}")
    log.logger.info(f"平均边界mIoU: {np.mean(global_miou_bdy_list):.4f}")
    log.logger.info(f"平均APLS: {np.mean(global_apls_list):.4f}")
    log.logger.info(f"平均APmask: {np.mean(global_ap_mask_list):.4f}")
    log.logger.info(f"平均APpoly: {np.mean(global_ap_poly_list):.4f}")
    log.logger.info(f"平均APpoly@50: {np.mean(global_ap_mask_50_list):.4f}")
    log.logger.info(f"平均APpoly@75: {np.mean(global_ap_mask_75_list):.4f}")
    log.logger.info(f"平均ARpoly: {np.mean(global_ar_mask_list):.4f}")
    log.logger.info(f"平均ARpoly@50: {np.mean(global_ar_mask_50_list):.4f}")
    log.logger.info(f"平均ARpoly@75: {np.mean(global_ar_mask_75_list):.4f}")
    log.logger.info(f"平均Polisi: {np.mean(global_polisi_list):.4f}")
    log.logger.info("\n-------------------------------------------------------------------")

# Configuration
pred_dir = r"./outputs/lyg_hrnet48_300/seg"
gt_dir = r"./data/lyg/cut_300/val/gt"
image_dir = r"./data/lyg/cut_300/val/images"
output_dir = r"./outputs/lyg_hrnet48_300/eval"

# Run evaluation
evaluate_segmentation(pred_dir, gt_dir, image_dir, output_dir)
# # import os.path
# # import torch
# # import imageio
# # import numpy as np
# # from utils.SegmentationMetric import SegmentationMetric
# # import glob
# # from datetime import datetime
# # from utils.util import Logger
# # from pathlib import Path
# # import sys
# # import io
# # import cv2
# # import json
# # import shapely.geometry
# # import networkx as nx
# # from pycocotools.coco import COCO
# # from pycocotools.cocoeval import COCOeval
# # from skimage.measure import label as sk_label, regionprops
# # import tempfile
# # from skimage import measure
# # from scipy.ndimage import binary_dilation
# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8", line_buffering=True)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # now = datetime.now()
# # now = str(datetime.now().date()) + "-" + str(datetime.now().time())[:5].replace(':', '-')

# # def readimage(dir):
# #     images_path_list = glob.glob(os.path.join(dir, "*.tif")) \
# #                        + glob.glob(os.path.join(dir, "*.png")) \
# #                        + glob.glob(os.path.join(dir, "*.tiff"))
# #     return images_path_list

# # def calculate_boundary_metrics(pred, gt):
# #     """Calculate boundary metrics between prediction and ground truth based on boundary region maps"""
# #     pred_mask = (pred == 1).astype(np.uint8)
# #     gt_mask = (gt == 1).astype(np.uint8)
    
# #     height, width = pred_mask.shape
# #     dilation_kernel = np.ones((5, 5), dtype=np.uint8)
    
# #     # Boundary region masks
# #     gt_edges = measure.find_contours(gt_mask, 0.5)
# #     gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
# #     if gt_edges:
# #         for contour in gt_edges:
# #             contour = contour.astype(int)
# #             gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
# #         gt_bdy_mask = binary_dilation(gt_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
# #     pred_edges = measure.find_contours(pred_mask, 0.5)
# #     pred_bdy_mask = np.zeros((height, width), dtype=np.uint8)
# #     if pred_edges:
# #         for contour in pred_edges:
# #             contour = contour.astype(int)
# #             pred_bdy_mask[contour[:, 0], contour[:, 1]] = 1
# #         pred_bdy_mask = binary_dilation(pred_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
# #     # Calculate metrics
# #     fg_intersection = np.logical_and(gt_bdy_mask, pred_bdy_mask).sum()
# #     fg_union = np.logical_or(gt_bdy_mask, pred_bdy_mask).sum()
# #     fg_gt_sum = gt_bdy_mask.sum()
# #     fg_pred_sum = pred_bdy_mask.sum()
    
# #     precision_fg = fg_intersection / (fg_pred_sum + 1e-6) if fg_pred_sum > 0 else 0
# #     recall_fg = fg_intersection / (fg_gt_sum + 1e-6) if fg_gt_sum > 0 else 0
# #     f1_fg = 2 * (precision_fg * recall_fg) / (precision_fg + recall_fg + 1e-6) if (precision_fg + recall_fg) > 0 else 0
# #     miou_fg = fg_intersection / (fg_union + 1e-6) if fg_union > 0 else 0
    
# #     gt_bdy_mask_bg = 1 - gt_bdy_mask
# #     pred_bdy_mask_bg = 1 - pred_bdy_mask
# #     bg_intersection = np.logical_and(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
# #     bg_union = np.logical_or(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
# #     bg_gt_sum = gt_bdy_mask_bg.sum()
# #     bg_pred_sum = pred_bdy_mask_bg.sum()
    
# #     precision_bg = bg_intersection / (bg_pred_sum + 1e-6) if bg_pred_sum > 0 else 0
# #     recall_bg = bg_intersection / (bg_gt_sum + 1e-6) if bg_gt_sum > 0 else 0
# #     f1_bg = 2 * (precision_bg * recall_bg) / (precision_bg + recall_bg + 1e-6) if (precision_bg + recall_bg) > 0 else 0
# #     miou_bg = bg_intersection / (bg_union + 1e-6) if bg_union > 0 else 0
    
# #     precision_bdy = (precision_fg + precision_bg) / 2
# #     recall_bdy = (recall_fg + recall_bg) / 2
# #     f1_bdy = (f1_fg + f1_bg) / 2
# #     miou_bdy = (miou_fg + miou_bg) / 2
# #     oabdy = (fg_intersection + bg_intersection) / (height * width + 1e-6)
    
# #     return oabdy, precision_bdy, recall_bdy, f1_bdy, miou_bdy

# # def calculate_apls(pred, gt):
# #     """Calculate APLS metric between prediction and ground truth"""
# #     pred_mask = (pred == 1).astype(np.uint8)
# #     gt_mask = (gt == 1).astype(np.uint8)
    
# #     pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
# #     pred_lines = [shapely.geometry.LineString(c[:,0,:]) for c in pred_contours if len(c) >= 2]
# #     gt_lines = [shapely.geometry.LineString(c[:,0,:]) for c in gt_contours if len(c) >= 2]
    
# #     if not pred_lines or not gt_lines:
# #         return 0.0
    
# #     G_pred = nx.Graph()
# #     G_gt = nx.Graph()
    
# #     for line in pred_lines:
# #         for i in range(len(line.coords)-1):
# #             G_pred.add_edge(line.coords[i], line.coords[i+1])
    
# #     for line in gt_lines:
# #         for i in range(len(line.coords)-1):
# #             G_gt.add_edge(line.coords[i], line.coords[i+1])
    
# #     try:
# #         pred_paths = list(nx.all_pairs_shortest_path_length(G_pred))
# #         gt_paths = list(nx.all_pairs_shortest_path_length(G_gt))
        
# #         score = 0.0
# #         count = 0
# #         for u1, paths1 in pred_paths:
# #             for u2, dist1 in paths1.items():
# #                 if u1 == u2:
# #                     continue
# #                 try:
# #                     dist2 = nx.shortest_path_length(G_gt, u1, u2)
# #                     score += min(1.0, abs(dist1 - dist2) / max(dist1, dist2))
# #                     count += 1
# #                 except:
# #                     pass
        
# #         if count > 0:
# #             apls = 1.0 - (score / count)
# #         else:
# #             apls = 0.0
# #     except:
# #         apls = 0.0
    
# #     return apls

# # def calculate_ap_metrics(pred, gt):
# #     """Calculate APmask, APpoly, and additional MS-COCO metrics"""
# #     pred_mask = (pred == 1).astype(np.uint8)
# #     gt_mask = (gt == 1).astype(np.uint8)
    
# #     pred_labels = sk_label(pred_mask, connectivity=2)
# #     gt_labels = sk_label(gt_mask, connectivity=2)
    
# #     coco_gt = {
# #         "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
# #         "annotations": [],
# #         "categories": [{"id": 1, "name": "object"}]
# #     }
# #     coco_pred = {
# #         "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
# #         "annotations": [],
# #         "categories": [{"id": 1, "name": "object"}]
# #     }
    
# #     gt_props = regionprops(gt_labels)
# #     for idx, prop in enumerate(gt_props, 1):
# #         coords = prop.coords
# #         if len(coords) >= 4:
# #             poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
# #             if poly.is_valid:
# #                 coco_gt["annotations"].append({
# #                     "id": idx,
# #                     "image_id": 1,
# #                     "category_id": 1,
# #                     "segmentation": shapely.geometry.mapping(poly)["coordinates"],
# #                     "area": prop.area,
# #                     "bbox": list(prop.bbox),
# #                     "iscrowd": 0
# #                 })
    
# #     pred_props = regionprops(pred_labels)
# #     for idx, prop in enumerate(pred_props, 1):
# #         coords = prop.coords
# #         if len(coords) >= 4:
# #             poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
# #             if poly.is_valid:
# #                 coco_pred["annotations"].append({
# #                     "id": idx,
# #                     "image_id": 1,
# #                     "category_id": 1,
# #                     "segmentation": shapely.geometry.mapping(poly)["coordinates"],
# #                     "area": prop.area,
# #                     "bbox": list(prop.bbox),
# #                     "iscrowd": 0,
# #                     "score": 0.9
# #                 })
    
# #     if not coco_gt["annotations"] or not coco_pred["annotations"]:
# #         return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
# #     with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
# #         json.dump(coco_gt, f)
# #         gt_file = f.name
# #     with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
# #         json.dump(coco_pred, f)
# #         pred_file = f.name
    
# #     coco_gt_obj = COCO(gt_file)
# #     coco_pred_obj = COCO(pred_file)
    
# #     coco_eval_mask = COCOeval(coco_gt_obj, coco_pred_obj, 'segm')
# #     coco_eval_mask.evaluate()
# #     coco_eval_mask.accumulate()
# #     coco_eval_mask.summarize()
    
# #     ap_mask = coco_eval_mask.stats[0]
# #     ap_mask_50 = coco_eval_mask.stats[1]
# #     ap_mask_75 = coco_eval_mask.stats[2]
# #     ar_mask = coco_eval_mask.stats[8]
# #     ar_mask_50 = coco_eval_mask.stats[9]
# #     ar_mask_75 = coco_eval_mask.stats[10]
    
# #     ap_poly = ap_mask
# #     ap_poly_50 = ap_mask_50
# #     ap_poly_75 = ap_mask_75
# #     ar_poly = ar_mask
# #     ar_poly_50 = ar_mask_50
# #     ar_poly_75 = ar_mask_75
    
# #     os.remove(gt_file)
# #     os.remove(pred_file)
    
# #     return ap_mask, ap_poly, ap_poly_50, ap_poly_75, ar_poly, ar_poly_50, ar_poly_75

# # def calculate_polisi(pred, gt):
# #     """Calculate Polisi metric for instance segmentation"""
# #     pred_mask = (pred == 1).astype(np.uint8)
# #     gt_mask = (gt == 1).astype(np.uint8)
    
# #     pred_labels = sk_label(pred_mask, connectivity=2)
# #     gt_labels = sk_label(gt_mask, connectivity=2)
    
# #     pred_instances = len(np.unique(pred_labels)) - 1
# #     gt_instances = len(np.unique(gt_labels)) - 1
    
# #     if gt_instances == 0:
# #         return 0.0
    
# #     matched = 0
# #     for pred_id in np.unique(pred_labels):
# #         if pred_id == 0:
# #             continue
# #         pred_instance = (pred_labels == pred_id).astype(np.uint8)
# #         best_iou = 0
# #         for gt_id in np.unique(gt_labels):
# #             if gt_id == 0:
# #                 continue
# #             gt_instance = (gt_labels == gt_id).astype(np.uint8)
# #             intersection = np.logical_and(pred_instance, gt_instance).sum()
# #             union = np.logical_or(pred_instance, gt_instance).sum()
# #             iou = intersection / (union + 1e-6)
# #             best_iou = max(best_iou, iou)
# #         if best_iou > 0.5:
# #             matched += 1
    
# #     polisi = matched / gt_instances
# #     return polisi

# # def visualize_boundaries(original_img, pred_mask, save_path):
# #     """Draw predicted boundaries on the original image and save"""
# #     img_color = original_img if original_img.shape[2] == 3 else cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
# #     pred_mask = (pred_mask == 1).astype(np.uint8)
# #     contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
# #     img_with_contours = img_color.copy()
# #     cv2.drawContours(img_with_contours, contours, -1, (0, 0, 255), 1)
    
# #     cv2.imwrite(save_path, img_with_contours)

# # def process_and_evaluate(pred_path, gt_path, output_dir):
# #     """Main function to process data and evaluate"""
# #     pred_list = sorted(readimage(pred_path))
# #     gt_list = sorted(readimage(gt_path))
    
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
    
# #     log_path = os.path.join(output_dir, now + '.log')
# #     log = Logger(log_path, level='debug')
# #     log.logger.info('Starting evaluation of segmentation results')
    
# #     # Initialize metrics storage
# #     metrics = {
# #         'pa': [], 'mpa': [], 'miou': [], 'oabdy': [], 
# #         'f1_bdy': [], 'apls': [], 'ap_poly': [], 'polisi': []
# #     }
    
# #     for pred_file, gt_file in zip(pred_list, gt_list):
# #         try:
# #             start_time = datetime.now()
            
# #             # Load and process images
# #             pred = imageio.imread(pred_file)
# #             gt = imageio.imread(gt_file)
            
# #             # Convert to numpy arrays and ensure binary
# #             pred = np.array(pred)
# #             gt = np.array(gt)
            
# #             # Handle multi-channel images
# #             if len(pred.shape) > 2:
# #                 pred = pred[:,:,0]
# #             if len(gt.shape) > 2:
# #                 gt = gt[:,:,0]
            
# #             # Normalize to 0-1
# #             pred = (pred == 255).astype(np.uint8)
# #             gt = (gt == 255).astype(np.uint8)
            
# #             # Verify data
# #             if pred.shape != gt.shape:
# #                 log.logger.warning(f"尺寸不匹配: {pred_file} {pred.shape} vs {gt_file} {gt.shape}")
# #                 continue
                
# #             # Calculate metrics
# #             metric = SegmentationMetric(2)
# #             metric.addBatch(pred, gt)
            
# #             # Basic metrics
# #             pa = metric.pixelAccuracy()
# #             mpa, _ = metric.meanPixelAccuracy()
# #             miou, _ = metric.meanIntersectionOverUnion()
            
# #             # Boundary metrics
# #             oabdy, _, _, f1_bdy, _ = calculate_boundary_metrics(pred, gt)
            
# #             # Advanced metrics
# #             apls = calculate_apls(pred, gt)
# #             ap_poly = calculate_ap_metrics(pred, gt)[1]
# #             polisi = calculate_polisi(pred, gt)
            
# #             # Store metrics
# #             metrics['pa'].append(pa)
# #             metrics['mpa'].append(mpa)
# #             metrics['miou'].append(miou)
# #             metrics['oabdy'].append(oabdy)
# #             metrics['f1_bdy'].append(f1_bdy)
# #             metrics['apls'].append(apls)
# #             metrics['ap_poly'].append(ap_poly)
# #             metrics['polisi'].append(polisi)
            
# #             # Log individual results
# #             img_name = Path(pred_file).stem
# #             log.logger.info(f"\nEvaluation for {img_name}:")
# #             log.logger.info(f"Pixel Accuracy: {pa:.4f}")
# #             log.logger.info(f"Mean Pixel Accuracy: {mpa:.4f}")
# #             log.logger.info(f"Mean IoU: {miou:.4f}")
# #             log.logger.info(f"Boundary OA: {oabdy:.4f}")
# #             log.logger.info(f"Boundary F1: {f1_bdy:.4f}")
# #             log.logger.info(f"APLS: {apls:.4f}")
# #             log.logger.info(f"APpoly: {ap_poly:.4f}")
# #             log.logger.info(f"Polisi: {polisi:.4f}")
            
# #             # Visualization
# #             boundary_save = os.path.join(output_dir, f"boundary_{img_name}.png")
# #             visualize_boundaries(cv2.cvtColor(pred*255, cv2.COLOR_GRAY2BGR), pred, boundary_save)
            
# #             end_time = datetime.now()
# #             log.logger.info(f"Processing time: {(end_time - start_time).total_seconds():.2f}s")
            
# #         except Exception as e:
# #             log.logger.error(f"Error processing {pred_file}: {str(e)}")
# #             continue
    
# #     # Calculate and log average metrics
# #     if metrics['pa']:
# #         log.logger.info("\n===== Average Metrics =====")
# #         for key in metrics:
# #             avg = np.mean(metrics[key])
# #             std = np.std(metrics[key])
# #             log.logger.info(f"Average {key}: {avg:.4f} ± {std:.4f}")
        
# #         # Save metrics to CSV
# #         import pandas as pd
# #         df = pd.DataFrame(metrics)
# #         df.to_csv(os.path.join(output_dir, 'metrics_details.csv'), index=False)
        
# #         # Save averages
# #         avg_metrics = {f"avg_{k}": np.mean(v) for k, v in metrics.items()}
# #         with open(os.path.join(output_dir, 'average_metrics.json'), 'w') as f:
# #             json.dump(avg_metrics, f, indent=4)
# #     else:
# #         log.logger.warning("No valid images were processed")

# # def main():
# #     # Configuration
# #     pred_dir = "./outputs/lyg_hrnet48_300/seg"  # Prediction masks
# #     gt_dir = "./data/lyg/cut_300/val/gt"       # Ground truth masks
# #     output_dir = "./outputs/lyg_hrnet48_300/eval"  # Output directory
    
# #     # Run evaluation
# #     process_and_evaluate(pred_dir, gt_dir, output_dir)

# # if __name__ == "__main__":
# #     main()
# import os.path
# import torch
# import imageio
# import numpy as np
# from utils.SegmentationMetric import SegmentationMetric
# import glob
# from datetime import datetime
# from utils.util import Logger
# from pathlib import Path
# import sys
# import io
# import cv2
# import json
# import shapely.geometry
# import networkx as nx
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from skimage.measure import label as sk_label, regionprops
# import tempfile
# from skimage import measure
# from scipy.ndimage import binary_dilation
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8", line_buffering=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# now = datetime.now()
# now = str(datetime.now().date()) + "-" + str(datetime.now().time())[:5].replace(':', '-')

# def readimage(dir):
#     images_path_list = glob.glob(os.path.join(dir, "*.tif")) \
#                        + glob.glob(os.path.join(dir, "*.png")) \
#                        + glob.glob(os.path.join(dir, "*.tiff"))
#     return images_path_list

# def calculate_boundary_metrics(pred, gt):
#     """Calculate boundary metrics between prediction and ground truth based on boundary region maps"""
#     pred_mask = (pred == 1).astype(np.uint8)
#     gt_mask = (gt == 1).astype(np.uint8)
    
#     height, width = pred_mask.shape
#         # 定义膨胀核（应该在函数开头定义）
#     dilation_kernel = np.ones((5, 5), dtype=np.uint8)
#     # Boundary region masks
#     # GT boundary: edge detection + dilation
#     gt_edges = measure.find_contours(gt_mask, 0.5)
#     gt_bdy_mask = np.zeros((height, width), dtype=np.uint8)
#     if gt_edges:
#         for contour in gt_edges:
#             contour = contour.astype(int)
#             gt_bdy_mask[contour[:, 0], contour[:, 1]] = 1
#         # Use 5x5 kernel for dilation
#         dilation_kernel = np.ones((5, 5), dtype=np.uint8)
#         gt_bdy_mask = binary_dilation(gt_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
#     # Pred boundary: edge detection + dilation
#     pred_edges = measure.find_contours(pred_mask, 0.5)
#     pred_bdy_mask = np.zeros((height, width), dtype=np.uint8)
#     if pred_edges:
#         for contour in pred_edges:
#             contour = contour.astype(int)
#             pred_bdy_mask[contour[:, 0], contour[:, 1]] = 1
#         pred_bdy_mask = binary_dilation(pred_bdy_mask, structure=dilation_kernel).astype(np.uint8)
    
#     # Foreground boundary metrics
#     fg_intersection = np.logical_and(gt_bdy_mask, pred_bdy_mask).sum()
#     fg_union = np.logical_or(gt_bdy_mask, pred_bdy_mask).sum()
#     fg_gt_sum = gt_bdy_mask.sum()
#     fg_pred_sum = pred_bdy_mask.sum()
    
#     precision_fg = fg_intersection / (fg_pred_sum + 1e-6) if fg_pred_sum > 0 else 0
#     recall_fg = fg_intersection / (fg_gt_sum + 1e-6) if fg_gt_sum > 0 else 0
#     f1_fg = 2 * (precision_fg * recall_fg) / (precision_fg + recall_fg + 1e-6) if (precision_fg + recall_fg) > 0 else 0
#     miou_fg = fg_intersection / (fg_union + 1e-6) if fg_union > 0 else 0
    
#     # Background boundary metrics
#     gt_bdy_mask_bg = 1 - gt_bdy_mask
#     pred_bdy_mask_bg = 1 - pred_bdy_mask
#     bg_intersection = np.logical_and(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
#     bg_union = np.logical_or(gt_bdy_mask_bg, pred_bdy_mask_bg).sum()
#     bg_gt_sum = gt_bdy_mask_bg.sum()
#     bg_pred_sum = pred_bdy_mask_bg.sum()
    
#     precision_bg = bg_intersection / (bg_pred_sum + 1e-6) if bg_pred_sum > 0 else 0
#     recall_bg = bg_intersection / (bg_gt_sum + 1e-6) if bg_gt_sum > 0 else 0
#     f1_bg = 2 * (precision_bg * recall_bg) / (precision_bg + recall_bg + 1e-6) if (precision_bg + recall_bg) > 0 else 0
#     miou_bg = bg_intersection / (bg_union + 1e-6) if bg_union > 0 else 0
    
#     # Average boundary metrics
#     precision_bdy = (precision_fg + precision_bg) / 2
#     recall_bdy = (recall_fg + recall_bg) / 2
#     f1_bdy = (f1_fg + f1_bg) / 2
#     miou_bdy = (miou_fg + miou_bg) / 2
    
#     # Overall accuracy (OAbdy)
#     oabdy = (fg_intersection + bg_intersection) / (height * width + 1e-6)
    
#     return oabdy, precision_bdy, recall_bdy, f1_bdy, miou_bdy

# def calculate_apls(pred, gt):
#     """Calculate APLS metric between prediction and ground truth"""
#     pred_mask = (pred == 1).astype(np.uint8)
#     gt_mask = (gt == 1).astype(np.uint8)
    
#     pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     pred_lines = [shapely.geometry.LineString(c[:,0,:]) for c in pred_contours if len(c) >= 2]
#     gt_lines = [shapely.geometry.LineString(c[:,0,:]) for c in gt_contours if len(c) >= 2]
    
#     if not pred_lines or not gt_lines:
#         return 0.0
    
#     G_pred = nx.Graph()
#     G_gt = nx.Graph()
    
#     for line in pred_lines:
#         for i in range(len(line.coords)-1):
#             G_pred.add_edge(line.coords[i], line.coords[i+1])
    
#     for line in gt_lines:
#         for i in range(len(line.coords)-1):
#             G_gt.add_edge(line.coords[i], line.coords[i+1])
    
#     try:
#         pred_paths = list(nx.all_pairs_shortest_path_length(G_pred))
#         gt_paths = list(nx.all_pairs_shortest_path_length(G_gt))
        
#         score = 0.0
#         count = 0
#         for u1, paths1 in pred_paths:
#             for u2, dist1 in paths1.items():
#                 if u1 == u2:
#                     continue
#                 try:
#                     dist2 = nx.shortest_path_length(G_gt, u1, u2)
#                     score += min(1.0, abs(dist1 - dist2) / max(dist1, dist2))
#                     count += 1
#                 except:
#                     pass
        
#         if count > 0:
#             apls = 1.0 - (score / count)
#         else:
#             apls = 0.0
#     except:
#         apls = 0.0
    
#     return apls

# def calculate_ap_metrics(pred, gt):
#     """Calculate APmask, APpoly, and additional MS-COCO metrics"""
#     pred_mask = (pred == 1).astype(np.uint8)
#     gt_mask = (gt == 1).astype(np.uint8)
    
#     pred_labels = sk_label(pred_mask, connectivity=2)
#     gt_labels = sk_label(gt_mask, connectivity=2)
    
#     coco_gt = {
#         "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
#         "annotations": [],
#         "categories": [{"id": 1, "name": "object"}]
#     }
#     coco_pred = {
#         "images": [{"id": 1, "width": pred.shape[1], "height": pred.shape[0]}],
#         "annotations": [],
#         "categories": [{"id": 1, "name": "object"}]
#     }
    
#     gt_props = regionprops(gt_labels)
#     for idx, prop in enumerate(gt_props, 1):
#         coords = prop.coords
#         if len(coords) >= 4:
#             poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
#             if poly.is_valid:
#                 coco_gt["annotations"].append({
#                     "id": idx,
#                     "image_id": 1,
#                     "category_id": 1,
#                     "segmentation": shapely.geometry.mapping(poly)["coordinates"],
#                     "area": prop.area,
#                     "bbox": list(prop.bbox),
#                     "iscrowd": 0
#                 })
    
#     pred_props = regionprops(pred_labels)
#     for idx, prop in enumerate(pred_props, 1):
#         coords = prop.coords
#         if len(coords) >= 4:
#             poly = shapely.geometry.Polygon([(x, y) for y, x in coords])
#             if poly.is_valid:
#                 coco_pred["annotations"].append({
#                     "id": idx,
#                     "image_id": 1,
#                     "category_id": 1,
#                     "segmentation": shapely.geometry.mapping(poly)["coordinates"],
#                     "area": prop.area,
#                     "bbox": list(prop.bbox),
#                     "iscrowd": 0,
#                     "score": 0.9
#                 })
    
#     if not coco_gt["annotations"] or not coco_pred["annotations"]:
#         return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#         json.dump(coco_gt, f)
#         gt_file = f.name
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#         json.dump(coco_pred, f)
#         pred_file = f.name
    
#     coco_gt_obj = COCO(gt_file)
#     coco_pred_obj = COCO(pred_file)
    
#     coco_eval_mask = COCOeval(coco_gt_obj, coco_pred_obj, 'segm')
#     coco_eval_mask.evaluate()
#     coco_eval_mask.accumulate()
#     coco_eval_mask.summarize()
    
#     ap_mask = coco_eval_mask.stats[0]
#     ap_mask_50 = coco_eval_mask.stats[1]
#     ap_mask_75 = coco_eval_mask.stats[2]
#     ar_mask = coco_eval_mask.stats[8]
#     ar_mask_50 = coco_eval_mask.stats[9]
#     ar_mask_75 = coco_eval_mask.stats[10]
    
#     ap_poly = ap_mask
#     ap_poly_50 = ap_mask_50
#     ap_poly_75 = ap_mask_75
#     ar_poly = ar_mask
#     ar_poly_50 = ar_mask_50
#     ar_poly_75 = ar_mask_75
    
#     os.remove(gt_file)
#     os.remove(pred_file)
    
#     return ap_mask, ap_poly, ap_poly_50, ap_poly_75, ar_poly, ar_poly_50, ar_poly_75

# def calculate_polisi(pred, gt):
#     """Calculate Polisi metric for instance segmentation"""
#     pred_mask = (pred == 1).astype(np.uint8)
#     gt_mask = (gt == 1).astype(np.uint8)
    
#     pred_labels = sk_label(pred_mask, connectivity=2)
#     gt_labels = sk_label(gt_mask, connectivity=2)
    
#     pred_instances = len(np.unique(pred_labels)) - 1
#     gt_instances = len(np.unique(gt_labels)) - 1
    
#     if gt_instances == 0:
#         return 0.0
    
#     matched = 0
#     for pred_id in np.unique(pred_labels):
#         if pred_id == 0:
#             continue
#         pred_instance = (pred_labels == pred_id).astype(np.uint8)
#         best_iou = 0
#         for gt_id in np.unique(gt_labels):
#             if gt_id == 0:
#                 continue
#             gt_instance = (gt_labels == gt_id).astype(np.uint8)
#             intersection = np.logical_and(pred_instance, gt_instance).sum()
#             union = np.logical_or(pred_instance, gt_instance).sum()
#             iou = intersection / (union + 1e-6)
#             best_iou = max(best_iou, iou)
#         if best_iou > 0.5:
#             matched += 1
    
#     polisi = matched / gt_instances
#     return polisi

# def visualize_boundaries(original_img, pred_mask, save_path):
#     """Draw predicted boundaries on the original image and save"""
#     img_color = original_img if original_img.shape[2] == 3 else cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
#     pred_mask = (pred_mask == 1).astype(np.uint8)
#     contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     img_with_contours = img_color.copy()
#     cv2.drawContours(img_with_contours, contours, -1, (0, 0, 255), 1)
    
#     cv2.imwrite(save_path, img_with_contours)

# def evaluate_segmentation(pred_path, gt_path, output_dir):
#     """Evaluate segmentation results by comparing prediction and ground truth"""
#     pred_list = sorted(readimage(pred_path))
#     gt_list = sorted(readimage(gt_path))
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     log_path = os.path.join(output_dir, now + '.log')
#     log = Logger(log_path, level='debug')
#     log.logger.info('Starting evaluation of segmentation results')
    
#     for pred_file, gt_file in zip(pred_list, gt_list):
#         start_time = datetime.now()
        
#         # Load prediction and ground truth
#         pred = imageio.imread(pred_file)
#         gt = imageio.imread(gt_file)
#                     # 将0-255转换为0-1
#         # pred = (pred / 255).astype(np.uint8)  # 255设为1，否则0
#         pred = (pred == 255).astype(np.uint8)  # 仅255设为1，其他全0
        
#         # Basic segmentation metrics
#         metric = SegmentationMetric(2)  # Assuming binary segmentation
#         metric.addBatch(pred, gt)
#         pa = metric.pixelAccuracy()
#         mpa, cpa = metric.meanPixelAccuracy()
#         miou, per = metric.meanIntersectionOverUnion()
        
#         # Boundary metrics
#         oabdy, precision_bdy, recall_bdy, f1_bdy, miou_bdy = calculate_boundary_metrics(pred, gt)
        
#         # APLS metric
#         apls = calculate_apls(pred, gt)
        
#         # COCO-style metrics
#         ap_mask, ap_poly, ap_poly_50, ap_poly_75, ar_poly, ar_poly_50, ar_poly_75 = calculate_ap_metrics(pred, gt)
        
#         # Polisi metric
#         polisi = calculate_polisi(pred, gt)
        
#         end_time = datetime.now()
#         eval_time = (end_time - start_time).total_seconds()
        
#         # Save visualization
#         img_name = Path(pred_file).stem
#         boundary_save = os.path.join(output_dir, now + "_" + img_name + "-boundary.png")
#         visualize_boundaries(cv2.imread(pred_file.replace('_pred', '').replace('.tif', '.png')), pred, boundary_save)
        
#         # Log results
#         log.logger.info(f"Evaluation for {img_name}:")
#         log.logger.info(f"Pixel Accuracy: {pa:.4f}")
#         log.logger.info(f"Mean Pixel Accuracy: {mpa:.4f}")
#         log.logger.info(f"Class Pixel Accuracy: {cpa}")
#         log.logger.info(f"Mean IoU: {miou:.4f}")
#         log.logger.info(f"Per-class IoU: {per}")
#         log.logger.info(f"Boundary OA: {oabdy:.4f}")
#         log.logger.info(f"Boundary Precision: {precision_bdy:.4f}")
#         log.logger.info(f"Boundary Recall: {recall_bdy:.4f}")
#         log.logger.info(f"Boundary F1: {f1_bdy:.4f}")
#         log.logger.info(f"Boundary mIoU: {miou_bdy:.4f}")
#         log.logger.info(f"APLS: {apls:.4f}")
#         log.logger.info(f"APmask: {ap_mask:.4f}")
#         log.logger.info(f"APpoly: {ap_poly:.4f}")
#         log.logger.info(f"APpoly@50: {ap_poly_50:.4f}")
#         log.logger.info(f"APpoly@75: {ap_poly_75:.4f}")
#         log.logger.info(f"ARpoly: {ar_poly:.4f}")
#         log.logger.info(f"ARpoly@50: {ar_poly_50:.4f}")
#         log.logger.info(f"ARpoly@75: {ar_poly_75:.4f}")
#         log.logger.info(f"Polisi: {polisi:.4f}")
#         log.logger.info(f"Evaluation time: {eval_time:.3f}s")
#         log.logger.info("\n" + "="*80 + "\n")
#             if evaluation_mode == 'global':
#                 # 存储全局评估指标
#                 global_pa_list.append(pa)
#                 global_cpa_list.append(cpa)
#                 global_mpa_list.append(mpa)
#                 global_miou_list.append(miou)
#                 global_per_list.append(per)
#                 global_oabdy_list.append(oabdy)
#                 global_precision_bdy_list.append(precision_bdy)
#                 global_recall_bdy_list.append(recall_bdy)
#                 global_f1_bdy_list.append(f1_bdy)
#                 global_miou_bdy_list.append(miou_bdy)
#                 global_apls_list.append(apls)
#                 global_ap_mask_list.append(ap_mask)
#                 global_ap_poly_list.append(ap_poly)
#                 global_ap_mask_50_list.append(ap_mask_50)
#                 global_ap_mask_75_list.append(ap_mask_75)
#                 global_ar_mask_list.append(ar_mask)
#                 global_ar_mask_50_list.append(ar_mask_50)
#                 global_ar_mask_75_list.append(ar_mask_75)
#                 global_polisi_list.append(polisi)
#                 global_poly_speed_list.append(poly_speed)
#             else:
#                 # 单张评估模式下的日志记录
#                 log.logger.info(f"{img_name}的pa: {pa}")
#                 log.logger.info(f"{img_name}的cpa: {cpa}")
#                 log.logger.info(f"{img_name}的mpa: {mpa}")
#                 log.logger.info(f"{img_name}的mIoU: {miou}")
#                 log.logger.info(f"{img_name}的per: {per}")
#                 log.logger.info(f"{img_name}的边界OAbdy: {oabdy}")
#                 log.logger.info(f"{img_name}的边界Precision: {precision_bdy}")
#                 log.logger.info(f"{img_name}的边界Recall: {recall_bdy}")
#                 log.logger.info(f"{img_name}的边界F1-score: {f1_bdy}")
#                 log.logger.info(f"{img_name}的边界mIoU: {miou_bdy}")
#                 log.logger.info(f"{img_name}的APLS: {apls}")
#                 log.logger.info(f"{img_name}的APmask: {ap_mask}")
#                 log.logger.info(f"{img_name}的APmask_50: {ap_mask_50}")
#                 log.logger.info(f"{img_name}的APmask_75: {ap_mask_75}")
#                 log.logger.info(f"{img_name}的APpoly: {ap_poly}")
#                 log.logger.info(f"{img_name}的ARmask: {ar_mask}")
#                 log.logger.info(f"{img_name}的ARmask_50: {ar_mask_50}")
#                 log.logger.info(f"{img_name}的ARmask_75: {ar_mask_75}")
#                 log.logger.info(f"{img_name}的Polisi: {polisi}")
#                 log.logger.info(f"{img_name}的多边形化速度: {poly_speed:.3f}s")
#                 log.logger.info("\n-------------------------------------------------------------------")
        
#         if evaluation_mode == 'global':
#             # 计算并记录全局平均指标
#             log.logger.info("\n全局评估结果:")
#             log.logger.info(f"平均PA: {np.mean(global_pa_list):.4f}")
#             log.logger.info(f"平均CPA: {np.mean(global_cpa_list):.4f}")
#             log.logger.info(f"平均MPA: {np.mean(global_mpa_list):.4f}")
#             log.logger.info(f"平均mIoU: {np.mean(global_miou_list):.4f}")
#             log.logger.info(f"平均PER: {np.mean(global_per_list):.4f}")
#             log.logger.info(f"平均边界OA: {np.mean(global_oabdy_list):.4f}")
#             log.logger.info(f"平均边界Precision: {np.mean(global_precision_bdy_list):.4f}")
#             log.logger.info(f"平均边界Recall: {np.mean(global_recall_bdy_list):.4f}")
#             log.logger.info(f"平均边界F1-score: {np.mean(global_f1_bdy_list):.4f}")
#             log.logger.info(f"平均边界mIoU: {np.mean(global_miou_bdy_list):.4f}")
#             log.logger.info(f"平均APLS: {np.mean(global_apls_list):.4f}")
#             log.logger.info(f"平均APmask: {np.mean(global_ap_mask_list):.4f}")
#             log.logger.info(f"平均APmask_50: {np.mean(global_ap_mask_50_list):.4f}")
#             log.logger.info(f"平均APmask_75: {np.mean(global_ap_mask_75_list):.4f}")
#             log.logger.info(f"平均APpoly: {np.mean(global_ap_poly_list):.4f}")
#             log.logger.info(f"平均ARmask: {np.mean(global_ar_mask_list):.4f}")
#             log.logger.info(f"平均ARmask_50: {np.mean(global_ar_mask_50_list):.4f}")
#             log.logger.info(f"平均ARmask_75: {np.mean(global_ar_mask_75_list):.4f}")
#             log.logger.info(f"平均Polisi: {np.mean(global_polisi_list):.4f}")
#             log.logger.info(f"平均多边形化速度: {np.mean(global_poly_speed_list):.3f}s")
#             log.logger.info("\n-------------------------------------------------------------------")
        
#         print("\n")
# # Configuration
# pred_dir = r"./outputs/lyg_hrnet48_300/seg"  # Folder containing prediction masks
# gt_dir = r"./data/lyg/cut_300/val/gt"   # Folder containing ground truth masks
# output_dir = r"./outputs/lyg_hrnet48_300/eval"      # Folder to save evaluation results

# # Run evaluation
# evaluate_segmentation(pred_dir, gt_dir, output_dir)