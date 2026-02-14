#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import argparse
color_map = {
    1: ((0, 255, 0), (0, 0, 255)),  # Green lines, Red vertices for category 1
    2: ((255, 0, 0), (255, 255, 0))  # Blue lines, Yellow vertices for category 2
}

import cv2
import numpy as np

def draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon=True, draw_bbox=True,
                                      color_map={1: ((0, 255, 0), (0, 0, 255)),  # category_id: (line_color, vertex_color)
                                                 2: ((255, 0, 0), (255, 255, 0))},
                                      bbox_color=(0, 255, 255),  # Default bbox color
                                      line_thickness=1, vertex_thickness=4):
    """
    Draw polygons and bounding boxes on an image based on category_id.

    Parameters:
    - image (numpy.ndarray): Input image.
    - annotations (list): List of annotations containing polygons and bounding boxes.
    - draw_polygon (bool): Whether to draw polygons. Default is True.
    - draw_bbox (bool): Whether to draw bounding boxes. Default is True.
    - color_map (dict): Mapping of category_id to (line_color, vertex_color).
    - bbox_color (tuple): Color of the bounding boxes.
    - line_thickness (int): Thickness of the polygon lines.
    - vertex_thickness (int): Thickness of the polygon vertices.

    Returns:
    - numpy.ndarray: Image with drawn polygons and bounding boxes.
    """
    for annotation in annotations:
        category_id = annotation['category_id']
        if category_id in color_map:
            line_color, vertex_color = color_map[category_id]
        else:
            line_color = (255, 255, 255)  # Default line color for unknown category
            vertex_color = (0, 0, 0)  # Default vertex color for unknown category

        if draw_polygon:
            segmentation = annotation['segmentation'][0]
            polygon = np.array(segmentation, np.int32).reshape((-1, 2))
            cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
            for vertex in polygon:
                cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

        if draw_bbox:
            bbox = annotation['bbox']
            x, y, w, h = bbox
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

    return image
# import cv2
# import numpy as np

# def draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon=True, draw_bbox=True,
#                                       color_map={1: ((0, 255, 0), (0, 0, 255)),  # category_id: (line_color, vertex_color)
#                                                  2: ((255, 0, 0), (255, 255, 0))},
#                                       bbox_color=(0, 255, 255),  # Bbox color (Cyan in BGR)
#                                       line_thickness=1, vertex_thickness=4):
#     """
#     Draw polygons and bounding boxes on an image based on category_id.
#     Assumes input image is in RGB format and converts it for OpenCV (BGR) drawing.

#     Parameters:
#     - image (numpy.ndarray): Input image in RGB channel order.
#     - annotations (list): List of annotations.
#     ...
#     Returns:
#     - numpy.ndarray: Image with drawings, in BGR channel order.
#     """
#     # 1. --- 正确的通道转换 ---
#     # 首先，复制图像以避免修改原始输入
#     # 然后，将整个图像从 RGB 转换为 BGR，因为 OpenCV 使用 BGR
#     # 注意：如果您的输入图像已经是 BGR，请删除下面这行
#     draw_image = image.copy()
#     if draw_image.shape[2] == 3: # 确保是彩色图像
#         draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)

#     for annotation in annotations:
#         category_id = annotation['category_id']
#         if category_id in color_map:
#             # Colors in color_map are defined as (R, G, B), 
#             # but OpenCV expects (B, G, R). Our cvtColor handles this.
#             # If not using cvtColor, you'd need to reverse colors here.
#             line_color, vertex_color = color_map[category_id]
#         else:
#             line_color = (255, 255, 255)  # Default: White
#             vertex_color = (0, 0, 0)      # Default: Black

#         if draw_polygon:
#             segmentation = annotation['segmentation'][0]
#             polygon = np.array(segmentation, np.int32).reshape((-1, 2))
            
#             # --- 2. 在转换后的图像上绘图 ---
#             cv2.polylines(draw_image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
#             for vertex in polygon:
#                 cv2.circle(draw_image, tuple(vertex), vertex_thickness, vertex_color, -1)

#         if draw_bbox:
#             bbox = annotation['bbox']
#             x, y, w, h = bbox
#             top_left = (int(x), int(y))
#             bottom_right = (int(x + w), int(y + h))
#             cv2.rectangle(draw_image, top_left, bottom_right, bbox_color, 2)

#     return draw_image
# def draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon=True, draw_bbox=True, line_color=(0, 255, 0),
#                                   line_thickness=1, vertex_color=(0, 0, 255), vertex_thickness=4,
#                                   bbox_color=(255, 0, 0)):
#     """
#     Draw polygons and bounding boxes on an image.

#     Parameters:
#     - image (numpy.ndarray): Input image.
#     - annotations (list): List of annotations containing polygons and bounding boxes.
#     - draw_polygon (bool): Whether to draw polygons. Default is True.
#     - draw_bbox (bool): Whether to draw bounding boxes. Default is True.
#     - line_color (tuple): Color of the polygon edges (default: green).
#     - vertex_color (tuple): Color of the vertices (default: red).
#     - vertex_thickness (int): Thickness of the vertices (default: 2).
#     - bbox_color (tuple): Color of the bounding boxes (default: blue).

#     Returns:
#     - numpy.ndarray: Image with drawn polygons and bounding boxes.
#     """
#     for annotation in annotations:
#         if draw_polygon:
#             segmentation = annotation['segmentation'][0]
#             polygon = np.array(segmentation, np.int32).reshape((-1, 2))
#             cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
#             for vertex in polygon:
#                 cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

#         if draw_bbox:
#             bbox = annotation['bbox']
#             x, y, w, h = bbox
#             top_left = (int(x), int(y))
#             bottom_right = (int(x + w), int(y + h))
#             cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

#     return image


def process_annotations(coco_json_file, png_folder, output_folder, mask_output_folder, draw_polygon=True,
                        draw_bbox=True, draw_mask=True):
    """
    Process a COCO dataset, drawing polygons, bounding boxes, and masks as specified.

    Parameters:
    - coco_json_file (str): Path to the COCO-format annotation file.
    - png_folder (str): Path to the folder containing PNG images.
    - output_folder (str): Path to save images with outlines.
    - mask_output_folder (str): Path to save mask images.
    - draw_polygon (bool): Whether to draw polygons on the images.
    - draw_bbox (bool): Whether to draw bounding boxes on the images.
    - draw_mask (bool): Whether to draw masks for the images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(png_folder, file_name)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = []

        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                annotations.append(annotation)

        image_with_polygons_and_bboxes = draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon, draw_bbox)

        if draw_mask:
            for annotation in annotations:
                segmentation = annotation['segmentation']
                exterior = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [exterior], 255)
                if len(segmentation) > 1:
                    for interior in segmentation[1:]:
                        interior_points = np.array(interior).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [interior_points], 0)

            mask_output_path = os.path.join(mask_output_folder, file_name)
            cv2.imwrite(mask_output_path, mask)
            print(f"Saved mask: {mask_output_path}")

        output_image_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_image_path, image_with_polygons_and_bboxes)
        print(f"Saved annotated image: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw polygons, bounding boxes, and masks on COCO dataset images.")
    parser.add_argument("--coco_json_file", type=str, required=True, help="Path to the COCO-format annotation file.")
    parser.add_argument("--png_folder", type=str, required=True, help="Path to the folder containing PNG images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save images with annotations.")
    parser.add_argument("--mask_output_folder", type=str, required=True, help="Path to save mask images.")
    parser.add_argument("--draw_polygon", action="store_true", help="Whether to draw polygons on the images.")
    parser.add_argument("--draw_bbox", action="store_true", help="Whether to draw bounding boxes on the images.")
    parser.add_argument("--draw_mask", action="store_true", help="Whether to draw masks for the images.")
    args = parser.parse_args()

    process_annotations(
        coco_json_file=args.coco_json_file,
        png_folder=args.png_folder,
        output_folder=args.output_folder,
        mask_output_folder=args.mask_output_folder,
        draw_polygon=args.draw_polygon,
        draw_bbox=args.draw_bbox,
        draw_mask=args.draw_mask
    )


# python visualize_annotations.py --coco_json_file ../data/yangzhiqu/lyg_3/cut_300/val/annotation.json \
#                                  --png_folder ../data/yangzhiqu/lyg_3/cut_300/val/images \
#                                  --output_folder ../data/yangzhiqu/lyg_3/cut_300/val/new/outlines \
#                                  --mask_output_folder ../data/yangzhiqu/lyg_3/cut_300/val/new/masks \
#                                  --draw_polygon 

# python visualize_annotations.py --coco_json_file /qiaowenjiao/HiSup/data/dongtou/geo/cut_2048_u8_new/train/annotation.json \
#                                  --png_folder /qiaowenjiao/HiSup/data/dongtou/geo/cut_2048_u8_new/train/images \
#                                  --output_folder /qiaowenjiao/HiSup/data/dongtou/geo/cut_2048_u8_new/train/vis/outlines \
#                                  --mask_output_folder /qiaowenjiao/HiSup/data/dongtou/geo/cut_2048_u8_new/train/vis/masks \
#                                  --draw_polygon --draw_bbox
# python vis_ann.py \
#     --coco_json_file ../data/sda/cut_512/val/annotation.json \
#     --png_folder ../data/sda/cut_512/val/images \
#     --output_folder ../data/sda/cut_512/val/outlines \
#     --mask_output_folder ../data/sda/cut_512/val/masks \
#     --draw_polygon
