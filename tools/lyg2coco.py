import os
import numpy as np
from skimage import io
import json
from tqdm import tqdm
from shapely.geometry import Polygon, mapping
from skimage.measure import label as ski_label
from skimage.measure import regionprops
import cv2
import glob
import math
import re

def extract_number_from_filename(filename):
    """从文件名中提取数字"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))  # 将匹配的字符串转换为整数
    else:
        return None  # 如果没有找到数字，返回None

def polygon2hbb(poly_coords):
    p_x = [point[0] for point in poly_coords]
    p_y = [point[1] for point in poly_coords]
    hbb_x = min(p_x)
    hbb_y = min(p_y)
    hbb_w = max(p_x) - hbb_x
    hbb_h = max(p_y) - hbb_y
    return [float(hbb_x), float(hbb_y), float(hbb_w), float(hbb_h)]

def bmask_to_poly(b_im, simplify_ind, tolerance=1.8):
    polygons = []
    try:
        label_img = ski_label(b_im > 0)
    except:
        print('error')
    props = regionprops(label_img)
    for prop in props:
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0]):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                contour -= 1
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    else:
                        if cv2.contourArea(closed_c.astype(int)) > 10:
                            intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
    return polygons

def clip_by_bound(poly, im_h, im_w):
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    p_x = np.clip(p_x, 0, im_w - 1)
    p_y = np.clip(p_y, 0, im_h - 1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)

if __name__ == '__main__':
    input_image_path = './data/sda/cut_512/val/img'
    input_gt_path = './data/sda/cut_512/val/gt'
    save_path = './data/sda/cut_512/val/'

    output_im_train = os.path.join(save_path, 'images')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)

    output_data_train = {
        'info': {'description': 'Aquaculture Dataset', 'year': 2025},
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'aquaculture', 'supercategory': 'aquaculture'}]
    }

    val_set = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(input_gt_path, "*.png")))[:5]]
    train_ob_id = 1

    input_label = sorted(os.listdir(input_gt_path))
    for g_id, label in enumerate(tqdm(input_label)):
        label_name = label.split('.')[0]
        print(f"Processing file: {label}")

        image_path = os.path.join(input_image_path, label_name + '.png')
        image_data = io.imread(image_path)
        gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.png'))

        im_h, im_w = gt_im_data.shape[:2]

        if label_name not in val_set:
            p_gt = gt_im_data
            p_im = image_data
            if np.sum(p_gt > 0) > 5:
                p_polygons = bmask_to_poly(p_gt, 1)
                for poly in p_polygons:
                    p_area = round(poly.area, 2)
                    if p_area > 0:
                        p_bbox = polygon2hbb(mapping(poly)['coordinates'][0])
                        if p_bbox[2] > 5 and p_bbox[3] > 5:
                            p_seg = []
                            coor_list = mapping(poly)['coordinates']
                            for part_poly in coor_list:
                                p_seg.append(np.asarray(part_poly).ravel().tolist())

                            image_id = extract_number_from_filename(label_name)
                            print(image_id)  # 输出: 1023
                            anno_info = {
                                'id': train_ob_id,
                                'image_id': image_id,
                                'segmentation': p_seg,
                                'area': p_area,
                                'bbox': p_bbox,
                                'category_id': 1,
                                'iscrowd': 0
                            }
                            output_data_train['annotations'].append(anno_info)
                            train_ob_id += 1

            image_info = {
                'id': image_id,
                'file_name': label_name + '.png',
                'width': im_w,
                'height': im_h
            }
            output_data_train['images'].append(image_info)
            io.imsave(os.path.join(output_im_train, label_name + '.png'), p_im)

    with open(os.path.join(save_path, 'annotation.json'), 'w') as f_json:
        json.dump(output_data_train, f_json)

# # Transform Inria gt dataset (binary image) to COCO format
# # Using cv2.findcontours and polygon simplify to convert raster label to vector label
# #
# # The first 5 images are kept as validation set

# from pycocotools.coco import COCO
# import os
# import numpy as np
# from skimage import io
# import json
# from tqdm import tqdm
# from itertools import groupby
# from shapely.geometry import Polygon, mapping
# from skimage.measure import label as ski_label
# from skimage.measure import regionprops
# from shapely.geometry import box
# import cv2
# # import glob
# import math
# import hashlib
# import re

# def extract_number_from_filename(filename):
#     """从文件名中提取数字"""
#     match = re.search(r'\d+', filename)
#     if match:
#         return int(match.group(0))  # 将匹配的字符串转换为整数
#     else:
#         return None  # 如果没有找到数字，返回None

# # # 示例使用
# # image_id = extract_number_from_filename(filename)
# # print(image_id)  # 输出: 1023


# def polygon2hbb(poly_coords):
#     """
#     Get horizontal bounding box (match COCO)
#     """
#     # poly_coords is expected to be a list of tuples, e.g., [[x1, y1], [x2, y2], ...]
#     p_x = [point[0] for point in poly_coords]  # Extract x coordinates
#     p_y = [point[1] for point in poly_coords]  # Extract y coordinates

#     hbb_x = min(p_x)  # Minimum x coordinate
#     hbb_y = min(p_y)  # Minimum y coordinate
#     hbb_w = max(p_x) - hbb_x  # Width
#     hbb_h = max(p_y) - hbb_y  # Height

#     hbox = [float(hbb_x), float(hbb_y), float(hbb_w), float(hbb_h)]
#     return hbox
# def polygon_in_bounding_box(polygon, bounding_box):
#     """
#     Returns True if all vertices of polygons are inside bounding_box
#     :param polygon: [N, 2]
#     :param bounding_box: [row_min, col_min, row_max, col_max]
#     :return:
#     """
#     result = np.all(
#         np.logical_and(
#             np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[0] + bounding_box[2]),
#             np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[1] + bounding_box[3])
#         )
#     )
#     return result

# def transform_poly_to_bounding_box(polygon, bounding_box):
#     """
#     Transform the original coordinates of polygon to bbox
#     :param polygon: [N, 2]
#     :param bounding_box: [row_min, col_min, row_max, col_max]
#     :return:
#     """
#     transformed_polygon = polygon.copy()
#     transformed_polygon[:, 0] -= bounding_box[0]
#     transformed_polygon[:, 1] -= bounding_box[1]
#     return transformed_polygon
# def clip_by_bound(poly, im_h, im_w):
#     """
#     Bound poly coordinates by image shape
#     """
#     p_x = poly[:, 0]
#     p_y = poly[:, 1]
#     p_x = np.clip(p_x, 0, im_w - 1)
#     p_y = np.clip(p_y, 0, im_h - 1)
#     return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)
# def bmask_to_poly(b_im, simplify_ind, tolerance=1.8, ):
#     """
#     Convert binary mask to polygons
#     """
#     polygons = []
#     # pad mask to close contours of shapes which start and end at an edge
#     try:
#         label_img = ski_label(b_im > 0)
#     except:
#         print('error')
#     props = regionprops(label_img)
#     for prop in props:
#         prop_mask = np.zeros_like(b_im)
#         prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
#         padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
#         contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours) > 1:
#             intp = []
#             for contour, h in zip(contours, hierarchy[0]):
#                 contour = np.array([c.reshape(-1).tolist() for c in contour])
#                 # subtract pad
#                 contour -= 1
#                 contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
#                 if len(contour) > 3:
#                     closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
#                     if h[3] < 0:
#                         extp = [tuple(i) for i in closed_c]
#                     else:
#                         if cv2.contourArea(closed_c.astype(int)) > 10:
#                             intp.append([tuple(i) for i in closed_c])
#             poly = Polygon(extp, intp)
#             if simplify_ind:
#                 poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
#                 if isinstance(poly, Polygon):
#                     polygons.append(poly)
#                 else:
#                     for idx in range(len(poly.geoms)):
#                         polygons.append(poly.geoms[idx])
#         elif len(contours) == 1:
#             contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
#             contour -= 1
#             contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
#             if len(contour) > 3:
#                 closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
#                 poly = Polygon(closed_c)

#             # simply polygon vertex
#                 if simplify_ind:
#                     poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
#                 if isinstance(poly, Polygon):
#                     polygons.append(poly)
#                 else:
#                     for idx in range(len(poly.geoms)):
#                         polygons.append(poly.geoms[idx])
#             # print(np.array(poly.exterior.coords).ravel().tolist())
#             # in case that after "simplify", one polygon turn to multiply polygons
#             # (pixels in polygon) are not connected
#     return polygons

# def rotate_image(image, angle):
#     """
#     Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
#     (in degrees). The returned image will be large enough to hold the entire
#     new image, with a black background
#     """

#     # Get the image size
#     # No that's not an error - NumPy stores image matricies backwards
#     image_size = (image.shape[1], image.shape[0])
#     image_center = tuple(np.array(image_size) / 2)

#     # Convert the OpenCV 3x2 rotation matrix to 3x3
#     rot_mat = np.vstack(
#         [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
#     )

#     rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

#     # Shorthand for below calcs
#     image_w2 = image_size[0] * 0.5
#     image_h2 = image_size[1] * 0.5

#     # Obtain the rotated coordinates of the image corners
#     rotated_coords = [
#         (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
#     ]

#     # Find the size of the new image
#     x_coords = [pt[0] for pt in rotated_coords]
#     x_pos = [x for x in x_coords if x > 0]
#     x_neg = [x for x in x_coords if x < 0]

#     y_coords = [pt[1] for pt in rotated_coords]
#     y_pos = [y for y in y_coords if y > 0]
#     y_neg = [y for y in y_coords if y < 0]

#     right_bound = max(x_pos)
#     left_bound = min(x_neg)
#     top_bound = max(y_pos)
#     bot_bound = min(y_neg)

#     new_w = int(abs(right_bound - left_bound))
#     new_h = int(abs(top_bound - bot_bound))

#     # We require a translation matrix to keep the image centred
#     trans_mat = np.matrix([
#         [1, 0, int(new_w * 0.5 - image_w2)],
#         [0, 1, int(new_h * 0.5 - image_h2)],
#         [0, 0, 1]
#     ])

#     # Compute the tranform for the combined rotation and translation
#     affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

#     # Apply the transform
#     result = cv2.warpAffine(
#         image,
#         affine_mat,
#         (new_w, new_h),
#         flags=cv2.INTER_LINEAR
#     )

#     return result

# def largest_rotated_rect(w, h, angle):
#     """
#     Given a rectangle of size wxh that has been rotated by 'angle' (in
#     radians), computes the width and height of the largest possible
#     axis-aligned rectangle within the rotated rectangle.

#     Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

#     Converted to Python by Aaron Snoswell
#     """

#     quadrant = int(math.floor(angle / (math.pi / 2))) & 3
#     sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
#     alpha = (sign_alpha % math.pi + math.pi) % math.pi

#     bb_w = w * math.cos(alpha) + h * math.sin(alpha)
#     bb_h = w * math.sin(alpha) + h * math.cos(alpha)

#     gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

#     delta = math.pi - alpha - gamma

#     length = h if (w < h) else w

#     d = length * math.cos(alpha)
#     a = d * math.sin(alpha) / math.sin(delta)

#     y = a * math.cos(gamma)
#     x = y * math.tan(gamma)

#     return (
#         bb_w - 2 * x,
#         bb_h - 2 * y
#     )

# def crop_around_center(image, width, height):
#     """
#     Given a NumPy / OpenCV 2 image, crops it to the given width and height,
#     around it's centre point
#     """

#     image_size = (image.shape[1], image.shape[0])
#     image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

#     if(width > image_size[0]):
#         width = image_size[0]

#     if(height > image_size[1]):
#         height = image_size[1]

#     x1 = int(image_center[0] - width * 0.5)
#     x2 = int(image_center[0] + width * 0.5)
#     y1 = int(image_center[1] - height * 0.5)
#     y2 = int(image_center[1] + height * 0.5)

#     return image[y1:y2, x1:x2]

# def rotate_crop(im, gt, crop_size, angle):
#     h, w = im.shape[0:2]
#     im_rotated = rotate_image(im, angle)
#     gt_rotated = rotate_image(gt, angle)
#     if largest_rotated_rect(w, h, math.radians(angle))[0] > crop_size:
#         im_cropped = crop_around_center(im_rotated, crop_size, crop_size)
#         gt_cropped = crop_around_center(gt_rotated, crop_size, crop_size)
#     else:
#         print('error')
#         im_cropped = crop_around_center(im, crop_size, crop_size)
#         gt_cropped = crop_around_center(gt, crop_size, crop_size)
#     return im_cropped, gt_cropped

# def lt_crop(im, gt, crop_size):
#     im_cropped = im[0:crop_size, 0:crop_size, :]
#     gt_cropped = gt[0:crop_size, 0:crop_size]
#     return im_cropped, gt_cropped

# def affine_transform(pt, t):
#     new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
#     new_pt = np.dot(t, new_pt)
#     return new_pt[:2]
# if __name__ == '__main__':
#     input_image_path = './data/lyg/cut_512/val/img'
#     input_gt_path = './data/lyg/cut_512/val/gt'
#     save_path = './data/lyg/cut_512/val/'

#     output_im_train = os.path.join(save_path, 'images')
#     if not os.path.exists(output_im_train):
#         os.makedirs(output_im_train)

#     output_data_train = {
#         'info': {'description': 'Aquaculture Dataset', 'year': 2024},
#         'images': [],
#         'annotations': [],
#         'categories': [{'id': 1, 'name': 'aquaculture', 'supercategory': 'aquaculture'}]
#     }

#     val_set = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(input_gt_path, "*.png")))[:5]]
#     train_ob_id = 1

#     input_label = sorted(os.listdir(input_gt_path))
#     for g_id, label in enumerate(tqdm(input_label)):
#         label_name = label.split('.')[0]
#         print(f"Processing file: {label}")

#         image_path = os.path.join(input_image_path, label_name + '.png')
#         image_data = io.imread(image_path)
#         gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.png'))

#         im_h, im_w = gt_im_data.shape[:2]

#         if label_name not in val_set:
#             p_gt = gt_im_data
#             p_im = image_data
#             if np.sum(p_gt > 0) > 5:
#                 p_polygons = bmask_to_poly(p_gt, 1)
#                 for poly in p_polygons:
#                     p_area = round(poly.area, 2)
#                     if p_area > 0:
#                         p_bbox = polygon2hbb(mapping(poly)['coordinates'][0])
#                         if p_bbox[2] > 5 and p_bbox[3] > 5:
#                             p_seg = []
#                             coor_list = mapping(poly)['coordinates']
#                             for part_poly in coor_list:
#                                 p_seg.append(np.asarray(part_poly).ravel().tolist())

#                             # image_id = get_image_id_from_name(label_name + '.png')  # 使用文件名的哈希值
#                             # 示例使用
#                             image_id = extract_number_from_filename(label_name)
#                             print(image_id)  # 输出: 1023
#                             anno_info = {
#                                 'id': train_ob_id,
#                                 'image_id': image_id,
#                                 'segmentation': p_seg,
#                                 'area': p_area,
#                                 'bbox': p_bbox,
#                                 'category_id': 1,
#                                 'iscrowd': 0
#                             }
#                             output_data_train['annotations'].append(anno_info)
#                             train_ob_id += 1

#             image_info = {
#                 'id': image_id,
#                 'file_name': label_name + '.png',
#                 'width': im_w,
#                 'height': im_h
#             }
#             output_data_train['images'].append(image_info)
#             io.imsave(os.path.join(output_im_train, label_name + '.png'), p_im)

#     with open(os.path.join(save_path, 'annotation_hash.json'), 'w') as f_json:
#         json.dump(output_data_train, f_json)

# # if __name__ == '__main__':
# #   # 保存路径
# #     input_image_path = './data/lyg/cut_512/train/img'
# #     input_gt_path = './data/lyg/cut_512/train/gt'
# #     save_path = './data/lyg/cut_512/train/'  

# #     output_im_train = os.path.join(save_path, 'images')
# #     if not os.path.exists(output_im_train):
# #         os.makedirs(output_im_train)

# #     # 初始化COCO格式的数据结构
# #     output_data_train = {
# #         'info': {'description': 'Aquaculture Dataset', 'year': 2024},
# #         'images': [],
# #         'annotations': [],
# #         'categories': [{'id': 1, 'name': 'aquaculture', 'supercategory': 'aquaculture'}]
# #     }
# #     val_set = []  # 或者根据实际情况填充这个列表
# #     train_ob_id = 1  # 从1开始计数
# #     image_id = get_image_id_from_name(label_name + '.png')  # 获取当前图像的哈希值作为image_id
# # #     input_label = os.listdir(input_gt_path)
    
# #     input_label = sorted(os.listdir(input_gt_path))   
# #     for g_id, label in enumerate(tqdm(input_label)):
        
# #         label_name = label.split('.')[0]  # 假设文件名不含扩展名以外的点
# #         print(f"Processing file: {label}")  # 打印正在处理的文件名，以便调试

# # #         image_data = io.imread(os.path.join(input_image_path, label_name + '.png'))  # 假设图像为png格式

# #         image_path = os.path.join(input_image_path, label_name + '.png')  # 确保这里的路径和扩展名与实际文件匹配
# #         image_data = io.imread(image_path)
        
# #         gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.png'))  # 假设标注为png格式
# #         if 'gt_im_data' in locals():
# #             im_h, im_w = gt_im_data.shape[:2]
# #             print(f"Ground truth image dimensions: Height={im_h}, Width={im_w}")
# #         else:
# #             print(f"Failed to read ground truth image: {gt_image_path}")

# #         # 如果图像数据成功读取，继续后续处理
# #         if 'image_data' in locals():
# #             print(f"Successfully read image: {image_path}")
# #         else:
# #             print(f"Failed to read image: {image_path}")
# #         im_h, im_w = gt_im_data.shape[:2]

# #         # 只处理训练集，不分割图像为多个patches
# #         if label_name not in val_set:  # 假设val_set包含验证集的label_name
# #             p_gt = gt_im_data
# #             p_im = image_data
# #             if np.sum(p_gt > 0) > 5:
# #                 p_polygons = bmask_to_poly(p_gt, 1)
# #                 for poly in p_polygons:
# #                     p_area = round(poly.area, 2)
# #                     if p_area > 0:
# #                         p_bbox = polygon2hbb(mapping(poly)['coordinates'][0])  # 使用polygon2hbb函数
# #                         # 假设poly是一个shapely.geometry.Polygon对象
# # #                         p_bbox = polygon2hbb(list(mapping(poly)['coordinates']))
# #                         # 假设poly是一个shapely.geometry.Polygon对象
# # #                         p_bbox = polygon2hbb(mapping(poly)['coordinates'])
# #                         # 假设poly是一个shapely.geometry.Polygon对象
# #                         if p_bbox[2] > 5 and p_bbox[3] > 5:
# #                             p_seg = []
# #                             coor_list = mapping(poly)['coordinates']
# #                             for part_poly in coor_list:
# #                                 p_seg.append(np.asarray(part_poly).ravel().tolist())
# #                             anno_info = {
# #                                 'id': train_ob_id,
# #                                 'image_id': train_im_id,
# #                                 'segmentation': p_seg,
# #                                 'area': p_area,
# #                                 'bbox': p_bbox,
# #                                 'category_id': 1,  # 修改类别ID
# #                                 'iscrowd': 0
# #                             }
# #                             output_data_train['annotations'].append(anno_info)
# #                             train_ob_id += 1
# #             # 添加图像信息
# #             image_info = {
# #                 'id': train_im_id,
# #                 'file_name': label_name + '.png',  # 假设图像为png格式
# #                 'width': im_w,
# #                 'height': im_h
# #             }
# #             output_data_train['images'].append(image_info)
# #             # 保存图像
# #             io.imsave(os.path.join(output_im_train, label_name + '.png'), p_im)  # 保存图像为png格式
# #             train_im_id += 1

# #     # 保存COCO格式的标注文件
# #     with open(os.path.join(save_path, 'annotation_hash.json'), 'w') as f_json:
# #         json.dump(output_data_train, f_json)
