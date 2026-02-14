import os
import os.path as osp
import json
import torch
import logging
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from skimage import io
from skimage.measure import label, regionprops
from shapely.geometry import Polygon
from osgeo import ogr, osr  # 修复 ogr 未定义问题
from pycocotools import mask as coco_mask
from hisup.utils.comm import to_single_device
from hisup.utils.polygon import generate_polygon, juncs_in_bbox, get_pred_junctions
from hisup.utils.visualizer import *
from hisup.dataset import build_test_dataset
from hisup.dataset.build import build_transform
# from tools.evaluation import coco_eval, boundary_eval, polis_eval
from tools.evaluation_new import *
from pyproj import Proj
# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def poly_to_bbox(poly):
    lt_x = np.min(poly[:, 0])
    lt_y = np.min(poly[:, 1])
    w = np.max(poly[:, 0]) - lt_x
    h = np.max(poly[:, 1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(polys, scores, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        if polygon.shape[0] < 3:
            continue
        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
            'image_id': img_id,
            'category_id': 1,
            'segmentation': [vec_poly],
            'bbox': poly_bbox,
            'score': float(scores[i]),
        }
        sample_ann.append(ann_per_building)
    return sample_ann

def generate_coco_mask(mask, img_id):
    sample_ann = []
    props = regionprops(label(mask > 0.50))
    for prop in props:
        if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
            prop_mask = np.zeros_like(mask, dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
            masked_instance = np.ma.masked_array(mask, mask=(prop_mask != 1))
            score = masked_instance.mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            ann_per_building = {
                'image_id': img_id,
                'category_id': 1,
                'segmentation': {
                    "size": encoded_region["size"],
                    "counts": encoded_region["counts"].decode()
                },
                'score': float(score),
            }
            sample_ann.append(ann_per_building)
    return sample_ann

def simplify_polygon(polygon, tolerance=0.5):
    """简化多边形，确保至少 4 个坐标点"""
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    if len(simplified.exterior.coords) < 4:
        logger.debug(f"简化后坐标数减少至 {len(simplified.exterior.coords)}，返回原始多边形")
        return polygon
    return simplified

def save_shapefile_from_polygons(polygons, image_path, output_shapefile_path, resolution):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(output_shapefile_path)
    layer = data_source.CreateLayer("buildings", geom_type=ogr.wkbPolygon)

    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(24)
    layer.CreateField(field_name)

    try:
        src_image = Image.open(image_path)
    except Exception as e:
        logger.error(f"无法打开图像文件: {image_path}, {e}")
        data_source = None
        return

    proj = Proj('+proj=latlong +datum=WGS84')

    for i, poly in enumerate(polygons):
        try:
            if not poly.is_valid or len(poly.exterior.coords) < 4:
                logger.warning(f"跳过无效多边形 {i}，坐标数: {len(poly.exterior.coords)}")
                continue
            simplified_poly = simplify_polygon(poly, tolerance=0.5)

            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("Name", f"Building_{i}")
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in zip(simplified_poly.exterior.xy[0], simplified_poly.exterior.xy[1]):
                lon, lat = proj(x, y, inverse=True)
                ring.AddPoint(lon, lat)
            poly_geom = ogr.Geometry(ogr.wkbPolygon)
            poly_geom.AddGeometry(ring)
            feature.SetGeometry(poly_geom)
            layer.CreateFeature(feature)
            feature = None
        except Exception as e:
            logger.error(f"处理多边形 {i} 失败: {e}")

    data_source = None
    logger.info(f"Shapefile 保存成功: {output_shapefile_path}")

class TestPipeline:
    def __init__(self, cfg, eval_type='pixel_boundary'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        # self.output_dir = cfg.OUTPUT_DIR
        self.output_dir = osp.abspath(cfg.OUTPUT_DIR)  # ✅ 转换为绝对路径

        self.dataset_name = cfg.DATASETS.TEST[0]
        # self.eval_type = eval_type
        self.eval_type = 'pixel_boundary'

        self.gt_file = ''
        self.dt_file = ''
    
    def test(self, model):
        if 'crowdai' in self.dataset_name:
            self.test_on_crowdai(model, self.dataset_name)
        elif 'inria' in self.dataset_name:
            self.test_on_inria(model, self.dataset_name)
        else:
            # self.test_on_inria(model, self.dataset_name)
            self.test_on_crowdai(model, self.dataset_name)


    def eval(self):
        logger.info('Evalutating on {}'.format(self.eval_type))
        if self.eval_type == 'coco_iou':
            coco_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'boundary_iou':
            boundary_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'polis':
            polis_eval(self.gt_file, self.dt_file)
        else:
            pixel_boundary_eval(self.gt_file, self.dt_file)
            

    def test_on_crowdai(self, model, dataset_name):
        logger.info('Testing on {} dataset'.format(dataset_name))
        
        results = []
        mask_results = []
        test_dataset, gt_file = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']

            for b in range(batch_size):
                filename = annotations[b]['filename']
                img_id = int(filename[:-4])

                scores = batch_scores[b]
                polys = batch_polygons[b]
                mask_pred = batch_masks[b]

                image_result = generate_coco_ann(polys, scores, img_id)
                if len(image_result) != 0:
                    results.extend(image_result)

                image_masks = generate_coco_mask(mask_pred, img_id)
                if len(image_masks) != 0:
                    mask_results.extend(image_masks)
        
        dt_file = osp.join(self.output_dir, '{}.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name, dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

        dt_file = osp.join(self.output_dir, '{}_mask.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name, dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(mask_results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

    def test_on_inria(self, model, dataset_name):
        logger.info(f'Testing on {dataset_name} dataset')

        # 更新路径以匹配您的环境
        # IM_PATH = '/root/shared-nvme/yuan/HiSup-main/data/lyg/pre/pre/'
        # IM_PATH = '/root/shared-nvme/HiSup-main/data/dt/20230216/cut_512/val/images'
        # IM_PATH = '/root/shared-nvme/HiSup-main/data/lyg/20231021/cut_512/val/images'
        # IM_PATH = '/root/shared-nvme/HiSup-main/data/lyg/20210221/cut_512/val/images'

        # IM_PATH = '/root/shared-nvme/HiSup-main/data/'
        IM_PATH = '/root/shared-nvme/HiSup-main/data/lyg/cut_300/val'

        # output_dir = osp.join(self.output_dir, 'seg')
        # print('self.output_dir',self.output_dir)
        # shp_output_dir = osp.join(self.output_dir, 'shp')
        # print('shp_output_dir',shp_output_dir)

        # os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(shp_output_dir, exist_ok=True)
     # ✅ 确保 output_dir 是绝对路径
        # self.output_dir = osp.abspath(self.output_dir)  # 转换为绝对路径
        
        # ✅ 正确拼接 seg 和 shp 子目录
        output_dir = osp.join(self.output_dir, 'seg')
        shp_output_dir = osp.join(self.output_dir, 'shp')
    
        # ✅ 打印路径检查
        print(f"Seg 保存路径: {output_dir}")
        print(f"Shp 保存路径: {shp_output_dir}")
    
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(shp_output_dir, exist_ok=True)
        transform = build_transform(self.cfg)
        test_imgs = [f for f in os.listdir(IM_PATH) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
        results = []
        mask_results = []

        for image_name in tqdm(test_imgs, desc='Total processing'):
            impath = osp.join(IM_PATH, image_name)
            try:
                image = io.imread(impath)
            except Exception as e:
                logger.error(f"无法读取图像 {impath}: {e}")
                continue

            # h_stride, w_stride = 400, 400
            # h_crop, w_crop = 512, 512
            h_stride, w_stride = 200, 200
            h_crop, w_crop = 300, 300
            h_img, w_img, _ = image.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
            count_mat = np.zeros([h_img, w_img])
            juncs_whole_img = []

            patch_weight = np.ones((h_crop + 2, w_crop + 2))
            patch_weight[0, :] = 0
            patch_weight[-1, :] = 0
            patch_weight[:, 0] = 0
            patch_weight[:, -1] = 0
            patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)[1:-1, 1:-1]

            for h_idx in tqdm(range(h_grids), leave=False, desc='processing on per image'):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)

                    crop_img = image[y1:y2, x1:x2, :]
                    try:
                        crop_img_tensor = transform(crop_img.astype(float)).unsqueeze(0).to(self.device)
                    except Exception as e:
                        logger.error(f"图像变换失败 {impath} 在位置 ({y1},{x1},{y2},{x2}): {e}")
                        continue

                    meta = {
                        'filename': impath,
                        'height': crop_img.shape[0],
                        'width': crop_img.shape[1],
                        'pos': [x1, y1, x2, y2]
                    }

                    with torch.no_grad():
                        try:
                            output, _ = model(crop_img_tensor, [meta])
                            output = to_single_device(output, 'cpu')
                            juncs_pred = output['juncs_pred'][0] + [x1, y1]
                            juncs_whole_img.extend(juncs_pred.tolist())
                            mask_pred = output['mask_pred'][0] * patch_weight
                            pred_whole_img[y1:y2, x1:x2] += mask_pred
                            count_mat[y1:y2, x1:x2] += patch_weight
                        except Exception as e:
                            logger.error(f"模型推理失败 {impath}: {e}")
                            continue

            juncs_whole_img = np.array(juncs_whole_img)
            try:
                pred_whole_img /= count_mat
            except RuntimeWarning:
                logger.warning(f"归一化时出现除零警告 {impath}")

            polygons = []
            props = regionprops(label(pred_whole_img > 0.5))
            logger.debug(f"检测到的区域数量 for {image_name}: {len(props)}")
            for prop in tqdm(props, leave=False, desc='polygon generation'):
                y1, x1, y2, x2 = prop.bbox
                bbox = [x1, y1, x2, y2]
                select_juncs = juncs_in_bbox(bbox, juncs_whole_img, expand=8)
                poly, juncs_sa, _, score, juncs_index = generate_polygon(prop, pred_whole_img, select_juncs, pid=0, test_inria=True)
                
                if len(poly) == 0 or juncs_sa.shape[0] == 0:
                    logger.debug(f"跳过空的或无效的多边形，prop {prop.label}, juncs 数量={len(select_juncs)}")
                    continue
                
                try:
                    if len(juncs_index) == 1:
                        poly_obj = Polygon(poly)
                        if poly_obj.is_valid and len(poly_obj.exterior.coords) >= 4:
                            poly_obj = simplify_polygon(poly_obj, tolerance=0.01)
                            if poly_obj.is_valid:
                                polygons.append(poly_obj)
                            else:
                                logger.debug(f"简化后多边形无效，prop {prop.label}")
                        else:
                            logger.debug(f"多边形太小或无效：{len(poly_obj.exterior.coords)} 个坐标，prop {prop.label}")
                    else:
                        valid_indices = [idx for idx in juncs_index[0] if idx < len(poly)]
                        if not valid_indices:
                            logger.debug(f"多边形长度 {len(poly)} 无有效索引，prop {prop.label}")
                            continue
                        poly_obj = Polygon(poly[valid_indices])
                        if poly_obj.is_valid and len(poly_obj.exterior.coords) >= 4:
                            poly_obj = simplify_polygon(poly_obj, tolerance=0.01)
                            if poly_obj.is_valid:
                                polygons.append(poly_obj)
                            else:
                                logger.debug(f"简化后多边形无效，prop {prop.label}")
                        else:
                            logger.debug(f"无效或太小的多边形：{len(poly_obj.exterior.coords)} 个坐标，prop {prop.label}")
                except Exception as e:
                    logger.error(f"多边形处理失败，prop {prop.label}: {e}")

            logger.debug(f"生成的多边形数量 for {image_name}: {len(polygons)}")

            # 保存 Shapefile
            output_shapefile_path = osp.join(shp_output_dir, f"{image_name.split('.')[0]}_polygons.shp")
            try:
                save_shapefile_from_polygons(polygons, impath, output_shapefile_path, resolution=1.8788188)
                logger.info(f"Shapefile 保存成功 for {image_name} at {output_shapefile_path}")
            except Exception as e:
                logger.error(f"保存 Shapefile 失败 for {image_name}: {e}")

            # 可视化（可选）
            try:
                # viz_inria(image, polygons, self.output_dir, image_name)
                save_viz(image, polygons, self.output_dir, image_name)
            except Exception as e:
                logger.error(f"可视化失败 for {image_name}: {e}")

            # 保存分割结果
            try:
                im = Image.fromarray(((pred_whole_img > 0.5) * 255).astype(np.uint8), 'L')
                im.save(osp.join(output_dir, image_name))
            except Exception as e:
                logger.error(f"保存分割结果失败 for {image_name}: {e}")

            # 生成 COCO 注解
            try:
                img_id = int(osp.splitext(image_name)[0])  # 从文件名提取 ID
                image_result = generate_coco_ann([np.array(p.exterior.coords) for p in polygons], [1.0] * len(polygons), img_id)
                if image_result:
                    results.extend(image_result)
                image_masks = generate_coco_mask(pred_whole_img, img_id)
                if image_masks:
                    mask_results.extend(image_masks)
            except Exception as e:
                logger.error(f"生成 COCO 注解失败 for {image_name}: {e}")

        dt_file = osp.join(self.output_dir, f'{dataset_name}.json')
        logger.info(f'Writing the results of the {dataset_name} dataset into {dt_file}')
        try:
            with open(dt_file, 'w') as _out:
                json.dump(results, _out)
        except Exception as e:
            logger.error(f"写入 JSON 文件失败 {dt_file}: {e}")

        dt_file = osp.join(self.output_dir, f'{dataset_name}_mask.json')
        logger.info(f'Writing the mask results of the {dataset_name} dataset into {dt_file}')
        try:
            with open(dt_file, 'w') as _out:
                json.dump(mask_results, _out)
        except Exception as e:
            logger.error(f"写入掩码 JSON 文件失败 {dt_file}: {e}")

        self.gt_file = ''  # 根据需要设置 ground truth 文件路径
        self.dt_file = dt_file

if __name__ == "__main__":
    pass