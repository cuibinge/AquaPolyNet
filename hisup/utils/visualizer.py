
import os
import cv2
import numpy as np
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from shapely.ops import transform
from shapely.affinity import scale
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from geopandas import GeoSeries
from PIL import Image,ImageDraw
def show_polygons(image, polys):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.fill(polygon[:,0], polygon[:, 1], color=color, alpha=0.3)
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    plt.show()

#####################################last############################################
'''
def save_viz(image, polys, save_path, filename):
    plt.axis('off')
    plt.imshow(image)
    print(f"Number of polygons: {len(polys)}")

    for i, polygon_str in enumerate(polys):
        print(f"Polygon {i}: {polygon_str}")

        # 转换多边形字符串为 numpy 数组
        polygon = convert_polygon_str_to_array(polygon_str)
        cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
        for vertex in polygon:
            cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)
#         color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=0.01))
#         plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.', markersize=0.05)
        plt.plot(polygon[:,0], polygon[:,1], color='red', marker='.', linestyle='-', linewidth=0.1)
        plt.plot(polygon[:, 0], polygon[:, 1], color='red', marker='.', markersize=2, linestyle=' ', linewidth=0.1)


    impath = osp.join(save_path, 'viz', 'lyg_test4', filename)
    os.makedirs(os.path.dirname(impath), exist_ok=True)
    plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()
'''
def save_shapefile_from_polygons(polygons, image_path, output_shapefile_path, resolution=0.1, crs= None):
    """
    将多边形保存为带有地理信息的 Shapefile 文件。

    参数:
        polygons (list): 包含 Shapely Polygon 对象的列表。
        image_path (str): 输入 GeoTIFF 图像的路径，用于获取地理信息。
        output_shapefile_path (str): 输出 Shapefile 文件的路径。
        resolution (float): 图像的分辨率（默认为 0.1）。
        crs (str or CRS): 坐标系（默认为 None，从输入图像中获取）。
    """
    # 检查 polygons 是否为空
    if not polygons:
        print("Warning: No polygons to save. Skipping Shapefile creation.")
        return

    # 检查输入图像路径是否为文件
    if not os.path.isfile(image_path):
        raise ValueError(f"Input image path is not a file: {image_path}")

    # 打开输入图像以获取地理信息
    with rasterio.open(image_path) as src:
        left, bottom = src.bounds.left, src.bounds.bottom
        if crs is None:
            crs = src.crs  # 从输入图像中获取坐标系

    # 将多边形转换为地理坐标系
    polygons_geo = []
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        x_ = [i * resolution + left for i in x]
        y_ = [j * resolution + bottom for j in y]
        poly_geo = Polygon(zip(x_, y_))
        polygons_geo.append(poly_geo)

    # 创建 GeoDataFrame
    polygons_gpd = gpd.GeoSeries(polygons_geo)

    # 翻转多边形（如果需要）
    origin = ((left + src.bounds.right) / 2, (bottom + src.bounds.top) / 2)
    flip = GeoSeries.scale(polygons_gpd, xfact=1.0, yfact=-1.0, zfact=0, origin=origin)

    # 创建 shp 文件夹
    shp_dir = os.path.join(os.getcwd(), "shp_300")  # 当前目录下的 shp 文件夹
    os.makedirs(shp_dir, exist_ok=True)  # 如果文件夹不存在，则创建

    # 设置输出路径
    output_shapefile_path = os.path.join(shp_dir, os.path.basename(output_shapefile_path))

    # 保存为 Shapefile
    gdf = gpd.GeoDataFrame(crs=crs, geometry=flip)
    gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')

    print(f"Shapefile saved to {output_shapefile_path}")
def save_viz(image, polys, save_path, filename):
    # 将 NumPy 图像转换为 PIL 图像
    output_image = Image.fromarray(image)

    # 动态计算角点大小
    marker_size = max(1, int(0.0001 * min(image.shape[:2])))

    for polygon_str in polys:
        polygon = convert_polygon_str_to_array(polygon_str)
        if polygon.shape[0] == 0:
            continue

        # 将多边形坐标转换为整数
        polygon = [(int(x), int(y)) for x, y in polygon]

        # 绘制多边形
        draw = ImageDraw.Draw(output_image)
        draw.polygon(polygon, outline="red", width=1)

        # 绘制角点
        for x, y in polygon:
            draw.ellipse([x - marker_size, y - marker_size, x + marker_size, y + marker_size], fill="red")

    # 创建保存路径并保存图像
    impath = os.path.join(save_path, 'viz', 'lyg_test_u8_better', filename)
    os.makedirs(os.path.dirname(impath), exist_ok=True)
    output_image.save(impath, dpi=(1200, 1200))  # 设置高 dpi
# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)

#         # 确保 polygon 的类型为 int32 且形状正确
# #         polygon = polygon.astype(np.int32)
# #         polygon = polygon.reshape((-1, 1, 2))  # 转换为 (N, 1, 2) 格式

#         # 检查多边形是否为空
#         if polygon.shape[0] == 0:
#             print(f"Warning: Polygon {i} is empty, skipping.")
#             continue

# #         # 绘制多边形边界
# #         cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

# #         # 绘制多边形顶点
# #         for vertex in polygon:
# #             cv2.circle(image, tuple(vertex[0]), 1, (255, 0, 0), -1)

# #         # 可选：添加边界框
#         plt.gca().add_patch(Patches.Polygon(polygon.reshape(-1, 2), fill=False, ec='green', linewidth=0.3))
# # #         # 可选: 使用 matplotlib 绘制边界
# #         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=0.01))
# #         plt.plot(polygon[:, 0], polygon[:, 1], color='green', marker='.', markersize=2, linestyle='-', linewidth=0.1)
#         # plt.plot(polygon[:, 0], polygon[:, 1], color='red', marker='.', markersize=1)  # 设置更大的点，颜色为红色
#             # 绘制所有角点
#         for j in range(polygon.shape[0]):
#             plt.plot(polygon[j, 0], polygon[j, 1], color='red', marker='o', markersize=2)  # 使用浮点数坐标

#     impath = osp.join(save_path, 'viz', 'lyg_test8', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()





def convert_polygon_str_to_array(polygon_str):
    """
    将 polygon 字符串、Polygon 对象或多边形数组转换为 numpy 数组。

    参数:
    - polygon_str: Polygon 对象或多边形坐标的字符串或数组

    返回:
    - numpy 数组，形状为 (N, 2)，每行是一个顶点的 (x, y) 坐标
    """
    # 如果传入的是 Polygon 对象
    if isinstance(polygon_str, Polygon):
        # 获取多边形的外部坐标并转换为 numpy 数组
        coords = np.array(polygon_str.exterior.coords, dtype=np.float32)

    # 如果传入的是字符串，假设它是一个表示多边形的坐标列表
    elif isinstance(polygon_str, str):
        # 尝试将字符串解析为坐标列表
        try:
            # 假设字符串是一个 JSON 格式或 CSV 格式的坐标数据
            coords = np.array(eval(polygon_str), dtype=np.float32)
        except Exception as e:
            print(f"Error parsing polygon string: {e}")
            return np.array([])  # 返回空数组作为错误处理

    # 如果传入的是普通的坐标数组
    elif isinstance(polygon_str, (list, np.ndarray)):
        coords = np.array(polygon_str, dtype=np.float32)
        # 确保数组是二维的，形状为 (N, 2)
        if coords.ndim == 1:
            coords = coords.reshape((-1, 2))

    else:
        print("Error: Invalid input type for polygon_str.")
        return np.array([])  # 返回空数组作为错误处理

    # 确保返回的数组是二维的，形状为 (N, 2)
    if coords.ndim == 1:
        coords = coords.reshape((-1, 2))

    return coords

def viz_inria(image, polygons, output_dir, file_name, alpha=0.5, linewidth=12, markersize=45):
    plt.rcParams['figure.figsize'] = (500,500)
    plt.rcParams['figure.dpi'] = 10
    plt.axis('off')
    plt.imshow(image)
    for n, poly in enumerate(polygons):
        poly_color = colormap[n%num_color]
        if poly.type == 'MultiPolygon':
            for p in poly:
                patch = PolygonPatch(p.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
                plt.gca().add_patch(Patches.Polygon(p.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                juncs = np.array(p.exterior.coords[:-1])
                plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
                if len(p.interiors) != 0:
                    for inter in p.interiors:
                        plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                        juncs = np.array(inter.coords[:-1])
                        plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
        else:
            try:
                patch = PolygonPatch(poly.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
            except TypeError:
                plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=True, ec=poly_color, fc=poly_color, linewidth=linewidth, alpha=alpha))
            plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
            juncs = np.array(poly.exterior.coords[:-1])
            plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
            if len(poly.interiors) != 0:
                for inter in poly.interiors:
                    plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                    juncs = np.array(inter.coords[:-1])
                    plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
    
    # save_filename = os.path.join(output_dir, 'inria_viz', file_name[:-4] + '.svg')
    # plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
    # plt.clf()
    plt.show()


def draw_predictions_with_mask_inria(img, junctions, polys_ids, save_dir, filename):
    plt.axis('off')
    plt.imshow(img)

    instance_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for h, idx in enumerate(contours):
            poly = junctions[idx]
            if h == 0:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=color, thickness=-1)
            else:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=0, thickness=-1)

            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=2))
    
    alpha_map = np.bitwise_or(instance_mask[:,:,0:1].astype(bool), 
                              instance_mask[:,:,1:2].astype(bool), 
                              instance_mask[:,:,2:3].astype(bool)).astype(np.float32)
    instance_mask = np.concatenate((instance_mask, alpha_map), axis=-1)
    plt.imshow(instance_mask, alpha=0.3)
    plt.show()


def draw_predictions_inria(img, junctions, polys_ids):
    plt.axis('off')

    plt.imshow(img)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for idx in contours:
            poly = junctions[idx]
            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=1.5))
            plt.plot(poly[:,0], poly[:,1], color=color, marker='.')
    plt.show()








