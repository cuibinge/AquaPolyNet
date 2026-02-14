#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os  # 确保 os 模块导入
import numpy as np
import torch
import argparse
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.dataset.build import build_transform
from hisup.utils.comm import to_single_device
import scipy.ndimage

# 调试：打印 os 模块路径
print(f"os module path: {os.__file__}")

def inference_single(cfg, model, image, device, output_dir, image_name, stride):
    """
    对单张图像进行推理，保存并可视化中间结果。
    
    参数：
    - cfg: 配置文件对象
    - model: 训练好的模型
    - image: 输入图像 (H, W, C)
    - device: 设备 (CPU/GPU)
    - output_dir: 输出目录
    - image_name: 图像名称（不含扩展名）
    - stride: 滑窗步幅
    """
    # 显式导入 os 模块，确保在函数作用域内可用
    try:
        import os
        print(f"Inside inference_single: os module path = {os.__file__}")
    except ImportError as e:
        print(f"Failed to import os module: {str(e)}")
        raise

    # 创建子目录
    try:
        afm_npy_dir = os.path.join(output_dir, "afm_npy")
        afm_viz_dir = os.path.join(output_dir, "afm_viz")
        juncs_npy_dir = os.path.join(output_dir, "juncs_npy")
        juncs_viz_dir = os.path.join(output_dir, "juncs_viz")
        mask_npy_dir = os.path.join(output_dir, "mask_npy")
        mask_viz_dir = os.path.join(output_dir, "mask_viz")
        
        os.makedirs(afm_npy_dir, exist_ok=True)
        os.makedirs(afm_viz_dir, exist_ok=True)
        os.makedirs(juncs_npy_dir, exist_ok=True)
        os.makedirs(juncs_viz_dir, exist_ok=True)
        os.makedirs(mask_npy_dir, exist_ok=True)
        os.makedirs(mask_viz_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories for {image_name}: {str(e)}")
        raise

    transform = build_transform(cfg)
    
    # 推理参数
    try:
        h_stride, w_stride = stride, stride  # 使用传入的 stride 参数
        h_crop, w_crop = cfg.DATASETS.ORIGIN.HEIGHT, cfg.DATASETS.ORIGIN.WIDTH
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
        
        patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
        patch_weight = patch_weight[1:-1, 1:-1]
    except Exception as e:
        print(f"Failed to initialize inference parameters for {image_name}: {str(e)}")
        raise

    # 推理过程
    try:
        for h_idx in tqdm(range(h_grids), desc=f'Processing image {image_name}'):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                crop_img = image[y1:y2, x1:x2, :]
                crop_img_tensor = transform(crop_img.astype(float))[None].to(device)
                
                meta = {
                    'height': crop_img.shape[0],
                    'width': crop_img.shape[1],
                    'pos': [x1, y1, x2, y2]
                }

                with torch.no_grad():
                    output, _ = model(crop_img_tensor, [meta])
                    output = to_single_device(output, 'cpu')

                juncs_pred = output['juncs_pred'][0]
                mask_pred = output['mask_pred'][0]
                juncs_pred += [x1, y1]
                juncs_whole_img.extend(juncs_pred.tolist())
                mask_pred *= patch_weight
                pred_whole_img += np.pad(mask_pred,
                                         ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                          (int(x1), int(pred_whole_img.shape[1] - x2))))
                count_mat[y1:y2, x1:x2] += patch_weight
    except Exception as e:
        print(f"Failed during inference for {image_name}: {str(e)}")
        raise

    juncs_whole_img = np.array(juncs_whole_img)
    pred_whole_img = pred_whole_img / count_mat

    # 提取 afm_pred
    try:
        crop_img_tensor = transform(image.astype(float))[None].to(device)
        meta = [{'height': image.shape[0], 'width': image.shape[1]}]
        with torch.no_grad():
            output, _ = model(crop_img_tensor, meta)
            output = to_single_device(output, 'cpu')
            print(f"Output keys for {image_name}: {list(output.keys())}")  # 调试输出
            afm_pred = output.get('afm_pred', None)  # 假设 afm_pred 在 output 中
    except Exception as e:
        print(f"Failed to extract afm_pred for {image_name}: {str(e)}")
        raise

    # 保存和可视化 afm_pred
    if afm_pred is not None:
        try:
            afm_np = afm_pred[0].detach().cpu().numpy()  # [C, H, W]
            # 保存 afm_pred
            afm_npy_path = os.path.join(afm_npy_dir, f"{image_name}_afm.npy")
            np.save(afm_npy_path, afm_np)

            # 可视化 afm_pred 的 dx 和 dy
            if afm_np.shape[0] >= 2:
                dx = afm_np[0]
                dy = afm_np[1]
                dx = (dx - dx.min()) / (dx.max() - dx.min() + 1e-8)  # 归一化
                dy = (dy - dy.min()) / (dy.max() - dy.min() + 1e-8)

                # 可视化 dx
                plt.figure(figsize=(8, 8))
                plt.imshow(dx, cmap='viridis')
                plt.axis('off')
                plt.gca().set_frame_on(False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(os.path.join(afm_viz_dir, f"{image_name}_dx.png"), bbox_inches='tight', pad_inches=0)
                plt.close()

                # 可视化 dy
                plt.figure(figsize=(8, 8))
                plt.imshow(dy, cmap='viridis')
                plt.axis('off')
                plt.gca().set_frame_on(False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(os.path.join(afm_viz_dir, f"{image_name}_dy.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                print(f"Warning: afm_pred for {image_name} does not have enough channels for dx and dy visualization.")
        except Exception as e:
            print(f"Failed to save or visualize afm_pred for {image_name}: {str(e)}")
            raise
    else:
        print(f"Warning: afm_pred not found in output for {image_name}.")

    # 保存并可视化 juncs_pred
    try:
        juncs_npy_path = os.path.join(juncs_npy_dir, f"{image_name}_juncs.npy")
        np.save(juncs_npy_path, juncs_whole_img)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        if len(juncs_whole_img) > 0:
            plt.scatter(juncs_whole_img[:, 0], juncs_whole_img[:, 1], c='red', s=10, marker='o')
        plt.axis('off')
        plt.gca().set_frame_on(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(juncs_viz_dir, f"{image_name}_juncs.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Failed to save or visualize juncs_pred for {image_name}: {str(e)}")
        raise

    # 保存并可视化 mask_pred
    try:
        mask_npy_path = os.path.join(mask_npy_dir, f"{image_name}_mask.npy")
        np.save(mask_npy_path, pred_whole_img)

        plt.figure(figsize=(8, 8))
        plt.imshow(pred_whole_img, cmap='gray')
        plt.axis('off')
        plt.gca().set_frame_on(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(mask_viz_dir, f"{image_name}_mask.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Failed to save or visualize mask_pred for {image_name}: {str(e)}")
        raise

    return juncs_whole_img, pred_whole_img

def get_pretrained_model_FT(cfg, dataset, device, path_model, pretrained=True):
    """
    加载预训练模型。
    """
    try:
        model = BuildingDetector(cfg, test=True)
        state_dict = torch.load(path_model)
        model.load_state_dict(state_dict["model"])
        model = model.eval()
        return model
    except Exception as e:
        print(f"Failed to load pretrained model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Visualize and save inference results for a dataset")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, help="Output directory for saving results")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for processing")
    parser.add_argument("--stride", type=int, default=400, help="Stride for sliding window")
    args = parser.parse_args()

    # 验证路径
    try:
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    except Exception as e:
        print(f"Path validation failed: {str(e)}")
        raise

    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory {args.output_dir}: {str(e)}")
        raise

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    try:
        dataset = 'inria'  # 假设数据集为 inria
        model = get_pretrained_model_FT(cfg, dataset, device, args.checkpoint, pretrained=True)
        model = model.to(device)
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        raise

    # 设置推理参数
    try:
        H, W = args.patch_size, args.patch_size
        cfg.DATASETS.ORIGIN.HEIGHT = args.patch_size if H > args.patch_size else H
        cfg.DATASETS.ORIGIN.WIDTH = args.patch_size if W > args.patch_size else W
    except Exception as e:
        print(f"Failed to set inference parameters: {str(e)}")
        raise

    # 遍历输入目录中的图像
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    try:
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
    except Exception as e:
        print(f"Failed to list image files in {args.input_dir}: {str(e)}")
        raise

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input_dir, image_file)
        image_name = os.path.splitext(image_file)[0]  # 去掉扩展名

        # 读取图像
        try:
            image = io.imread(image_path)[:, :, :3]
        except Exception as e:
            print(f"Failed to read image {image_path}: {str(e)}")
            continue

        # 进行推理并保存结果
        try:
            juncs_whole_img, pred_whole_img = inference_single(
                cfg, model, image, device, args.output_dir, image_name, args.stride
            )
            print(f"Processed {image_file} successfully.")
        except Exception as e:
            print(f"Failed to process image {image_file}: {str(e)}")
            continue

    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()