import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_dx_dy(afm_pred_data, output_dir="viz", image_name="image"):
    """
    从 afm_pred 数据生成并保存 dx 和 dy 的可视化图像。

    参数：
    - afm_pred_data: 可以是 afm_pred 张量（[C, H, W]）或保存 afm_pred 的 .npy 文件路径
    - output_dir: 保存可视化图像的目录
    - image_name: 图像名称，用于生成文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 如果 afm_pred_data 是文件路径，则加载数据
    if isinstance(afm_pred_data, str):
        afm_np = np.load(afm_pred_data)  # 加载 .npy 文件
    else:
        # 假设 afm_pred_data 是一个张量，转换为 numpy
        afm_np = afm_pred_data.detach().cpu().numpy()  # [C, H, W]

    # 确保 afm_np 至少有 2 个通道（dx 和 dy）
    if afm_np.shape[0] < 2:
        raise ValueError("afm_pred must have at least 2 channels for dx and dy visualization.")

    # 提取 dx 和 dy
    dx = afm_np[0]  # 提取 dx 分量，形状 [H, W]
    dy = afm_np[1]  # 提取 dy 分量，形状 [H, W]

    # 增强对比度：对 dx 和 dy 进行归一化
    dx = (dx - dx.min()) / (dx.max() - dx.min() + 1e-8)  # 归一化到 [0, 1]
    dy = (dy - dy.min()) / (dy.max() - dy.min() + 1e-8)  # 归一化到 [0, 1]

    # 处理文件名，确保安全
    safe_name = "".join(c for c in image_name if c.isalnum() or c in ('.', '_', '-'))

    # 可视化 dx
    plt.figure(figsize=(8, 8))
    plt.imshow(dx, cmap='coolwarm')  # 使用 coolwarm 色图
    plt.axis('off')  # 关闭坐标轴和刻度
    plt.gca().set_frame_on(False)  # 移除边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除所有边距
    plt.savefig(os.path.join(output_dir, f"{safe_name}_dx.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # 可视化 dy
    plt.figure(figsize=(8, 8))
    plt.imshow(dy, cmap='coolwarm')  # 使用 coolwarm 色图
    plt.axis('off')  # 关闭坐标轴和刻度
    plt.gca().set_frame_on(False)  # 移除边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除所有边距
    plt.savefig(os.path.join(output_dir, f"{safe_name}_dy.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def process_npy_folder(input_dir, output_dir="viz"):
    """
    遍历输入文件夹中的所有 .npy 文件，生成 dx 和 dy 的可视化图像，并保存到输出文件夹。

    参数：
    - input_dir: 包含 .npy 文件的输入文件夹路径
    - output_dir: 保存可视化图像的输出文件夹路径
    """
    # 确保输入文件夹存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹中的所有 .npy 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            # 提取 image_name，去掉后缀 "_afm.npy"
            image_name = filename.replace("_afm.npy", "")
            npy_path = os.path.join(input_dir, filename)
            print(f"Processing {npy_path}...")
            try:
                visualize_dx_dy(npy_path, output_dir=output_dir, image_name=image_name)
                print(f"Generated dx and dy visualizations for {image_name}")
            except Exception as e:
                print(f"Error processing {npy_path}: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_directory = "./afm_visualizations"  # 包含 .npy 文件的文件夹
    output_directory = "./afm_visualizations/viz"  # 保存 dx.png 和 dy.png 的文件夹
    process_npy_folder(input_directory, output_directory)

'''
python tools/visualize_inference.py \
  --input_dir data/lyg/cut_300/val/img \
  --output_dir vis_afm/test_300 \
  --config outputs/lyg_hrnet48_u8/config.yml \
  --checkpoint outputs/lyg_hrnet48_u8/model_00050.pth \
  --patch_size 512 \
  --stride 400
'''