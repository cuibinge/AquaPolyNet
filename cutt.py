# import imageio
import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile
import imageio.v2 as imageio
# import imageio
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 根目录为当前文件所在目录
root = os.getcwd()


# def not_mostly_black(block, threshold_ratio=0.02):
#     # 计算图像块中值为1的像素的数量
#     value_one_pixels = np.sum(block == 1)
#     # 计算图像块的总像素数量
#     total_pixels = block.size
#     # 计算值为1的像素的比例
#     value_one_ratio = value_one_pixels / total_pixels
#     # 如果值为1的像素的比例大于阈值，则认为图像块中值为1的像素数大于30%
#     return value_one_ratio > threshold_ratio
def not_mostly_black(block, threshold_ratio=0.05):
    # 计算图像块中值为1或2的像素的数量
    value_one_or_two_pixels = np.sum((block == 1) | (block == 2)|(block == 3) | (block == 4))
    # 计算图像块的总像素数量
    total_pixels = block.size
    # 计算值为1或2的像素的比例
    value_one_or_two_ratio = value_one_or_two_pixels / total_pixels
    # 如果值为1或2的像素的比例大于阈值，则认为图像块不是大部分为黑色
    return value_one_or_two_ratio > threshold_ratio
# ====================================================================================================================================

def cut_image(image, label_raw, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]
    dep = image.shape[2]
    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='uint8')
    padding_img[:row, :col, :] = image[:row, :col, :]
    num = 0
    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            i_ = row - block_size
        else:
            i_ = i

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1
            num += 1

            if (j + block_size) > col:
                j_ = col - block_size
            else:
                j_ = j

            block = np.array(padding_img[i_: i_ + block_size, j_: j_ + block_size, :])
            # block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.png'
            block_name = str(num) + '.png'
            block[np.isnan(block)] = 0
#             if not np.all(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] == 0, axis=2).any():
            # if np.any(label_raw[i_: i_ + block_size, j_: j_ + block_size] != 0):
            if not_mostly_black(label_raw[i_: i_ + block_size, j_: j_ + block_size]):
                imageio.imwrite(os.path.join(save_path, block_name), block)


def cut_label(image, label_raw, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]

    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')
    padding_pre[:row, :col] = image[:row, :col]
    num = 0
    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            i_ = row - block_size
        else:
            i_ = i

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1
            num += 1
            if (j + block_size) > col:
                j_ = col - block_size
            else:
                j_ = j

            block = np.array(padding_pre[i_: i_ + block_size, j_: j_ + block_size])
            #            if np.min(block) == 0:
            #                print('ok')
            # block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.png'
            block_name = str(num) + '.png'

            # block[block == 2] = 1
            block[np.isin(block, [2, 3, 4])] = 1
            block[block != 1] = 0
        #             if not np.all(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] == 0, axis=2).any():
            # if np.any(label_raw[i_: i_ + block_size, j_: j_ + block_size] != 0):
            if not_mostly_black(label_raw[i_: i_ + block_size, j_: j_ + block_size]):
                imageio.imwrite(os.path.join(save_path, block_name), block)


def truncate_stretch(image, min_percentile=2, max_percentile=98):
    stretched_channels = []
    # 排除所有通道全为0的像素
    image_n0 = image[np.any(image != 0, axis=-1)]
    for channel in range(image.shape[2]):
        image_channel = image[:, :, channel]

        # 计算每个通道的百分位数
        min_val = np.percentile(image_n0[:, channel], min_percentile)
        max_val = np.percentile(image_n0[:, channel], max_percentile)
        print("min: ", min_val)
        print("max: ", max_val)
        # 将像素值限制在截断范围内
        clipped_channel = np.clip(image_channel, min_val, max_val)

        # 缩放到0-255范围内
        stretched_channel = ((clipped_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        stretched_channels.append(stretched_channel)

    # 将各个通道重新堆叠成图像
    stretched_image = np.stack(stretched_channels, axis=2)

    return stretched_image


# ====================================================================================================================================

"""
生成彩色图像，为了标记样本
"""

# image 文件路径
image_path = r"./data/sda/img"
# image_path = r"./data/lyg/img"

image_path_list_raw = glob.glob(os.path.join(image_path, "*.tif")) + glob.glob(
    os.path.join(image_path, "*.png")) + glob.glob(os.path.join(image_path, "*.tiff"))

# label 文件路径
label_path = r"./data/sda/gt"
label_path_list_raw = glob.glob(os.path.join(label_path, "*.tif")) + glob.glob(
    os.path.join(label_path, "*.png")) + glob.glob(os.path.join(label_path, "*.tiff"))

image_path_list = sorted(image_path_list_raw)
label_path_list = sorted(label_path_list_raw)

# mean_all = np.array([0, 0, 0, 0])
# std_all = np.array([0, 0, 0, 0])

# arr = np.array([0, 0, 0, 0, 0, 0])
# arr = np.zeros(25, dtype=np.int64)
for i in range(len(image_path_list)):
    # if i < 303:
    #     continue
    # 整幅图像的数据和标签
    image_data_raw = imageio.imread(image_path_list[i])
    label_data_raw = imageio.imread(label_path_list[i])

    image_data = imageio.imread(image_path_list[i])
    label_data = imageio.imread(label_path_list[i])

    img8_path = r"./data/sda/val8"
    if not os.path.exists(img8_path):
        os.makedirs(img8_path)

    file_name = os.path.basename(image_path_list[i])
    file_name = os.path.splitext(file_name)[0]
    print(file_name)
    # 如果图像数据的形状为 (channels, height, width)，则转置为 (height, width, channels)
    if image_data.shape[0] == 4:  # 这里假设 channels 是第一个维度
        image_data = np.transpose(image_data, (1, 2, 0))

    image_data = image_data[:, :, [2, 1, 0]]
    print(image_data.shape)
    image_data = truncate_stretch(image_data)
    # imageio.imwrite(os.path.join(img8_path, file_name + '.png'), image_data)

   
    print("-------", i, "-------")
 

    train_img_path = r"./data/sda/cut_512/img"
#     train_img_path = os.path.join(train_img_path, file_name)
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)

    lab_path = r"./data/sda/cut_512/gt"
    if not os.path.exists(lab_path):
        os.makedirs(lab_path)

    '''
    裁图
    block_size为结果图大小
    stride为裁图步长
    '''
    
    cut_image(image_data, label_data_raw, str(i), block_size=512, stride=512,
              save_path=train_img_path)
    cut_label(label_data, label_data_raw, str(i), block_size=512, stride=512,
              save_path=lab_path)

