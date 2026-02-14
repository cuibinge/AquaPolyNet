import os
import shutil
from sklearn.model_selection import train_test_split
import random

# train_img_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_2048_have8/img"
# lab_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_2048_have8/gt"
# 假设你的数据和标签存储在 'data' 文件夹中
# data_dir = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_lyg_1024_norm"
# data_img = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_lyg_1024_norm/img"
#     # train_img_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4"
#     # lab_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4/gt"
# train_dir = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_lyg_1024_norm/train"
# test_dir =r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_lyg_1024_norm/test"
# val_dir= r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_lyg_1024_norm/val"
# data_dir = r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_new/"
# data_img = r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_new/img"
#     # train_img_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4"
#     # lab_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4/gt"
# train_dir = r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_new/train"
# test_dir =r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_new/test"
# val_dir= r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_new/val"
data_dir = r"./data/sda/cut_512/"
# shared-nvme/HiSup-main/data/sda/cut_512/img
    # train_img_path = r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/20231021/cut_512_gf2_streth256/img"
    # lab_path = r"/work/home/acrjggsb3t/shuguang/GeoSeg/data/yangzhiqu/cut_2048_gf2_streth256/gt"
data_img = r"./data/sda/cut_512/img"
    # train_img_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4"
    # lab_path = r"/work/home/acrjggsb3t/shuguang/unet/data/lyg/1/cut_1024_n4/gt"
train_dir = r"./data/sda/cut_512/train"
# test_dir =r"./data/sda/cut_512/test"
val_dir= r"./data/sda/cut_512/val"
# 创建子文件夹
os.makedirs(os.path.join(train_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'gt'), exist_ok=True)
# os.makedirs(os.path.join(test_dir, 'img'), exist_ok=True)
# os.makedirs(os.path.join(test_dir, 'gt'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'gt'), exist_ok=True)

# 获取所有数据文件
data_files = [f for f in os.listdir(data_img) if os.path.isfile(os.path.join(data_img, f))]

# 随机打乱数据顺序
# random.shuffle(data_files)

# # 分割数据集
# train_files, midle_files = train_test_split(data_files, test_size=0.2, random_state=16)
# val_files, test_files = train_test_split(midle_files, test_size=0.5, random_state=16)
train_files, val_files = train_test_split(
    data_files, test_size=0.1, random_state=16
)
# 第一次分割，将数据集分割成训练集和临时测试集（包含测试集和验证集）
# train_files, midle_files = train_test_split(
#     data_files, test_size=0.2, random_state=16
# )

# 第二次分割，将临时测试集分割成测试集和验证集
# 这里我们将临时测试集的50%作为测试集，另外50%作为验证集
# val_files, test_files = train_test_split(
#     midle_files, test_size=0.5, random_state=16
# )
# 复制文件到训练集和测试集文件夹
for file in train_files:
    shutil.copy(os.path.join(data_dir, 'img', file), os.path.join(train_dir, 'img', file))
    shutil.copy(os.path.join(data_dir, 'gt', file), os.path.join(train_dir, 'gt', file))

# for file in test_files:
#     shutil.copy(os.path.join(data_dir, 'img', file), os.path.join(test_dir, 'img', file))
#     shutil.copy(os.path.join(data_dir, 'gt', file), os.path.join(test_dir, 'gt', file))
    
for file in val_files:
    shutil.copy(os.path.join(data_dir, 'img', file), os.path.join(val_dir, 'img', file))
    shutil.copy(os.path.join(data_dir, 'gt', file), os.path.join(val_dir, 'gt', file))


print(f"训练集文件已复制到 {train_dir}，数量：{len(train_files)}")
# print(f"测试集文件已复制到 {test_dir}，数量：{len(test_files)}")
print(f"验证集文件已复制到 {val_dir}，数量：{len(val_files)}")