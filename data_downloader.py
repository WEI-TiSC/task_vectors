import os
import shutil
from sklearn.model_selection import train_test_split

# 定义路径
data_dir = 'data/EuroSAT_splits'
train_dir = 'data/EuroSAT_splits/train'
test_dir = 'data/EuroSAT_splits/test'

# 定义训练集和测试集的比例
train_ratio = 0.8

# 获取所有类别
classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls)) and cls not in ['train', 'test']]
print(classes)

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images = os.listdir(cls_path)

    # 划分数据集
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

    train_cls_dir = os.path.join(train_dir, cls)
    test_cls_dir = os.path.join(test_dir, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    # 复制图像到训练集和测试集文件夹
    for img in train_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_cls_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_cls_dir, img))