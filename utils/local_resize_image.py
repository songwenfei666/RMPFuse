import os

import cv2
import numpy as np
from PIL import Image
result_path = '../all_data/result/Ablation/RoadScene04943/'
dataset_path = '../all_data/Dataset/Ablation/RoadScene04943/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

# TNO1
pt1=(248, 137)
pt2=(446, 183)
width1=52
hight1=32
width2=52
hight2=32
def show_cvimg(im):
    return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

def stack_image(path:str):
    # 在指定的位置绘制正方形并在右下角放大该局部位置
    # pt:正方形位置,width : 正方形宽度 scale: 放大倍数
    image = cv2.imread(path)
    h, w, c = image.shape

    if w % 2 != 0:
        if h % 2 != 0:  # 宽高都是奇数
            image = cv2.resize(image, (w - 1, h - 1))
        else:  # 宽是奇数，高是偶数
            image = cv2.resize(image, (w - 1, h))
    elif h % 2 != 0:  # 高是奇数，宽是偶数
        image = cv2.resize(image, (w, h - 1))

    # 定义缩放比
    scale = h / w
    # patch1
    pt1_ = (pt1[0] + width1, pt1[1] + hight1)
    cv2.rectangle(image, pt1, pt1_, (0, 0, 255), 2)
    # patch2
    pt2_ = (pt2[0] + width2, pt2[1] + hight2)
    cv2.rectangle(image, pt2, pt2_, (0, 0, 255), 2)
    # 要放大的部分
    patch1_ = image[pt1[1] + 2:pt1[1] + hight1 - 2, pt1[0] + 2:pt1[0] + width1 -2, :]
    t1 = patch1_.copy()
    cv2.rectangle(t1, (0, 0), (t1.shape[1]-1, t1.shape[0]-1), (0, 0, 255), 1)
    t1 = cv2.resize(t1, (int(w / 2), int(h / 2)))


    patch2_ = image[pt2[1] + 2:pt2[1] + hight2 - 2, pt2[0] +2:pt2[0] + width2 - 2, :]
    t2 = patch2_.copy()
    cv2.rectangle(t2, (0, 0), (t2.shape[1] - 1, t2.shape[0] - 1), (0, 0, 255), 1)
    t2 = cv2.resize(t2, (int(w / 2), int(h / 2)))

    patch = np.hstack((t1, t2))
    image_stack = np.vstack((image, patch))
    return image_stack


if __name__ == '__main__':

    import os
    from PIL import Image

    # 设置文件夹路径
    folder_path = dataset_path

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件路径和文件扩展名
        file_path = os.path.join(folder_path, filename)
        name, ext = os.path.splitext(filename)

        # 检查文件是否为支持的图像格式
        if ext.lower() in [".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
            # 打开图像并转换为PNG格式
            with Image.open(file_path) as img:
                # 保存图像为PNG格式，使用原文件名但扩展名为 .png
                png_path = os.path.join(folder_path, f"{name}.png")
                img.save(png_path, "PNG")

            # 删除原始图像文件
            os.remove(file_path)

    temp_list = []
    for root, dict, files in os.walk(dataset_path):
        for file in files:
            temp_list.append(os.path.join(root, file))
    for i in range(len(temp_list)):
        cv2.imwrite(os.path.join(result_path, os.path.basename(temp_list[i])), stack_image(temp_list[i]), [cv2.IMWRITE_PNG_COMPRESSION, 0])


# # 保存图片
# image_stack = stack_image()
# cv2.imwrite('image_stack.png', image_stack, [cv2.IMWRITE_PNG_COMPRESSION, 0])

