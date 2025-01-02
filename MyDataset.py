import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, train: bool, transform=None):
        self.train = train
        if self.train:
            with open('./data/train_datasets.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('./data/val_datasets.pkl', 'rb') as f:
                self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data['ir'][index], self.data['vi'][index]

    def __len__(self):
        assert len(self.data['ir']) == len(self.data['vi'])
        return len(self.data['ir'])

if __name__ == '__main__':
    # dataset = MyDataset(train=True)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # print(dataset.__len__())
    # for index, (ir_img, vi_img) in enumerate(dataloader):
    #     # test mydataset
    #     # Remove all channels of length 1
    #     ir_image_to_display = ir_img.squeeze()
    #     # Gray scale mode display
    #     plt.imshow(ir_image_to_display, cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    #
    #     vi_image_to_display = vi_img.squeeze()
    #     plt.imshow(vi_image_to_display, cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    import pickle

    # 加载 train_datasets.pkl 文件
    with open('./data/train_datasets.pkl', 'rb') as f:
        data = pickle.load(f)

    # 获取可见光和红外图像数量
    num_ir_images = len(data['ir'])
    num_vi_images = len(data['vi'])

    print("红外图像数量:", num_ir_images)
    print("可见光图像数量:", num_vi_images)
import os


def count_images(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

    return image_count


# 替换成你的文件夹路径
folder_path = '/home/dmh/My_code/TBMDFuse/test_images/MSRS/ir'

