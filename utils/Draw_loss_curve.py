import os
import matplotlib
matplotlib.use('Agg')  # 或者 'TkAgg'，根据你的需要选择
from matplotlib import pyplot as plt

# 替换成你的文件夹路径
save_path = ""
if not os.path.exists(save_path):
    os.makedirs(save_path)
def Draw_loss_curve(Epochs, Mean_Loss,run_time):
    epochs = list(range(1, Epochs+1))
    Mean_Loss = Mean_Loss
    plt.plot(epochs, Mean_Loss, marker='o', linestyle='-', color='blue', label='Mean Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{run_time}loss_curve.png'))
    plt.show()
    plt.close()
if __name__ == '__main__':
    Mean_Loss = [2.7728, 1.9127, 1.6158, 1.6406, 1.6229, 1.6692, 1.6087, 1.5764, 1.4699, 1.8056]
    Draw_loss_curve(10, Mean_Loss=Mean_Loss,run_time=1)
