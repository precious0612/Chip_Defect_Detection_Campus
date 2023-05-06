from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
import math

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """

    def __init__(self, df, list_IDs,
                 to_fit=True, batch_size=8,
                 n_channels=3, n_classes=10, shuffle=True):
        """ 初始化方法
        :param df: 存放数据路径和标签的数据框
        :param list_IDs: 数据索引列表
        :param to_fit: 设定是否返回标签y
        :param batch_size: batch size
        :param dim: 图像大小
        :param n_channels: 图像通道
        :param n_classes: 标签类别
        :param shuffle: 每一个epoch后是否打乱数据
        """
        self.x = scaler.fit_transform(df.reshape(-1, 1)).reshape((-1,)+df.shape[-2:])
        self.y = list_IDs
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
          return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([ file for file in batch_x]), np.array(batch_y)
    
    def on_epoch_end(self):
        """每个epoch之后更新索引
        """
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
