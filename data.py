import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.pipeline import Pipeline
import numpy as np
from utils import StandardScaler, Gaussianize


class DataProcessor:
    def __init__(self, data_colname='Adj Close'):
        self.data_colname = data_colname

    def preprocess(self, data_path):
        df = pd.read_csv(data_path)#.tail(3000)
        df = df[self.data_colname]
        self.log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)

        self.pipeline = Pipeline([
            ('scl1', StandardScaler()),
            ('scl2', StandardScaler()),
            ('gau', Gaussianize()),
        ])
        
        log_returns_preprocessed = self.pipeline.fit_transform(self.log_returns)

        return log_returns_preprocessed

    def postprocess(self, y):
        standardScaler1, standardScaler2, gaussianize = self.pipeline[0], self.pipeline[1], self.pipeline[2]
        y = y[:, :, 0]
        y = (y - y.mean(axis=0))/y.std(axis=0)
        y = standardScaler2.inverse_transform(y)
        y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
        y = standardScaler1.inverse_transform(y)

        # some basic filtering to reduce the tendency of GAN to produce extreme returns
        y = y[(y.max(axis=1) <= 2 * self.log_returns.max()) & (y.min(axis=1) >= 2 * self.log_returns.min())]
        y -= y.mean()

        return y


class StockDataset(Dataset):
    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)
        
    def __len__(self):
        return max(len(self.data)-self.length, 0)

