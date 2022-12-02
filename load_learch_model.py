import os

import torch
from torch import nn

from model import PolicyFeedforward as PolicyFeed
import numpy as np

# 将learch的模型转化为libtorch支持加载的格式
class PolicyFeedforwardNp:
    def __init__(self, input_dim, hidden_dim, state_dict: dict):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mean = state_dict["scaler"].mean_.reshape(1, -1) # (1, 47)
        self.scale = state_dict["scaler"].scale_.reshape(1, -1) # (1, 47)
        self.linear1 = state_dict["state_dict"]["net.0.weight"].numpy().T # (47, 64)
        self.bias1 = state_dict["state_dict"]["net.0.bias"].numpy().reshape(1, -1) # (1, 64)
        self.linear2 = state_dict["state_dict"]["net.2.weight"].numpy().T # (64, 64)
        self.bias2 = state_dict["state_dict"]["net.2.bias"].numpy().reshape(1, -1) # (1, 64)
        self.linear3 = state_dict["state_dict"]["net.4.weight"].numpy().T # (64, 1)
        self.bias3 = state_dict["state_dict"]["net.4.bias"].numpy() # (1,)


    def standardize(self, x: np.ndarray):
        return np.divide(x - self.mean, self.scale)

    def relu(self, x: np.ndarray):
        return np.where(x >= 0, x, 0)

    def forward(self, x: np.ndarray): # size = (1, 47)
        out = self.standardize(x)
        res = np.matmul(out, self.linear1)
        out = self.relu(np.add(np.matmul(out, self.linear1), self.bias1))
        out = self.relu(np.add(np.matmul(out, self.linear2), self.bias2))
        out = np.add(np.matmul(out, self.linear3), self.bias3)
        return out[0, 0]

    def save(self, path: str):
        self.mean.tofile(os.path.join(path, "mean"), sep="\n")
        self.scale.tofile(os.path.join(path, "scale"), sep="\n")
        self.linear1.tofile(os.path.join(path, "linear1"), sep="\n")
        self.bias1.tofile(os.path.join(path, "bias1"), sep="\n")
        self.linear2.tofile(os.path.join(path, "linear2"), sep="\n")
        self.bias2.tofile(os.path.join(path, "bias2"), sep="\n")
        self.linear3.tofile(os.path.join(path, "linear3"), sep="\n")
        self.bias3.tofile(os.path.join(path, "bias3"), sep="\n")
        # np.savez(os.path.join(path, "model"),
        #          mean = np.asfortranarray(self.mean), scale = np.asfortranarray(self.scale),
        #          linear1 = np.asfortranarray(self.linear1), bias1 = np.asfortranarray(self.bias1),
        #          linear2 = np.asfortranarray(self.linear2), bias2 = np.asfortranarray(self.bias2),
        #          linear3 = np.asfortranarray(self.linear3), bias3 = np.asfortranarray(self.bias3)
        #          )



if __name__ == '__main__':
    model_path = "model/feedforward_3.pt"

    models = torch.load(model_path, map_location='cpu')
    model_np = PolicyFeedforwardNp(47, 64, models)
    # 下面的代码用来判断新模型和旧模型输出是否一致
    model1: PolicyFeed = PolicyFeed.load(model_path)
    # vec = np.random.rand(1, 47)
    vec = np.ones(shape=(1, 47), dtype=np.float64) * 1
    print(model_np.forward(vec))
    vec1 = model1.scaler.transform(vec)
    result1 = model1(torch.from_numpy(vec1).float())
    print(result1)

    # model_np.save("model/feedforward")