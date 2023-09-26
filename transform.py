import numpy as np
import torch

# transform 구현
class MyToTensor(object):
    def __call__(self, data):
        inputs, labels = data['inputs'], data['labels']
        
        inputs = inputs.transpose((2, 0, 1)).astype(np.float32)
        labels = labels.transpose((2, 0, 1)).astype(np.float32)
        
        # inputs = torch.from_numpy(inputs)
        # labels = torch.from_numpy(labels)
        
        data = {'inputs' : torch.from_numpy(inputs), 'labels' : torch.from_numpy(labels)}
        
        return data
        
class MyNormalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        inputs, labels = data['inputs'], data['labels']
        
        inputs = (inputs - self.mean) / self.std
        
        data = {'inputs': inputs, 'labels': labels}
        return data

class RandomFlip(object):
    def __call__(self, data):
        inputs, labels = data['inputs'], data['labels']
        
        # 50% 좌우 반전
        if np.random.rand() > 0.5:
            inputs = np.fliplr(inputs)
            labels = np.fliplr(labels)
            
        # 50% 상하 반전
        if np.random.rand() > 0.5:
            inputs = np.flipud(inputs)
            labels = np.flipud(labels)
            
        data = {'inputs': inputs, 'labels': labels}
        return data