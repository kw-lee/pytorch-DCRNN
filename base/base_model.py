import torch.nn as nn
import torch 
import numpy as np
from lib import utils
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, filter_type="random_walk", sparse_supports=True):
        super(BaseModel, self).__init__()
        self._sparse_supports = sparse_supports
        self._filter_type = filter_type
        
        if self._filter_type == "dual_random_walk":
            self._len_supports = 2
        elif self._filter_type == "random_walk":
            self._len_supports = 1
        elif self._filter_type == "laplacian":
            self._len_supports = 1
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
