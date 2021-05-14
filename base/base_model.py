import torch.nn as nn
import torch 
import numpy as np
from lib import utils
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, supports, filter_type, adj_mat):
        super(BaseModel, self).__init__()
        self._supports = supports
        self._filter_type = filter_type
        self._adj_mat = adj_mat
        
        if self._supports is None:
            self._supports = []
            supports = []
            if self._filter_type == "laplacian":
                supports.append(utils.calculate_scaled_laplacian(self._adj_mat, lambda_max=None))
            elif self._filter_type == "random_walk":
                supports.append(utils.calculate_random_walk_matrix(self._adj_mat).T)
            elif self._filter_type == "dual_random_walk":
                supports.append(utils.calculate_random_walk_matrix(self._adj_mat))
                supports.append(utils.calculate_random_walk_matrix(self._adj_mat.T))
            else:
                supports.append(utils.calculate_scaled_laplacian(self._adj_mat))
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))

            self._supports = nn.Parameter(torch.stack(self._supports))
            self._supports.requires_grad = False

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

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
