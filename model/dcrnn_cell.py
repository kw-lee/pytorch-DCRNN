from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from base import BaseModel


class DiffusionGraphConv(BaseModel):
    def __init__(self, input_dim, hid_dim, max_diffusion_step, output_dim, filter_type="random_walk", bias_start=0.0, sparse_supports=True):
        super(DiffusionGraphConv, self).__init__(filter_type=filter_type,
                                                 sparse_supports=sparse_supports)
        # NUM_MATRICES: ORDER
        self.num_matrices = self._len_supports * max_diffusion_step + 1  # Don't forget to add for x itself.
        input_size = input_dim + hid_dim
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)
        self._mm = torch.sparse.mm if self._sparse_supports else torch.mm

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state, output_size, supports):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs: (time_dim, batch_size, num_nodes, input_dim)
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        num_nodes = supports[0].shape[0]
        inputs = torch.reshape(inputs, (batch_size, num_nodes, -1))
        state = torch.reshape(state, (batch_size, num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2) # (batch_size, num_nodes, input_DIM + hidden_dim)
        input_size = inputs_and_state.shape[2]
        
        # dtype = inputs.dtype

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = self._mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * self._mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
   
        x = torch.reshape(x, shape=[batch_size * num_nodes, input_size * self.num_matrices])


        x = torch.matmul(x, self.weight)  # (batch_size * num_nodes, output_size)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, num_nodes * output_size])

class DCGRUCell(BaseModel):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, input_dim, num_units, max_diffusion_step,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True,  
                 filter_type="random_walk", 
                 sparse_supports=True):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__(filter_type=filter_type,
                                        sparse_supports=sparse_supports)
        self._activation = activation
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self._input_dim = input_dim

        # supports = utils.calculate_scaled_laplacian(adj_mat, lambda_max=None)  # scipy coo matrix
        # supports = self._build_sparse_matrix(supports).cuda()  # to pytorch sparse tensor

        self.dconv_gate = DiffusionGraphConv(input_dim=self._input_dim,
                                             hid_dim=num_units,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2,
                                             bias_start=1.0,
                                             filter_type=self._filter_type,
                                             sparse_supports=self._sparse_supports)
        self.dconv_candidate = DiffusionGraphConv(input_dim=self._input_dim,
                                                  hid_dim=num_units,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units,
                                                  bias_start=1.0,
                                                  filter_type=self._filter_type,
                                                  sparse_supports=self._sparse_supports)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)

    # @property
    # def output_size(self):
    #     output_size = self._num_nodes * self._num_units
    #     if self._num_proj is not None:
    #         output_size = self._num_nodes * self._num_proj
    #     return output_size

    def forward(self, inputs, state, supports):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        num_nodes = supports[0].shape[0]
        # we start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
            
        value = torch.sigmoid(fn(inputs, state, output_size, supports))
        value = torch.reshape(value, (-1, num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, num_nodes * self._num_units))
        u = torch.reshape(u, (-1, num_nodes * self._num_units))
        c = self.dconv_candidate(inputs, r * state, self._num_units, supports)  # batch_size, num_nodes * output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            output = torch.reshape(self.project(output), shape=(batch_size, -1))  # (50, 207*1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size, num_nodes, device=torch.device('cpu')):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, num_nodes * self._num_units, device=device)
