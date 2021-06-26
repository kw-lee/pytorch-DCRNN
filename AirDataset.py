import torch 
import numpy as np 
from copy import deepcopy

def sample_z(*size, device=torch.device('cpu'), seed_scale=0.01):
    return torch.rand(*size, device=device) * seed_scale

# custom datset 
class AirDataset(torch.utils.data.Dataset):
    """Airkorea + KMA dataset"""
    def __init__(self, training_set, time_dim, n_o_n_m, n_m, seed_scale, PM25_idx, missing_time=1):
        self.dataset = training_set
        self.time_dim = time_dim
        self.n_o_n_m = n_o_n_m
        self.n_m = n_m
        self.seed_scale = seed_scale
        self.PM25_idx = PM25_idx
        self.missing_time = missing_time # self.time_dim // 2
    
    def __len__(self):
        return self.dataset.shape[1] - self.time_dim

    def __getitem__(self, idx):
        output = torch.from_numpy(self.dataset[:, idx:idx+self.time_dim, :]).float()
        input = deepcopy(output)

        s_missing = np.random.choice(self.n_o_n_m-1, size=self.n_m, replace=False)
#         t_missing = np.random.choice(self.time_dim, size=self.missing_time, replace=False)
        input[:, -self.missing_time:, :] = float('nan')
        input[:, :, s_missing] = float('nan')
        # (var_dim, time_dim, space_dim)
        # input[self.PM25_idx, half_time:, :] = float('nan')

        # where are not NAs
        input_omask = torch.ones_like(input)
        output_omask = torch.ones_like(output)
        input_omask[torch.isnan(input)] = 0
        output_omask[torch.isnan(output)] = 0

        input[torch.isnan(input)] = 0
        output[torch.isnan(output)] = 0

        # Normalize the value according to experience
        M_mb = input_omask # more hide
        Mf_input = input * M_mb # most hidden
        # less hidden mask (our information)

        # sampling
        Z_mb = sample_z(*Mf_input.shape, seed_scale=self.seed_scale)
        Mf_input = M_mb * Mf_input + (1 - M_mb) * Z_mb

        return Mf_input, M_mb, output, output_omask