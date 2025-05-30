import numpy as np
import torch

def load_recons_data(sampled_data):
    sampled_data = torch.as_tensor(sampled_data, dtype=torch.float32)
    flattened_sampled_data = []
    for i in range(sampled_data.shape[0]):
        for j in range(sampled_data.shape[1] - 2):
            flattened_sampled_data.append(sampled_data[i, j:j + 3, ...])
            # mask_lst.append(mask)

    flattened_sampled_data = torch.stack(flattened_sampled_data, dim=0)
    print(f'data shape: {flattened_sampled_data.shape}')

    return flattened_sampled_data
