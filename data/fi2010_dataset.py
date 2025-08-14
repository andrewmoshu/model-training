import torch
from torch.utils.data import Dataset
import numpy as np

class FI2010Dataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        
        # Load the pre-generated tensor
        self.data = torch.from_numpy(np.load('fi2010_data.npy')).float()
        
        self.shape_config = config.shape
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        n_context_p = self.shape_config.n_context + self.shape_config.n_prompt
        
        V_context = sequence[:self.shape_config.n_context, :]
        V_prompt = sequence[self.shape_config.n_context:n_context_p, :]
        
        t = torch.arange(sequence.shape[0]).float()

        T_context = t[:self.shape_config.n_context].unsqueeze(-1).unsqueeze(0)
        T_prompt = t[self.shape_config.n_context:n_context_p].unsqueeze(-1).unsqueeze(0)
        
        V_context = V_context.unsqueeze(0)
        V_prompt = V_prompt.unsqueeze(0)

        return (
            T_context, T_prompt, V_context, V_prompt,
            T_prompt, V_prompt, T_prompt, V_prompt,
            {}, {}
        )
