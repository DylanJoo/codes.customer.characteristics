import numpy as np
from torch.utils.data import Dataset, DataLoader

class InvoiceDataset(Dataset):

    def __init__(self, record_arr, user_id_arr, month_arr):
        self.input = record_arr
        self.user = user_id_arr
        self.month = month_arr

    def from_csv(self):
        pass

    def __getitem__(self, index):
        x_user = self.user[index]
        x_input = self.input[index] # list of input (purchasing items)
        return {'user_ids': x_user.astype(np.int32), 'user_inputs': x_input.astype(np.float32)}

    def __len__(self):
        return len(self.input)
