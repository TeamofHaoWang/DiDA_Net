import sys
sys.path.append("..")
from torch.utils.data import Dataset
import numpy as np

#  ---------------------------- Dataset ----------------------------------
class XJTUData(Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index][-1].reshape(-1)

        return sample_x, sample_y

    def __len__(self):
        return len(self.data_x)

class XJTUData_index(Dataset):

    def __init__(self, data_x, basis_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.basis_x = basis_x

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index][-1].reshape(-1)
        basis_x = self.basis_x[index][:, np.newaxis]

        return sample_x, basis_x, sample_y

    def __len__(self):
        return len(self.data_x)


class XJTUData_index_PINN(Dataset):

    def __init__(self, data_x, basis_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.basis_x = basis_x

    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]
        basis_x = self.basis_x[index]

        return sample_x, basis_x, sample_y

    def __len__(self):
        return len(self.data_x)