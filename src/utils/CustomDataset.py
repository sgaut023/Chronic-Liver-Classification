from torch.utils.data import DataLoader, Dataset
import torch


class CustomDataset(Dataset):
   def __init__(self, data, label):
        self.data = data
        self.label = torch.Tensor(label.values)# float
        
   def __len__(self):
        return len(self.label)
    
   def __getitem__(self, index):
        target = self.label[index].view(1,)
        data = self.data[index].view(1, 1, 217, 54, 79)
        return data,target

