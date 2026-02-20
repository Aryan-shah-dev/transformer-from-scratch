import torch
import config
class TextDataset:
    def __init__(self,tok):
        self.dataset = torch.tensor(tok.encode(tok.text),dtype = torch.long) 
        self.train = self.dataset[:int(len(self.dataset)*config.split_ratio)]
        self.val = self.dataset[int(len(self.dataset)*config.split_ratio):]
        self.block_size = config.block_size
        self.batch_size = config.batch_size
    def get_batch(self,split):
        if(split == "train"):
            data = self.train 
        else:
            data = self.val
        max_i = len(data) - self.block_size - 1
        index = torch.randint(0,max_i+1 , (self.batch_size,))
        batch_x = [data[i:i+self.block_size] for i in index]
        batch_x = torch.stack(batch_x)
        batch_y = [data[i+1:i+self.block_size + 1 ] for i in index]
        batch_y = torch.stack(batch_y)
        return batch_x,batch_y