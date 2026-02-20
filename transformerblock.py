import config
import torch.nn as nn
import torch
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = config.num_heads 
        self.dims = config.d_model//self.heads 
        assert config.d_model% self.heads ==0 
        self.Wq = nn.Linear(config.d_model,config.d_model)
        self.Wk = nn.Linear(config.d_model,config.d_model)
        self.Wv = nn.Linear(config.d_model,config.d_model)
        self.Ln1 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(nn.Linear(config.d_model,config.d_model*4)
                            ,nn.GELU()
                            ,nn.Linear(config.d_model*4,config.d_model))
        self.Ln2 = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model,config.d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).bool()
            )
    def forward(self , x):
        x = x + self.attention(self.Ln1(x)) 
        x = x + self.ff(self.Ln2(x)) 
        return x 
    def attention(self,x):
        B,T,C = x.shape
        h = self.heads 
        d = self.dims 

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x) 

        Q = Q.view(B,T,h,d).transpose(1,2) 
        
        K = K.view(B,T,h,d).transpose(1,2)
        V = V.view(B,T,h,d).transpose(1,2)

        scores = Q@ K.transpose(-2,-1) 
        
        scores = scores/(d **0.5) 

        scores = scores.masked_fill(~self.mask[:T, :T] ,float('-inf'))

        scores = torch.softmax(scores , dim = -1)

        out = scores @ V 
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.proj(out) 
        return out 