import torch.nn as nn
import config
import torch
class TransformerModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,config.d_model) 
        self.pos_embedding = nn.Embedding(config.block_size, config.d_model) 
        self.Wq = nn.Linear(config.d_model,config.d_model)
        self.Wk = nn.Linear(config.d_model,config.d_model)
        self.Wv = nn.Linear(config.d_model,config.d_model)
        self.Ln1 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(nn.Linear(config.d_model,config.d_model*4)
                            ,nn.GELU()
                            ,nn.Linear(config.d_model*4,config.d_model))
        self.Ln2 = nn.LayerNorm(config.d_model)
        self.Ln3 = nn.LayerNorm(config.d_model) 
        self.Lm_head = nn.Linear(config.d_model , vocab_size) 
    def forward(self , idx):
        B,T = idx.shape #B = batch size , T = block size 
        #get embedding for each token in each batch 
        tok_emb = self.embedding(idx) 
        pos = torch.arange(T,device = idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        #attention noww 
        x = x + self.attention(self.Ln1(x))
        x = x + self.ff(self.Ln2(x)) 
        logits = self.Lm_head(self.Ln3(x)) 
        return logits 
    def attention(self,x):
        B,T,C = x.shape
        #attention noww 
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x) 
        #now the main attention logic
        scores = Q@ K.transpose(-2,-1) 
        #no shape is (B,T,T) 
        scores = scores/(config.d_model **0.5) 
        #scale by d_model **0.5 to have better dirstibution for gradients to update better 
        #if you dont scale then final softmax can look like [1,0,0,0] so bad gradient updates for the 0,0,0 because they dont know what went wrong 

        mask = torch.tril(torch.ones(T,T,device = x.device))
        #tril = lower triangle preserved 
        scores = scores.masked_fill(mask == 0 ,float('-inf'))
        #mask to -inf so current token cant peek to future 
        #i had the idea to just multiply matrices but e^0 is still 1 so theyll have probability 
        scores = torch.softmax(scores , dim = -1)
        #dim = -1 so each row will add to 1 in softmax 
        out = scores @ V 
        return out 