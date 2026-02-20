import torch.nn as nn
import config
import torch
from transformerblock import TransformerBlock
class TransformerModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,config.d_model) 
        self.pos_embedding = nn.Embedding(config.block_size, config.d_model) 
        self.ln_f = nn.LayerNorm(config.d_model) 
        self.lm_head = nn.Linear(config.d_model , vocab_size) 
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.num_block)])
    def forward(self , idx):
        B,T = idx.shape 
        tok_emb = self.embedding(idx) 
        pos = torch.arange(T,device = idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        
        #attention noww 
        for block in self.blocks:
            x = block(x) 
        logits = self.lm_head(self.ln_f(x)) 
        return logits 
    @torch.no_grad()
    def generate(self , idx ,max_generation , tokenizer):
        for _ in range(max_generation):
            temp_idx = idx[:,-config.block_size:]   
            logits = self(temp_idx) 
            logits = logits [:,-1,: ]/config.temperature
            probs = torch.softmax(logits,dim=-1)
            next_token = torch.multinomial(probs,num_samples=1) 
            idx = torch.cat((idx,next_token) , dim=1) 
        tokens = idx[0].tolist()
        text = tokenizer.decode(tokens)
        print(text)
        return idx
