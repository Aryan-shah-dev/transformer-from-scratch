from model import TransformerModel 
from data.tokenizer import CharTokenizer 
from data.dataset import TextDataset
import config 
import torch.nn as nn 
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
tokenizer = CharTokenizer("/Users/aryanshah/transformer-from-scratch/data/input.txt")
Dataset = TextDataset(tokenizer)
model = TransformerModel(tokenizer.vocab_size)
optimizer = optim.Adam(model.parameters() , lr = 1e-3)
criterion = nn.CrossEntropyLoss()

#main training loop lezgoooo 
losses = []
steps = range(50000) 
print(tokenizer.vocab_size)
for i in range(50000):
    x,y = Dataset.get_batch("train") 
    #do this or gradients will add up and accumulate  
    optimizer.zero_grad()
    #entire forward pass in one line :kekw: 
    logits = model(x) 
    #now get shapes 
    B,T,C = logits.shape 
    #flatten so you get all examples as answers 
    logits = logits.view(B*T,C)
    #to calculate losss
    y = y.view(B*T) 
    loss = criterion(logits,y) 
    loss.backward()
    optimizer.step()
    losses.append(loss.item()) 
    if i%100 == 0:
        print(f"current i = {i} : current loss = {loss.item()}")
plt.plot(steps,losses)
plt.show()