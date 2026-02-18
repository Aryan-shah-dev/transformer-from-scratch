from model import TransformerModel 
from data.tokenizer import CharTokenizer 
from data.dataset import TextDataset
import config 
import torch.nn as nn 
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
tokenizer = CharTokenizer(r"C:\Users\ARYAN SHAH\transformer-from-scratch\data\input.txt")
Dataset = TextDataset(tokenizer)
model = TransformerModel(tokenizer.vocab_size)
optimizer = optim.Adam(model.parameters() , lr = 1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#main training loop lezgoooo 
losses = []
steps = range(100000) 
print(tokenizer.vocab_size)
for i in range(100000):
    x,y = Dataset.get_batch("train") 
    x = x.to(device)
    y = y.to(device)
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
start_text = torch.tensor([tokenizer.encode("william")],dtype= torch.long ,device = device)
model.generate(start_text,5000,tokenizer)