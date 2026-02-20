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

losses = []
steps = range(1000) 
print(tokenizer.vocab_size)

for i in range(1000):

    x,y = Dataset.get_batch("train") 
    x1,y1 = Dataset.get_batch("test")

    x1 = x1.to(device) 
    y1 = y1.to(device)
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(x)

    B,T,C = logits.shape 
    logits = logits.view(B*T,C)
    y = y.view(B*T) 

    loss = criterion(logits,y) 
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_val = model(x1)
        B1,T1,C1 = logits_val.shape
        logits_val = logits_val.view(B1*T1,C1)
        y1 = y1.view(B1*T1)
        loss1 = criterion(logits_val,y1)
    model.train()

    losses.append(loss.item()) 

    if i%100 == 0:
        print(f"current i = {i} : train loss = {loss.item()} | validation loss = {loss1.item()}")

start_text = torch.tensor([tokenizer.encode("william")],dtype= torch.long ,device = device)
model.generate(start_text,5000,tokenizer)
