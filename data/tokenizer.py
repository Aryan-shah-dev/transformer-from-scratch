import os 
class CharTokenizer:
    def __init__ (self,dataset_path):
        with open(dataset_path,'r',encoding='utf-8') as file:
            #used utf-8 here to make sure no matter what input.txt contains it doesnt break 
            #because chars like é dont have ascii
            self.text = file.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        #map char to integer with stoi
        self.stoi = {}
        #map interger to char with itos 
        #would be cooler if called iots because like opposite of stoi but whatever lol 
        self.itos = {} 
        for i,j in enumerate(self.chars):
            self.stoi[j] = i
            self.itos[i] = j
    def encode(self,text):
        encoded = []
        for char in text:
            if char in self.stoi:
                encoded.append(self.stoi[char]) 
            else:
                raise ValueError (f"invalid character: {repr(char)} not in vocabulary")
        return encoded
    def decode(self,encoded_text):
        return "".join(self.itos[int(i)] for i in encoded_text) 