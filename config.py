#all global variables here so easier to change everyhwere 
#will add the stuff i am using rn the other stuff comes later 
block_size = 256 #sees last 128 tokens when generating 
batch_size = 64 #how many sequences train parallely 
split_ratio = 1 #split between train:validation 
d_model = 256 #size of embedding per token
num_heads = 8 #attention heads 
temperature = 0.7#works for probability distribution <1 helps amplify top probabilities >1 flattens 