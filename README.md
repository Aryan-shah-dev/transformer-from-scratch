# transformer-from-scratch
This is my personal project to deeply understand how Transformers work by implementing them step by step from scratch.

Instead of using high-level libraries, the goal is to gradually build and scale each component while observing how architectural changes affect training dynamics and model behavior.

Tokenization : 
. currently using character level encoding 
. going to upgrade to BPE soon 

Model architecture :
. embedding (positional + token )
. multi head attention 
. feedforward network
. residual connection + a bunch of layernorms 
. soon will stack transformers 

Goals :
. learn mechanics of a transformer at the tensor level 
. understanding impact of depth on results through testing 
. testing scaling laws on a small scale by moving values in config.py 
. eventually scale to a deep network with roughly 1B params on cloud gpus (my mac could never :skull:)

Dataset :
. currently using tiny shakesphere 
link - https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

