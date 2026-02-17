# transformer-from-scratch

This is my personal project to deeply understand how Transformers work by implementing them step by step from scratch.

Instead of using high-level libraries, the goal is to gradually build and scale each component while observing how architectural changes affect training dynamics and model behavior.

---

## Tokenization

- Currently using character-level encoding  
- Going to upgrade to BPE soon  

---

## Model Architecture

- Embedding (positional + token)  
- Multi-head attention  
- Feedforward network  
- Residual connections + a bunch of LayerNorms  
- Soon will stack transformer blocks  

---

## Goals

- Learn mechanics of a transformer at the tensor level  
- Understand the impact of depth on results through testing  
- Test scaling laws on a small scale by moving values in `config.py`  
- Eventually scale to a deep network with roughly 1B params on cloud GPUs (my Mac could never 💀)  

---

## Dataset

Currently using Tiny Shakespeare:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt