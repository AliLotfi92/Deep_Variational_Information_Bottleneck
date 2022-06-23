# Deep Variational Information Bottleneck

This repository provides the implementation of Deep Variational Information Bottleneck. The main idea of DVIB is to impose a bottleneck (here in the dimensionality) through which only necessary information for the reconstruction of $X$ can pass. I tried to implement this in the simplest from so that _Information Bottleneck_ can be easily leveraged as a regularizer or metric for other projects.

### Requirements
- $X$ is the input, 
- $Y$ is the label,
- We look for a latent variable $Z$ that maximizes the mutual information $I(Z;Y)$, meanwhile, it has to minimize $I(Z;X)$. 
- For more detials and theoritical proofs please check https://arxiv.org/abs/1612.00410

### How to run
```bash
python VIBV4.py
```

