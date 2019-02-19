# Deep-Variational-Information-Bottlenck

Here in this repository you can find a simple implementation of Deep variational Information Bottleneck. The idea of information bottleneck is very simple.

You have an input $X$, 
You have its corresponding labels $Y$,
You look for a latent variable (clearly much less dimension as $X$) $Z$ by maximizing mutual information between $Z$ and $Y$ ($I(Z, Y)$), meanwhile you have to minimize $I(Z, X)$. 

For some especial cases there are some closed form solutions but in general it is an NP-hard optimization. 

There are some approaches based on families of deep neural networks. Here I implemented this popular one. 
https://arxiv.org/abs/1612.00410

The code is based on Tensorflow! ;)



