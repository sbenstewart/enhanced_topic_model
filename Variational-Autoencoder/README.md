##Variational Auto-encoder

This is an improved implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling. This code uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

To run the MNIST experiment:

`python run.py`

Setting the continuous boolean to true will make the script run the freyfaces experiment. It is necessary to tweak the batch_size and learning rate parameter for this to run smoothly.


The code is MIT licensed.
