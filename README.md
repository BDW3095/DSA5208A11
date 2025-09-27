## DSA5208 Project 1: Distributive SGD

### Usage (train and predict)

During preprocessing, the data is preprocessed and stored in `processedX.csv`, `processedy.csv` in the current working directory, which will be read by `SDG.py`.

`SGDfit.py` first initializes a neural network model with `1` hidden layer, reads the processed data collectively and then performs train-test split. Distributive SGD is applied to train the model on training sets. `main()` function keeps track of training loss `0` and reports test loss. 

In a `Python3.x` environment with Open MPI and dependent packages properly configured, `SGDfit.py` runs as follows: 

```
$ mpiexec -np nprocs python3.x SGDfit.py actv width lrate0 lrate1 nrandrows randseed threshold cycle
```
The parameter `nprocs` specifies the number of processes. Activation function `actv` must be one of `relu`, `sigmoid`. `width` specifies the number of neurons in the hidden layer. `lrate0` and `lrate1` specify the learning rate of weights in input layer and output layer, respectively. `nrandrows` specifies how many rows are sampled for each  stochastic gradient computation. `randseed` must be either `None` or an integer, as specifies the random seed. `threshold` specifies the terminating criterion. The fitting process terminates when successive training MSE differ less than `threshold`. Finally `cycle` specifies the frequency of MSE estimation in fitting process. When `cycle=5`, MSE is computed every `5` iterations of gradient descent. 