## DSA5208 Project 1: Distributive SGD

### Usage (train and predict)

During preprocessing, the data is preprocessed and stored in `Xtrain.csv`, `ytrain.csv`, `Xtest.csv`, `ytest.csv` in directory `./processedData`, which will be read by `SDG.py`.

`SGDfit.py` first initializes a neural network model with one hidden layer, reads the processed data collectively and then performs train-test split. Distributive SGD is applied to train the model on training sets. `trainAndReport()` function trains a model with input parameters and reports training loss history and test loss in a tuple `([numpy array], [numpy float])`. `main()` funtion trains the model with parameter specified in terminal without reporting.  

In a `Python3.x` environment with Open MPI and dependent packages properly configured, `SGDfit.py` can be directly run as follows: 

```
$ mpiexec -np nprocs python3.x SGDfit.py actv width lrate0 lrate1 nrandrows randseed threshold cycle
```
*The same parametric structure is seen in the `params` argument passed to `trainAndReport(comm, nprocs, rank, Xtrain, ytrain, Xtest, ytest, params)`*

The parameter `nprocs` specifies the number of processes. Activation function `actv` must be one of `softplus`, `sigmoid`. `width` specifies the number of neurons in the hidden layer. `lrate0` and `lrate1` specify the learning rate of weights in input layer and output layer, respectively. `nrandrows` specifies how many rows are sampled for each  stochastic gradient computation. `randseed` must be either `None` or an integer, as specifies the random seed. `threshold` specifies the terminating criterion. The fitting process terminates when successive training MSE differ less than `threshold`. Finally `cycle` specifies the frequency of MSE estimation in fitting process. When `cycle=5`, MSE is computed every `5` iterations of gradient descent. 