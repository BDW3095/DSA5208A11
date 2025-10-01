from mpi4py import MPI
import numpy  as np
import pandas as pd
import time
import csv
import sys


class ActivationFunction:
    def __init__(self, func, grad):
        self.func = func
        self.grad = grad

ActivationDict   = dict()

class nn1Layer():
    
    def __init__(self, comm, nprocs, rank, inputDim, hiddenDim, actv):
        """Initialize the neural network with input & hidden size and the type of activation function. """
        self.sigma = ActivationDict[actv]
        self.comm = comm
        self.nprocs = nprocs
        self.rank = rank
        self.unboundedGrad = bool(actv in ['softplus', 'relu'])
        self.hDim =hiddenDim
        self.iDim = inputDim

        self.w0 = np.zeros((self.iDim+1, self.hDim), dtype=np.float32)
        self.w1 = np.zeros((self.hDim+1, 1),         dtype=np.float32)

    def forwardBroadcast(self, X, fit=False, _hiddenLayerRecv=None, _hiddenLayerActv=None):
        """Broadcast data X through the network while keeping track of middle values when 'fit' is set to 'True'"""
        nRows = X.shape[0]
        yHat = np.zeros((nRows, 1), dtype=np.float32)

        if not fit:
            _hiddenLayerRecv = np.zeros((nRows, self.hDim), dtype=np.float32)
            _hiddenLayerActv = np.zeros((nRows, self.hDim), dtype=np.float32)

        _hiddenLayerRecv[:,:]= np.matmul(X, self.w0[:self.iDim,:])+ self.w0[-1,:]
        _hiddenLayerActv[:,:]= self.sigma.func(_hiddenLayerRecv)
        yHat[:] = np.matmul(_hiddenLayerActv, self.w1[:self.hDim])+ self.w1[-1,:]

        return yHat

    def calculateLoss(self, X, y):
        """
        Input: data X, data y. Return the loss of current network. 
        """
        yDelta= self.forwardBroadcast(X) - y
        sse = np.zeros(1, np.float32) 
        sse[0] = np.sum(yDelta**2)
        sst = None
        if self.rank == 0:
            sst = np.empty(self.nprocs, np.float32)
        self.comm.Gather(sse, sst, root=0)
        
        mse = None
        if self.rank == 0:
            mse = np.sum(sst) / X.shape[0]/ self.nprocs
            print('{:07.5f}'.format(mse))
        return mse

    def initializeWeights(self, seed=None):
        """
        Initialize the weights of the neural network with normalized Xavier initialization. 
        """
        if self.rank == 0:
            np.random.seed(seed)
            bound0 = np.sqrt(6 / (self.iDim+ self.hDim))
            bound1 = np.sqrt(6 / (self.hDim+ 1))
            self.w0[:,:]= np.random.uniform(-bound0, bound0, self.w0.shape)
            self.w1[:,:]= np.random.uniform(-bound1, bound1, self.w1.shape)
        self.comm.Bcast(self.w0, root=0)
        self.comm.Bcast(self.w1, root=0)
        np.random.seed()

    def stochasticGradient(self, Xtrain, ytrain, batchSize , gradBound=20):

        localGrad0 = np.zeros(self.w0.shape, dtype=np.float32)
        localGrad1 = np.zeros(self.w1.shape, dtype=np.float32)

        idx=np.random.randint(Xtrain.shape[0], size=batchSize)
        Xselect, yselect= Xtrain[idx], ytrain[idx]

        _hLayerRecv= np.zeros((batchSize,self.hDim), dtype=np.float32)
        _hLayerActv= np.zeros((batchSize,self.hDim), dtype=np.float32)

        yDelta = self.forwardBroadcast(Xselect, True, _hLayerRecv, _hLayerActv)- yselect
        hLayerBcast =   np.transpose(self.w1[:self.hDim]) * self.sigma.grad(_hLayerRecv)

        localGrad0[:self.iDim,:] = np.matmul(np.transpose(yDelta* Xselect), hLayerBcast)
        localGrad1[:self.hDim,:] = np.matmul(np.transpose(_hLayerActv), yDelta)
        localGrad0[-1,:] = np.matmul(np.transpose(yDelta), hLayerBcast)
        localGrad1[-1,:] = np.sum(yDelta)

        localGrad0= localGrad0/ batchSize
        localGrad1= localGrad1/ batchSize

        buffer0 = np.empty((self.nprocs,)+localGrad0.shape, np.float32)
        buffer1 = np.empty((self.nprocs,)+localGrad1.shape, np.float32)

        self.comm.Allgather(localGrad0, buffer0)
        self.comm.Allgather(localGrad1, buffer1)

        grad0 =np.sum(buffer0, axis=0) /self.nprocs
        grad1 =np.sum(buffer1, axis=0) /self.nprocs

        if self.unboundedGrad:

            grad0 = np.clip(grad0,-gradBound, gradBound)
            grad1 = np.clip(grad1,-gradBound, gradBound)

        return  grad0 , grad1

    def fit(self, Xtrain, ytrain, learningRates, batchSize, seed, threshold, cycle, timeElapsed=np.empty(2, np.float32), lossTrack=[], timestampMse=3.0):
        """
        Train the model with SGD and store loss history & training time to references passed in. 
        Input: comm configs, training set, learning rates, width, initial params random seed, 
               terminate threshold, mse record cycle, reference (size-2-array) to store training
               times, reference (list) to store training history, target mse whose first-reached 
               training time will be recorded in pos 0 of the size-2-array. 
        Return: None
        """
        
        self.initializeWeights( seed)

        grad0 = np.zeros(self.w0.shape, dtype=np.float32)
        grad1 = np.zeros(self.w1.shape, dtype=np.float32)

        initmse = self.calculateLoss(Xtrain, ytrain)

        if self.rank == 0:
            lossTrack.append(initmse)

        s = 0
        t = 0
        l =16
        tolerance = int(l *threshold)
        convergence = 0
        timeRecorded= 0
        timeStart = time.time()
        timeTerminate = float()
        timeTargetMSE = float()

        monoIndicator = np.zeros(l , np.int8)

        while not convergence:
            t += 1

            grad0[:,:], grad1[:,:] = self.stochasticGradient(Xtrain, ytrain, batchSize)

            self.w0-= learningRates[0]* grad0
            self.w1-= learningRates[1]* grad1

            if t == cycle:

                mse =  self.calculateLoss(Xtrain, ytrain)
                if self.rank == 0:
                    monoIndicator[s] = lossTrack[-1]<=mse
                    s = (s+1) % l
                    if np.sum(monoIndicator)>= tolerance:
                        convergence = 1
                    else:
                        lossTrack.append(mse)

                    if not timeRecorded:
                        if mse< timestampMse:
                            timeRecorded =  1
                            timestampMse = time.time()

                t = 0
                convergence= self.comm.bcast(convergence, root=0)

        if self.rank == 0:
            timeElapsed[0] = timeTargetMSE - timeStart
            timeElapsed[1] = timeTerminate - timeStart
            
        return   None

def getnLines(fname):
    """Get num of rows of large .csv files"""
    with open(fname, 'r') as f:
        csv_reader= csv.reader(f)
        next(csv_reader)
        numOfRows = sum(1 for _ in csv_reader)
    return numOfRows

def collectiveRead(fname, comm, nprocs, rank):
    """
    Read in large .csv numerical data files collectively. 
    """
    # numOfRows = getnLines(fname)
    numOfRows = 700000 # small size to debug

    localSize = int(numOfRows/ nprocs)
    numOfAttributes = np.empty(1, 'i')
    if rank == 0:
        head= pd.read_csv(fname, nrows=10)
        numOfAttributes[0] = head.shape[1]

    comm.Bcast([numOfAttributes, MPI.INT], root= 0)   


    localData = np.zeros((localSize,numOfAttributes[0]), dtype=np.float32)
    sendData  = np.zeros((localSize,numOfAttributes[0]), dtype=np.float32)

    if rank == 0:
        dataChunks = pd.read_csv(fname, chunksize=localSize, index_col=False, nrows=numOfRows)

        for r, chunk in enumerate(dataChunks):
            sendData[:,:] = chunk.to_numpy(dtype=np.float32)
            if r == 0:
                localData[:,:] = sendData[:,:]
            elif r < nprocs:
                comm.Send(sendData, r)
            else:
                break
    else:
        comm.Recv(localData, source=0)
    
    return  localData

def trainAndReport(comm, nprocs, rank, Xtrain, ytrain, Xtest, ytest, params):
    """
    Train the model on training data with parameters params, evaluate loss on test data and returns a tuple 
    (trainning loss history list, test loss, train time) as   ([numpy array], [numpy float], [numpy array]) 
    """
    ### params: [actv, width, lrates, nrandrows, randseed, threshold, cycle]
    actv      = str(params[0])
    hiddenDim = int(params[1])
    lr        = float(params[2]), float(params[3])
    batchSize = int(int(params[4]) / nprocs)
    seed = None
    try: 
        seed  = int(params[5])
    except ValueError:
        pass
    threshold = float(params[6])
    mseCycle  = int(params[7])
    inputDim  = Xtrain .shape[1]

    nn  = nn1Layer(comm, nprocs, rank, inputDim, hiddenDim, actv)
    timeElapsed = np.empty(2, np.float32)
    lossTrack = list()

    nn.fit(Xtrain, ytrain, lr, batchSize, seed, threshold, mseCycle, timeElapsed, lossTrack)


    return lossTrack, nn.calculateLoss(Xtest, ytest), timeElapsed

b, k = 10, 0.01

def sigmoid_f(x):
    x = np.clip(x,-b, b)
    return 1 / (1+ np.exp(-x))

def sigmoid_g(x):
    x = np.clip(x,-b, b)
    return np.exp(-x) / (1 + np.exp(-x)) **2

def softplusf(x):
    x = np.clip(x,-b, b)
    return np.log(1+np.exp(x))

def softplusg(x):
    x = np.clip(x,-b, b)
    return 1 / (1+ np.exp(-x))

def relu_f(x):
    return np.maximum(x, k *x)

def relu_g(x):
    return np.piecewise(x, [x<0, x>=0], [k, 1.0])

def tanh_f(x):
    return np.tanh(x)

def tanh_g(x):
    return 1 - (np.tanh(x)) **2

ActivationDict['sigmoid' ] = ActivationFunction(sigmoid_f, sigmoid_g)
ActivationDict['tanh']     = ActivationFunction(tanh_f,  tanh_g)
ActivationDict['relu']     = ActivationFunction(relu_f,  relu_g)
ActivationDict['softplus'] = ActivationFunction(softplusf, softplusg)


def main():

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    args = sys.argv
     
    dataDirectory = 'processedData/'
    # dataDirectory = '/mnt/d/test/25_09/'

    Xtrain = collectiveRead(dataDirectory+'Xtrain.csv', comm, nprocs, rank)
    ytrain = collectiveRead(dataDirectory+'ytrain.csv', comm, nprocs, rank)
    Xtest  = collectiveRead(dataDirectory+'Xtest.csv' , comm, nprocs, rank)
    ytest  = collectiveRead(dataDirectory+'ytest.csv' , comm, nprocs, rank)

    trainLossHistory, testLoss, trainTime = trainAndReport(comm, nprocs, rank, Xtrain, ytrain, Xtest, ytest, args[1:])

if __name__ == '__main__':

    main()