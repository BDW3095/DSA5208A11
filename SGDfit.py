from mpi4py import MPI
import numpy  as np
import pandas as pd
import csv
import sys

from sklearn.model_selection import train_test_split

class ActivationFunction:
    def __init__(self, func, grad):
        self.func = func
        self.grad = grad

b, k = 10, 0.2

def sigmoid_f(x):
    x = np.clip(x,-b, b)
    return 1 / (1+ np.exp(-x))

def sigmoid_g(x):
    x = np.clip(x,-b, b)
    return np.exp(-x) / (1 + np.exp(-x)) **2

def relu_f(x):
    return np.maximum(x, k *x)

def relu_g(x):
    return np.piecewise(x, [x<0, x>=0], [k, 1.0])

def tanh_f(x):
    return np.tanh(x)

def tanh_g(x):
    return 1 - (np.tanh(x)) **2

ActivationDict = dict()

ActivationDict['sigmoid'] = ActivationFunction(sigmoid_f, sigmoid_g)
ActivationDict['relu'] =    ActivationFunction(relu_f,  relu_g)
ActivationDict['tanh'] =    ActivationFunction(tanh_f,  tanh_g)

class nn1Layer():
    
    def __init__(self, nFeatures, width, actvFuncName):
        self.sigma = ActivationDict[actvFuncName]
        self.isReLU = bool(actvFuncName== 'relu')
        self.width = width
        self.f = nFeatures
        self.w0 = np.zeros((self.f+1, width), dtype=np.float32)
        self.w1 = np.zeros((width+1, 1),      dtype=np.float32)

    def forwardBroadcast(self, X, fit=False, _hiddenLayerRecv=None, _hiddenLayerActv=None):
        nRows = X.shape[0]
        yHat = np.zeros((nRows, 1), dtype=np.float32)

        if not fit:
            _hiddenLayerRecv = np.zeros((nRows, self.width), dtype=np.float32)
            _hiddenLayerActv = np.zeros((nRows, self.width), dtype=np.float32)

        _hiddenLayerRecv[:,:]= np.matmul(X, self.w0[:self.f,:])+ self.w0[-1,:]
        _hiddenLayerActv[:,:] = self.sigma.func(_hiddenLayerRecv)
        yHat[:]= np.matmul(_hiddenLayerActv, self.w1[:self.width])+self.w1[-1]

        return yHat


    def calculateLoss(self, X, y, comm, nprocs, rank):
        yDelta = self.forwardBroadcast(X) - y
        sse    = np.zeros(1,  np.float32) 
        sse[0] = np.sum(yDelta **2)
        sst = None
        comm.Barrier()
        if rank == 0:
            sst =np.empty(nprocs, np.float32)
        comm.Gather(sse, sst, root= 0)
        mse = None
        if rank == 0:
            mse = np.sum(sst) / X.shape[0] / nprocs
            # print('{:07.2f}'.format(mse))
            print(mse)
        return mse

    
    def initializeWeights(self, comm, rank, seed=None):
        if rank == 0:
            np.random.seed(seed)
            bound0 = np.sqrt(6 / (self.f+ self.width))
            bound1 = np.sqrt(6 / (self.width+ 1))
            self.w0[:,:]= np.random.uniform(-bound0, bound0, self.w0.shape)
            self.w1[:,:]= np.random.uniform(-bound1, bound1, self.w1.shape)
        comm.Bcast(self.w0, root=0)
        comm.Bcast(self.w1, root=0)
        np.random.seed()


    def fit(self, Xtrain, ytrain, learningRates, M, comm, nprocs, rank, seed, threshold, T, lossTrack=[], gradBound=4):
        
        localGrad0 = np.zeros(self.w0.shape, dtype=np.float32)
        localGrad1 = np.zeros(self.w1.shape, dtype=np.float32)

        _hiddenLayerRecv= np.zeros((M,self.width), dtype=np.float32)
        _hiddenLayerActv= np.zeros((M,self.width), dtype=np.float32)

        self.initializeWeights(comm, rank, seed)

        mse = self.calculateLoss(Xtrain, ytrain, comm, nprocs, rank)
        if rank == 0:
            lossTrack.append(mse)
        period = 0
        convergence = 0
        while not convergence:
            period += 1
            indices= np.random.randint(Xtrain.shape[0], size=M)
            Xselect, yselect = Xtrain[indices], ytrain[indices]

            yDelta= self.forwardBroadcast(Xselect, True, _hiddenLayerRecv, _hiddenLayerActv)- yselect

            hiddenLayerBcast = np.transpose(self.w1[:self.width]) * self.sigma.grad(_hiddenLayerRecv)
            localGrad0[:self.f,:] = np.matmul(np.transpose(yDelta * Xselect), hiddenLayerBcast)
            localGrad1[:self.width,:]= np.matmul(np.transpose(_hiddenLayerActv), yDelta)
            localGrad0[-1,:] = np.matmul(np.transpose(yDelta), hiddenLayerBcast)
            localGrad1[-1,:] = np.sum(yDelta)

            localGrad0= localGrad0 / M / nprocs
            localGrad1= localGrad1 / M / nprocs

            if self.isReLU:
                
                localGrad0 = np.clip(localGrad0, -gradBound, gradBound)
                localGrad1 = np.clip(localGrad1, -gradBound, gradBound)

            buffer0 = np.empty((nprocs,) +localGrad0.shape, np.float32)
            buffer1 = np.empty((nprocs,) +localGrad1.shape, np.float32)

            comm.Allgather(localGrad0, buffer0)
            comm.Allgather(localGrad1, buffer1)

            self.w0-= learningRates[0]* np.sum(buffer0, axis=0)
            self.w1-= learningRates[1]* np.sum(buffer1, axis=0)

            if period == T:
                mse =self.calculateLoss(Xtrain, ytrain, comm, nprocs, rank)

                if rank == 0:
                    if np.abs(lossTrack[-1] - mse) < threshold:
                        convergence = 1
                    else:
                        lossTrack.append(mse)
                convergence = comm.bcast(convergence, root=0)

                period = 0
        
        return None

def getnLines(fname):
    with open(fname, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        nLines = sum(1 for _ in csv_reader)
    return nLines

def collectiveRead(fname, comm, nprocs, rank):

    numOfLines = getnLines(fname)

    localSize = int(numOfLines/nprocs)
    numOfAttributes = np.empty(1, 'i')
    if rank == 0:
        head= pd.read_csv(fname, nrows=10)
        numOfAttributes[0] = head.shape[1]

    comm.Bcast([numOfAttributes, MPI.INT], root= 0)   


    localData = np.zeros((localSize,numOfAttributes[0]), dtype=np.float32)

    if rank == 0:
        dataChunks = pd.read_csv(fname, chunksize=localSize, index_col=False)

        for r, chunk in enumerate(dataChunks):
            sendData = chunk.to_numpy(dtype=np.float32)
            if r == 0:
                localData = sendData
            else:
                comm.Send([sendData, MPI.FLOAT], r)
    else:
        comm.Recv([localData, MPI.FLOAT], source=0)
    
    return localData

def main():

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    args = sys.argv

    actvFName = str(args[1])
    width     = int(args[2])
    lr        = float(args[3]), float(args[4])
    M         = int(args[5])
    seed = None
    try: 
        seed  = int(args[6])
    except ValueError:
        pass
    threshold = float(args[7])

    T         = int(args[8])

    X = collectiveRead('processedX.csv', comm, nprocs, rank)
    y = collectiveRead('processedy.csv', comm, nprocs, rank)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)
    nFeatures = X.shape[1]
    nn = nn1Layer(nFeatures, width, actvFName)
    lossTrack = list()

    nn.fit(Xtrain, ytrain, lr, M, comm, nprocs, rank, seed, threshold, T, lossTrack)

    predictionMeanSquareError = nn.calculateLoss(Xtest, ytest, comm, nprocs, rank) # stored only on root 0

if __name__ == '__main__':

    main()