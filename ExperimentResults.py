from SGDfit import *
import json

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank   = comm.Get_rank()

# actvList  = ["relu", "tanh", "sigmoid"]
# batchSizeList = [32, 64, 128, 256, 512, 1024]
# hiddenDimList = [32, 64, 128, 256, 512]

actvList = ["relu"]
batchSizeList = [32]
hiddenDimList = [16]

dfResult = None
lossHistoryDict = dict()

if rank == 0:
    dfResult = pd.DataFrame(columns = ["activation function", "hidden dimension",  
                                       "batch size",  "training RMSE", "testing RMSE", "time 1", "time 2", "seed"])


lr0 = .001
lr1 = .001

seed= 5208
threshold = .5

lr= (lr0, lr1)
mseCycle = 100

totalSize = len(actvList)* len(hiddenDimList)* len(batchSizeList)

modelList = [None] * totalSize
paramList = [None] * totalSize
testrmseList = [.0] * totalSize

i = 0
for actv in actvList:
    for batchSize in batchSizeList:
        for hDim  in hiddenDimList:
            paramList[i] = [actv, hDim, batchSize]
            i += 1

inputDirectory  =  './processedData/'
outputDirectory =  './trainingOutcome/'

rowLimit = None

if __name__ == '__main__':

    Xtrain = collectiveRead(inputDirectory+'Xtrain.csv', comm, nprocs, rank, rowLimit)
    ytrain = collectiveRead(inputDirectory+'ytrain.csv', comm, nprocs, rank, rowLimit)
    Xtest  = collectiveRead(inputDirectory+'Xtest.csv' , comm, nprocs, rank, rowLimit)
    ytest  = collectiveRead(inputDirectory+'ytest.csv' , comm, nprocs, rank, rowLimit)


    for i, par in enumerate(paramList):
        modelList[i] = nn1Layer(comm, nprocs, rank, Xtrain.shape[1], par[1], par[0])

    for i, par in enumerate(paramList):
        timeElapsed , lossHistory = np.empty(2, np.float32) , list()
        modelList[i].fit(Xtrain, ytrain, lr, par[2], seed, threshold, 
                        mseCycle, timeElapsed, lossHistory)
        lossHistoryDict[i]= [float(_) for _ in lossHistory]
        testMSE  = modelList[i].calculateLoss(Xtest, ytest)
        if rank == 0:
            dfResult.loc[i] = par + [np.sqrt(lossHistory[-1]), np.sqrt(testMSE), timeElapsed[0], timeElapsed[1], seed]

    if rank == 0:

        for i, nn in enumerate(modelList):
            np.savetxt(outputDirectory+f"w0_{i:02d}.csv".format(i), nn.w0, delimiter=",")
            np.savetxt(outputDirectory+f"w1_{i:02d}.csv".format(i), nn.w1, delimiter=",")


        with open(outputDirectory+"lossHistory.json", "w") as fh:
            json.dump(lossHistoryDict, fh)
        with open(outputDirectory+"paramList.json"  , "w") as fh:
            json.dump(paramList      , fh)

        dfResult.to_csv(outputDirectory+ "experimentResults.csv")
